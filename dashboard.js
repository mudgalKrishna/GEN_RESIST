import { sbSaveUpload, sbLoadTheme, sbSaveTheme, onAuthStateChange, sbFetchFileText, supabase, sbLogout } from './supabase.js';

let isDark = false;
const lightTheme = {
    "--divBg": "#F7F7F7",
    "--Bg": "white",
    "--text-muted": "#888888",
    "--text-color": "#000",
    "--gradient": "linear-gradient(135deg, #21422E, #407E56)",
    "--shortcut": "#E9E9E9"
};

const darkTheme = {
    "--divBg": "#1c1c1c",
    "--Bg": "#121212",
    "--text-muted": "#bbbbbb",
    "--text-color": "#ffffff",
    "--gradient": "linear-gradient(135deg, #0f2218, #1f4732)",
    "--shortcut": "#272727ff"
};

function applyTheme(themeName) {
    const root = document.documentElement;
    const themeObj = themeName === 'dark' ? darkTheme : lightTheme;
    Object.keys(themeObj).forEach(key => root.style.setProperty(key, themeObj[key]));
    isDark = (themeName === 'dark');
}

const cachedTheme = localStorage.getItem('theme') || 'light';
applyTheme(cachedTheme);
// Auth Check
// In dashboard.js

onAuthStateChange(async (user) => {
    if (!user) {
        window.location.href = 'login.html';
        return;
    }

    const dbTheme = await sbLoadTheme(user.id);

    if (dbTheme && dbTheme !== cachedTheme) {
        console.log("Syncing theme from DB:", dbTheme);
        applyTheme(dbTheme);
        localStorage.setItem('theme', dbTheme); // Update cache for next time
    }

    // 2. FILL USER INFO (Add this part)
    const nameEl = document.getElementById("name");
    const emailEl = document.getElementById("email");

    if (nameEl) {
        // We look in user_metadata first (fastest), fallback to "User"
        nameEl.textContent = user.user_metadata?.name || "User";
    }

    if (emailEl) {
        emailEl.textContent = user.email;
    }
});

// network datasets (GLOBAL)
let nodes = new vis.DataSet([]);
let edges = new vis.DataSet([]);

// dashboard.js
(() => {
    // ---------- Config ----------
    const BACKEND_URL = "https://icybear02-gen-resist.hf.space/predict"; // update if needed

    // ---------- DOM refs ----------
    const uploadBtn = document.querySelector('.topNav .upload');
    const barsCont = document.querySelector('.barsCont');
    const genesCont = document.querySelector('.genesCont');
    const antibioticCont = document.querySelector('.antibioticCont');
    const visStat = document.querySelector('.visStat');

    // find smallStat by heading text (robust to DOM order)
    function getSmallStatEl(headingText) {
        return Array.from(document.querySelectorAll('.smallStat')).find(s => {
            const h = s.querySelector('.statHeading');
            return h && h.textContent.trim().toLowerCase() === headingText.trim().toLowerCase();
        });
    }

    const totalAntEl = getSmallStatEl('Total Antibiotics');
    const resistantAntEl = getSmallStatEl('Resistance Antibiotics');
    const gcEl = getSmallStatEl('GC Content');
    const lenNormEl = getSmallStatEl('Lenght Norm'); // note: "Lenght" typo in HTML

    // invisible file input (created by JS so HTML stays clean)
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = ".fasta,.fna,.fa";
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    // ---------- state ----------
    let antibiotics = [];
    let genes = [];
    let net = null;

    // ---------- UI helpers ----------
    function setUploading(is, label = null) {
        if (is) {
            uploadBtn.classList.add('loading');
            uploadBtn.querySelector('span')?.remove?.(); // optional
            uploadBtn.innerHTML = '<i class="bi bi-upload"></i> Processing...';
            uploadBtn.classList.remove('completed');
            uploadBtn.classList.add('uploading');
        } else {
            uploadBtn.classList.remove('loading');
            uploadBtn.innerHTML = '<i class="bi bi-upload"></i><span>Upload</span><div class="shortcut"><i class="bi bi-command"></i> U</div>';
            uploadBtn.classList.remove('uploading');
            uploadBtn.classList.add('completed');
            setTimeout(() => {
                uploadBtn.classList.remove('completed');
            }, 500);
            if (label) uploadBtn.title = label;
        }
        uploadBtn.disabled = is;
    }

    function safeNum(v, fallback = '—') {
        if (v === null || typeof v === 'undefined' || Number.isNaN(Number(v))) return fallback;
        return v;
    }

    // ---------- file upload flow ----------
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
        if (!fileInput.files.length) return;
        uploadAndPredict(fileInput.files[0]);
    });

    // keyboard shortcut: Cmd/Ctrl + U
    document.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'u') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Change the function signature to accept 'shouldSave' (default is true)
    async function uploadAndPredict(file, shouldSave = true) {
        const form = new FormData();
        form.append('file', file);
        const BACKEND_URL = "https://icybear02-gen-resist.hf.space/predict";

        try {
            setUploading(true, 'Analyzing genome...');
            const resp = await fetch(BACKEND_URL, { method: 'POST', body: form });
            if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);

            const data = await resp.json();

            // ONLY save if it's a new upload (shouldSave is true)
            if (shouldSave) {
                await sbSaveUpload(file, data).catch(err => console.warn('save upload failed', err));
            }

            applyFetchedData(data);
            setUploading(false, 'Analysis complete');
        } catch (err) {
            console.error(err);
            setUploading(false, 'Error');
            alert("Failed: " + err.message);
        }
    }

    // ---------- data parsing ----------
    function parseAntibioticsFromData(data) {
        // returns [{name, value (0-100), label?}, ...]
        let out = [];
        try {
            if (data.predictions && data.probabilities) {
                out = Object.keys(data.predictions).map(k => ({
                    name: k,
                    value: Math.round((data.probabilities[k] ?? 0) * 100),
                    label: data.predictions[k]
                }));
            } else if (data.predictions) {
                out = Object.keys(data.predictions).map(k => ({
                    name: k,
                    value: data.predictions[k] && String(data.predictions[k]).toLowerCase().includes('resist') ? 100 : 0,
                    label: data.predictions[k]
                }));
            } else if (Array.isArray(data.antibiotics)) {
                out = data.antibiotics.map(a => ({ name: a.name || a.antibiotic || 'Unknown', value: Number(a.value ?? a.percent ?? 0) }));
            } else if (Array.isArray(data.results)) {
                out = data.results.map(r => ({ name: r.name || r.antibiotic, value: Number(r.value ?? r.percent ?? 0) }));
            } else {
                out = [];
            }
        } catch (e) { out = []; }
        return out;
    }

    function parseGenesFromData(data) {
        let out = [];
        if (Array.isArray(data.genes) && data.genes.length) {
            out = data.genes.map(g => ({ name: g.name || g.gene || String(g.id || ''), mech: g.mech || g.mechanism || g.description || '—', impact: Number(g.impact ?? g.score ?? 50) }));
        } else if (Array.isArray(data.detected_genes)) {
            out = data.detected_genes.map(gname => ({ name: String(gname), mech: '—', impact: 50 }));
        } else {
            out = [];
        }
        return out;
    }

    // ---------- apply fetched data to UI ----------
    function applyFetchedData(data) {
        antibiotics = parseAntibioticsFromData(data);
        genes = parseGenesFromData(data);

        // update small stats
        if (totalAntEl) {
            const count = antibiotics.length;
            totalAntEl.querySelector('.statValue').textContent = String(count);
        }
        if (resistantAntEl) {
            let resCount = 0;
            if (data.predictions) {
                resCount = Object.values(data.predictions).filter(lbl => String(lbl).toLowerCase().includes('resist')).length;
            } else {
                resCount = antibiotics.filter(a => Number(a.value) > 50).length;
            }
            resistantAntEl.querySelector('.statValue').textContent = String(resCount);
        }
        if (gcEl) {
            const v = data.genome_stats && typeof data.genome_stats.gc_content === 'number' ? (data.genome_stats.gc_content * 100).toFixed(2) + '%' : '—';
            gcEl.querySelector('.statValue').textContent = v;
        }
        if (lenNormEl) {
            let rawVal = data.genome_stats && typeof data.genome_stats.length_normalized !== 'undefined'
                ? Number(data.genome_stats.length_normalized)
                : null;

            if (rawVal === null) {
                lenNormEl.querySelector('.statValue').textContent = '—';
            } else {
                lenNormEl.querySelector('.statValue').textContent = rawVal.toFixed(3);

                // 🚨 VALIDATION: too short genome / incomplete file
                if (rawVal < 0.5) {
                    alert("The uploaded genome file appears incomplete or too small.\nPlease upload a valid FASTA file.");
                    location.reload(); // reload the page
                    return; // stop further updates
                }
            }
        }

        // render components
        renderBars(antibiotics);
        renderGenes(genes);
        renderTreatments(antibiotics);
        renderNetwork(genes, antibiotics);
    }

    // ---------- render bars (pill-like) ----------
    function renderBars(list) {
        // each bar: .bar (container) with .barInner (height percentage) + .barValue (label) + .barName
        barsCont.innerHTML = '';
        if (!list || !list.length) {
            barsCont.innerHTML = '<div style="color:var(--muted,#999);padding:12px">No data</div>';
            return;
        }

        // sort by descending value to show highest first (optional)
        const display = [...list]; // keep original order; change if needed

        display.forEach((a, idx) => {
            const barWrap = document.createElement('div');
            barWrap.className = 'bar';
            // create visual pill element
            const barInner = barWrap
            const val = Number(a.value ?? 0);
            barInner.style.height = `${Math.max(6, val)}%`; // min height for visibility
            barInner.style.setProperty('--val', `${val}%`);
            if (val > 50) barInner.classList.add('high');
            else if (val > 10) barInner.classList.add('mid');
            else barInner.classList.add('low');

            // percentage label
            const barValue = document.createElement('div');
            barValue.className = 'barValue';
            barValue.textContent = `${val}%`;

            // name
            const barName = document.createElement('div');
            barName.className = 'barName';
            barName.textContent = a.name;

            // tooltip on hover (native small popup)
            barInner.addEventListener('mouseenter', (e) => {
                barInner.classList.add('hover');
            });
            barInner.addEventListener('mouseleave', (e) => {
                barInner.classList.remove('hover');
            });

            barWrap.appendChild(barValue);
            barWrap.appendChild(barName);
            barsCont.appendChild(barWrap);
        });
    }

    // ---------- render genes ----------
    function renderGenes(list) {
        genesCont.innerHTML = '';
        if (!list || !list.length) {
            genesCont.innerHTML = '<div style="color:var(--muted,#999)">No genes detected</div>';
            return;
        }
        list.forEach(g => {
            const el = document.createElement('div');
            el.className = 'gene';
            el.textContent = g.name;
            genesCont.appendChild(el);
        });
    }

    // ---------- render treatment suggestions ----------
    function renderTreatments(antibioticsList) {
        antibioticCont.innerHTML = '';
        if (!antibioticsList || !antibioticsList.length) {
            antibioticCont.innerHTML = '<div style="color:var(--muted,#999)">No suggestions</div>';
            return;
        }
        // choose lowest resistance as recommended (4)
        const sorted = [...antibioticsList].sort((a, b) => Number(a.value) - Number(b.value));
        const top = sorted.slice(0, 6);
        top.forEach(a => {
            const el = document.createElement('div');
            el.className = 'antibiotic';
            let className;
            if (Number(a.value) > 50) className = 'high';
            else if (Number(a.value) > 10) className = 'mid';
            else className = 'low';
            el.innerHTML = `
                        <div class="antibioticIcon ${className}"></div>
                        <div class="antibioticInfo">
                            <div class="antibioticName">${a.name}</div>
                            <div class="antibioticResistance">${Number(a.value)}% resistance</div>
                        </div>
            `;
            antibioticCont.appendChild(el);
        });
    }

    // ---------- vis-network ----------
    function ensureNetworkContainer() {
        const node = document.getElementById('genresist-network');

        if (!node) {
            console.error("Missing #genresist-network in HTML.");
            return null;
        }

        // Attach event listeners to buttons only once
        const fitBtn = document.getElementById('networkFitBtn');
        const saveBtn = document.getElementById('networkSaveBtn');

        if (fitBtn && !fitBtn._bound) {
            fitBtn.addEventListener('click', () => net && net.fit({ animation: { duration: 400 } }));
            fitBtn._bound = true;
        }

        if (saveBtn && !saveBtn._bound) {
            saveBtn.addEventListener('click', async () => {
                if (!node) return;
                const canvas = await html2canvas(node, { backgroundColor: '#0b0c10', scale: 2 });
                const a = document.createElement('a');
                a.href = canvas.toDataURL('image/png');
                a.download = 'network.png';
                a.click();
            });
            saveBtn._bound = true;
        }

        return node;
    }

    function renderNetwork(genesList, antibioticsList) {
        const container = ensureNetworkContainer();
        // initialize datastructs
        nodes.clear();
        edges.clear();

        // plasmid center node if genes exist
        let plasmidId = null;
        if (genesList && genesList.length) {
            plasmidId = 9999;
            nodes.add({ id: plasmidId, label: 'Plasmid', shape: 'diamond', size: 28, color: { background: '#ffd54f', border: '#ffb300' } });
        }

        // gene nodes
        genesList.forEach((g, i) => {
            const id = i + 1;
            const impact = Number(g.impact ?? 50);
            const size = Math.max(8, Math.min(26, 8 + impact / 3));
            const color = impact > 70 ? '#ff6a6a' : impact > 40 ? '#ffa84c' : '#4dd0e1';
            nodes.add({ id, label: g.name, shape: 'dot', size: size, color: { background: color, border: '#222' } });
            if (plasmidId) {
                edges.add({ 
                    from: plasmidId, 
                    to: id, 
                    arrows: 'to', 
                    label: 'carries', 
                    color: { color: '#21422E' },
                    length: 300 // <--- Add this: Force plasmid connections to be extra long
                });
            }
        });

        // antibiotic nodes (limit for clarity)
        (antibioticsList || []).slice(0, 12).forEach((a, idx) => {
            const id = 2000 + idx;
            nodes.add({ id, label: a.name, shape: 'box', size: 9, color: { background: '#81c784', border: '#4caf50' } });
            // connect to genes heuristically
            genesList.forEach((g, gi) => {
                if ((g.impact ?? 50) > 40) {
                    edges.add({ from: id, to: gi + 1, arrows: 'to', color: { color: '#407E56' } });
                }
            });
        });

        // create network
        if (net) net.destroy(); // cleanup
        net = new vis.Network(container, { nodes, edges }, {
            interaction: { hover: true, dragNodes: true, zoomView: true },
            // ... inside new vis.Network( ... )

            physics: {
                stabilization: true,
                barnesHut: {
                    // 1. INCREASE REPULSION (Push nodes apart)
                    // Was -1600. Changing to -12000 makes them push away much harder.
                    gravitationalConstant: -12000, 

                    // 2. REDUCE CENTRAL PULL (Don't let them clump in the middle)
                    centralGravity: 0.3, 

                    // 3. INCREASE EDGE LENGTH (Make lines longer)
                    springLength: 200, 

                    // 4. LOOSEN SPRINGS (Make lines more flexible)
                    springConstant: 0.015,

                    // 5. PREVENT OVERLAP (Force nodes to not sit on top of each other)
                    avoidOverlap: 1 
                },
                minVelocity: 0.75 // Stops the animation once it's "good enough" so it doesn't wiggle forever
            },

            // ... rest of your config

            // Node styling (black text)
            nodes: {
                font: { color: '#1a1a1a', size: 9, face: 'Inter' },
                borderWidth: 1
            },

            // Edge styling (dark gray lines + dark arrows)
            edges: {
                color: {
                    color: '#555',          // normal edge
                    highlight: '#222',      // when hovered
                    hover: '#000'           // darker on hover
                },
                arrows: {
                    to: {
                        enabled: true,
                        type: 'arrow',
                        scaleFactor: 0.6,
                        color: '#333'        // arrowheads become dark
                    }
                },
                width: 1.4,
                smooth: { type: 'dynamic' },
                font: {
                    color: '#333',
                    size: 11,
                    face: 'Inter',
                    strokeWidth: 0
                }
            }
        });

        // try fit
        try { net.fit({ animation: { duration: 350 } }); } catch (e) { }
    }

    // ---------- sample loader for testing ----------
    window.__loadSample = function () {
        const sample = {
            detected_genes: ["aac(6')-Ib", "mcr-1", "qnrB", "tetB"],
            genome_stats: { gc_content: 0.5058407187, length_normalized: 1 },
            predictions: {
                "Ampicillin": "Susceptible",
                "Nalidixic_acid": "Resistant",
                "Colistin": "Resistant",
                "Ciprofloxacin": "Susceptible"
            },
            probabilities: {
                "Ampicillin": 0.05,
                "Nalidixic_acid": 0.506,
                "Colistin": 0.85,
                "Ciprofloxacin": 0.24
            }
        };
        applyFetchedData(sample);
    };

    // ---------- initial state ----------
    // You may call __loadSample() from console to see UI fill
    renderBars([]);
    renderGenes([]);
    renderTreatments([]);

    // ---------- HISTORY / COMMUNITY LOADER ----------
    const urlParams = new URLSearchParams(window.location.search);
    const historyUrl = urlParams.get('url');
    const historyName = urlParams.get('name');

    if (historyUrl && historyName) {
        console.log("Loading:", historyName);

        sbFetchFileText(historyUrl).then(text => {
            // 1. Safety: Remove empty lines/spaces from start of file
            // FASTA files MUST start with ">"
            const cleanText = text.trim();

            if (!cleanText.startsWith(">")) {
                alert("Error: File content does not look like FASTA (must start with '>').");
                console.error("Bad Content Header:", cleanText.substring(0, 50));
                return;
            }

            // 2. Fix Name: Force extension to .fasta
            // We strip any existing extension and add .fasta to be 100% sure
            let safeName = historyName;
            // Remove .fna, .fa, .fasta from the end (case insensitive)
            safeName = safeName.replace(/\.(fna|fa|fasta)$/i, '');
            // Append clean .fasta
            safeName = safeName + ".fasta";

            console.log("Sending as:", safeName); // Check console to see the fixed name

            // 3. Create File with "application/octet-stream" to mimic real upload
            const f = new File([cleanText], safeName, { type: "application/octet-stream" });

            // 4. Send to backend
            uploadAndPredict(f, false);

        }).catch(err => {
            console.error("Failed to load file:", err);
            alert("Could not load file. Link might be broken.");
        });

        // Clean URL
        window.history.replaceState({}, document.title, "dashboard.html");
    }
})();

// ---------- OPEN BIG NETWORK VIEW ----------
const openBigBtn = document.getElementById("openBig");
const bigModal = document.getElementById("bigNetworkModal");
const closeBigBtn = document.getElementById("closeBig");
const bigContainer = document.getElementById("bigNetworkContainer");

let bigNet = null;

openBigBtn.addEventListener("click", () => {
    bigModal.classList.add("active");

    // destroy previous fullscreen network if any
    if (bigNet) {
        bigNet.destroy();
        bigNet = null;
    }

    // recreate network in fullscreen container
    bigNet = new vis.Network(bigContainer, { nodes, edges }, {
        interaction: { hover: true, dragNodes: true, zoomView: true },
        physics: { stabilization: true },

        nodes: {
            font: { color: '#1a1a1a', size: 16 },
            borderWidth: 1
        },

        edges: {
            color: {
                color: '#333333',     // normal edge + arrow color
                highlight: '#111111', // on hover 
                hover: '#000000'
            },
            arrows: {
                to: {
                    enabled: true,
                    type: "arrow",
                    scaleFactor: 0.7
                }
            },
            width: 1.5
        }
    });

    // fit after a tiny delay (ensures container size is computed)
    setTimeout(() => {
        bigNet.fit({ animation: { duration: 400 } });
    }, 120);
});

closeBigBtn.addEventListener("click", () => {
    bigModal.classList.remove("active");
});

// click outside content to close
bigModal.addEventListener("click", (e) => {
    if (e.target === bigModal) {
        bigModal.classList.remove("active");
    }
});

// ESC to close
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        bigModal.classList.remove("active");
    }
});

const darkModeBtn = document.getElementById("darkMode");

if (darkModeBtn) {
    darkModeBtn.addEventListener("click", async () => {
        // 1. Toggle State
        const newTheme = isDark ? "light" : "dark";

        // 2. Apply Visually (Instant)
        applyTheme(newTheme);
        localStorage.setItem('theme', newTheme);

        // 3. Save to Supabase (Async)
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
            sbSaveTheme(user.id, newTheme);
        }
    });
}

const logout = document.getElementById("logout")
logout.addEventListener("click", async () => {
    sbLogout()
});