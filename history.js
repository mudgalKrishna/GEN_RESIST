import { sbSaveUpload, sbLoadTheme, sbSaveTheme, onAuthStateChange, sbFetchFileText, supabase, sbLogout } from './supabase.js';

let isDark = false;

function applyTheme(themeName) {
    const root = document.documentElement;
    const themeObj = themeName === 'dark' ? darkTheme : lightTheme;
    Object.keys(themeObj).forEach(key => root.style.setProperty(key, themeObj[key]));
    isDark = (themeName === 'dark');
}
// In dashboard.js

onAuthStateChange(async (user) => {
    if (!user) {
        window.location.href = 'login.html';
        return;
    }

    // 1. Load Theme
    const theme = await sbLoadTheme(user.id);
    applyTheme(theme);

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


const darkModeBtn = document.getElementById("darkMode");

if (darkModeBtn) {
    darkModeBtn.addEventListener("click", async () => {
        // 1. Toggle State
        const newTheme = isDark ? "light" : "dark";
        
        // 2. Apply Visually (Instant)
        applyTheme(newTheme);

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