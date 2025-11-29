// supabase.js
// Import Supabase from CDN
import { createClient } from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm";

// REPLACE THESE WITH YOUR ACTUAL KEYS FROM SUPABASE DASHBOARD
const SUPABASE_URL = "https://qatkzusxuggkvnbuvpjd.supabase.co"; 
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFhdGt6dXN4dWdna3ZuYnV2cGpkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMjkyODYsImV4cCI6MjA3OTkwNTI4Nn0.x_c0SQihruoogxpg2TqKtvMMOu0ca-jPuVszqBBYaFc";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ----- AUTH HELPER -----
export async function sbRegister(name, email, password) {
    // We only need to Sign Up. The Database Trigger handles the Profile creation!
    const { data, error } = await supabase.auth.signUp({
        email: email,
        password: password,
        options: {
            data: {
                name: name, // <--- We pass name here so the Trigger sees it
            }
        }
    });

    if (error) throw error;
    
    // No manual insert needed anymore!
    return data.user;
}

export async function sbLogin(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
    });
    if (error) throw error;
    return data.user;
}

export async function sbLogout() {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
}

// ----- THEME -----
export async function sbLoadTheme(uid) {
    const { data } = await supabase
        .from('profiles')
        .select('theme')
        .eq('id', uid)
        .maybeSingle();
    return data?.theme || 'light';
}

// ----- UPLOAD -----
export async function sbSaveUpload(file, resultData) {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) throw new Error("Not authenticated");

    // 1. Upload File to Storage Bucket "genomes"
    const filePath = `user_uploads/${user.id}/${Date.now()}_${file.name}`;
    const { data: uploadData, error: uploadError } = await supabase
        .storage
        .from('genomes')
        .upload(filePath, file);

    if (uploadError) throw uploadError;

    // 2. Get Public URL
    const { data: urlData } = supabase.storage.from('genomes').getPublicUrl(filePath);
    const publicUrl = urlData.publicUrl;

    // 3. Save Metadata to Database Table "user_uploads"
    const metadata = {
        gc: resultData?.genome_stats?.gc_content ?? null,
        lenNorm: resultData?.genome_stats?.length_normalized ?? null,
        resistant: resultData?.predictions ? Object.values(resultData.predictions).filter(v => String(v).toLowerCase().includes("resist")).length : 0,
        antibioticsList: Object.keys(resultData.predictions || {})
    };

    const { data: insertData, error: dbError } = await supabase
        .from('user_uploads')
        .insert([{
            user_id: user.id,
            filename: file.name,
            file_url: publicUrl,
            metadata: metadata
        }])
        .select()
        .single();

    if (dbError) throw dbError;
    return insertData;
}

// ----- HISTORY -----
export async function sbLoadHistory() {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return [];

    const { data, error } = await supabase
        .from('user_uploads')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

    if (error) {
        console.error(error);
        return [];
    }
    return data;
}

// ----- COMMUNITY -----
export async function sbLoadCommunity() {
    const { data, error } = await supabase
        .from('community_uploads')
        .select('*')
        .order('created_at', { ascending: false });
    
    if (error) return [];
    return data;
}

// ----- UTILS -----
export async function sbFetchFileText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("Failed to fetch file");
    return await r.text();
}

// Auth State Observer Wrapper
export function onAuthStateChange(callback) {
    return supabase.auth.onAuthStateChange((event, session) => {
        callback(session?.user || null);
    });
}

// Add this to supabase.js
// Add to supabase.js
export async function sbSaveTheme(uid, theme) {
    const { error } = await supabase
        .from('profiles')
        .update({ theme: theme })
        .eq('id', uid);

    if (error) console.error("Error saving theme:", error);
}