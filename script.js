// ═══════════════════════════════════════════
// Arabic ABSA UI — script.js
// Bilingual: English / Arabic toggle
// ═══════════════════════════════════════════
 
// ── TRANSLATIONS ──────────────────────────
const TRANSLATIONS = {
  en: {
    logoSub:          "Arabic Sentiment Analysis",
    heroTag:          "Advanced AI Model",
    heroTitle:        "Analyze Sentiment in",
    heroTitleAccent:  "Arabic Reviews",
    heroDesc:         "Enter any Arabic review and the model will identify all mentioned aspects and classify the sentiment for each one independently",
    inputLabel:       "Enter your review here (Arabic text)",
    placeholder:      "Example: الأكل كان ممتاز بس الخدمة كانت بطيئة جداً والسعر مناسب...",
    chars:            "chars",
    trySample:        "Try a sample:",
    sampleRestaurant: "Restaurant",
    sampleApp:        "App",
    sampleCafe:       "Cafe",
    sampleDelivery:   "Delivery",
    analyzeBtn:       "Analyze Review",
    results:          "Results",
    emptyHint:        "Enter a review and click Analyze to see results",
    stat1:            "Training Reviews",
    stat2:            "F1 Accuracy",
    stat3:            "Supported Aspects",
    stat4:            "Sentiment Classes",
    legendTitle:      "Supported Aspects",
    detectedAspects:  "Detected Aspects",
    aspect:           "aspect",
    aspects:          "aspects",
    confidence:       "Confidence",
    summary:          "Summary:",
    positive:         "Positive",
    negative:         "Negative",
    neutral:          "Neutral",
    errEmpty:         "Please enter a review text to analyze",
    errServer:        "Could not connect to server. Make sure python app.py is running.",
    langLabel:        "العربية",
    btnArrow:         "→",
  },
  ar: {
    logoSub:          "تحليل المشاعر العربي",
    heroTag:          "نموذج ذكاء اصطناعي متقدم",
    heroTitle:        "حلل مشاعر تقييماتك",
    heroTitleAccent:  "بدقة احترافية",
    heroDesc:         "أدخل أي تقييم عربي وسيحدد النموذج الجوانب المذكورة ومشاعر كل جانب على حدة",
    inputLabel:       "أدخل التقييم هنا",
    placeholder:      "مثال: الأكل كان ممتاز بس الخدمة كانت بطيئة جداً والسعر مناسب...",
    chars:            "حرف",
    trySample:        "جرب مثالاً:",
    sampleRestaurant: "مطعم",
    sampleApp:        "تطبيق",
    sampleCafe:       "كافيه",
    sampleDelivery:   "توصيل",
    analyzeBtn:       "تحليل التقييم",
    results:          "النتائج",
    emptyHint:        "أدخل تقييماً واضغط تحليل لرؤية النتائج",
    stat1:            "تقييم للتدريب",
    stat2:            "دقة النموذج F1",
    stat3:            "جانب مدعوم",
    stat4:            "مشاعر محللة",
    legendTitle:      "الجوانب المدعومة",
    detectedAspects:  "الجوانب المكتشفة",
    aspect:           "جانب",
    aspects:          "جوانب",
    confidence:       "الثقة",
    summary:          "الملخص:",
    positive:         "إيجابي",
    negative:         "سلبي",
    neutral:          "محايد",
    errEmpty:         "الرجاء إدخال نص للتحليل",
    errServer:        "تعذر الاتصال بالخادم. تأكد من تشغيل python app.py",
    langLabel:        "English",
    btnArrow:         "←",
  }
};
 
const ASPECT_INFO = {
  food:           { en: "Food",         ar: "الطعام",    emoji: "🍕" },
  service:        { en: "Service",      ar: "الخدمة",    emoji: "👨‍💼" },
  price:          { en: "Price",        ar: "السعر",     emoji: "💰" },
  cleanliness:    { en: "Cleanliness",  ar: "النظافة",   emoji: "✨" },
  delivery:       { en: "Delivery",     ar: "التوصيل",  emoji: "🚚" },
  ambiance:       { en: "Ambiance",     ar: "الأجواء",   emoji: "🏮" },
  app_experience: { en: "App",          ar: "التطبيق",  emoji: "📱" },
  general:        { en: "General",      ar: "عام",       emoji: "⭐" },
  none:           { en: "None",         ar: "لا شيء",   emoji: "➖" },
};
 
// ── STATE ─────────────────────────────────
let currentLang = "en";
 
// ── LANGUAGE TOGGLE ───────────────────────
function toggleLang() {
  currentLang = currentLang === "en" ? "ar" : "en";
  applyLanguage();
}
 
function applyLanguage() {
  const t    = TRANSLATIONS[currentLang];
  const html = document.getElementById("htmlRoot");
  const isAr = currentLang === "ar";
 
  // Set direction on whole page
  html.setAttribute("lang", currentLang);
  html.setAttribute("dir",  isAr ? "rtl" : "ltr");
 
  // Update toggle button
  document.getElementById("langLabel").textContent = t.langLabel;
  document.getElementById("btnArrow").textContent  = t.btnArrow;
 
  // Update all data-i18n elements
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (t[key] !== undefined) el.textContent = t[key];
  });
 
  // Update placeholders
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (t[key] !== undefined) el.setAttribute("placeholder", t[key]);
  });
 
  // Re-render results if visible (so sentiment labels update too)
  const content = document.getElementById("resultsContent");
  if (content.style.display !== "none" && content._lastData) {
    renderResults(content._lastData);
  }
}
 
// ── CHAR COUNTER ──────────────────────────
const textarea  = document.getElementById("reviewInput");
const charCount = document.getElementById("charCount");
textarea.addEventListener("input", () => {
  charCount.textContent = textarea.value.length;
});
 
// ── SAMPLE REVIEWS ────────────────────────
function setSample(text) {
  textarea.value = text;
  charCount.textContent = text.length;
  textarea.focus();
}
 
// ── ANALYZE ───────────────────────────────
async function analyzeReview() {
  const text = textarea.value.trim();
  const t    = TRANSLATIONS[currentLang];
 
  if (!text) { showError(t.errEmpty); return; }
 
  const btn     = document.getElementById("analyzeBtn");
  const empty   = document.getElementById("resultsEmpty");
  const content = document.getElementById("resultsContent");
 
  btn.classList.add("loading");
  btn.disabled = true;
  empty.style.display   = "none";
  content.style.display = "none";
 
  try {
    const response = await fetch("/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error("Server error");
    const data = await response.json();
    content._lastData = data;
    renderResults(data);
  } catch (err) {
    showError(t.errServer);
    empty.style.display = "block";
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}
 
// ── RENDER RESULTS ────────────────────────
function renderResults(data) {
  const content = document.getElementById("resultsContent");
  const t       = TRANSLATIONS[currentLang];
  const { aspects, aspect_sentiments, confidence } = data;
 
  const counts = { positive: 0, negative: 0, neutral: 0 };
  aspects.forEach(a => { const s = aspect_sentiments[a]; if (s) counts[s]++; });
 
  const aspectCountLabel = aspects.length === 1
    ? `1 ${t.aspect}`
    : `${aspects.length} ${t.aspects}`;
 
  let html = `
    <div class="result-header">
      <div class="result-title">${t.detectedAspects}</div>
      <div class="result-count">${aspectCountLabel}</div>
    </div>
    <div class="aspects-grid">
      ${aspects.map((asp, i) => {
        const sent = aspect_sentiments[asp] || "neutral";
        const info = ASPECT_INFO[asp] || { en: asp, ar: asp, emoji: "❓" };
        const conf = confidence ? Math.round(confidence[asp] || 0) : 0;
        const name = currentLang === "ar" ? info.ar : info.en;
        return `
          <div class="aspect-card ${sent}" style="animation-delay:${i * 0.07}s">
            <div class="aspect-card-top">
              <span class="aspect-emoji">${info.emoji}</span>
              <span class="sentiment-badge badge-${sent}">
                ${sentimentIcon(sent)} ${t[sent]}
              </span>
            </div>
            <div class="aspect-name-ar">${name}</div>
            <div class="aspect-name-en">${asp}</div>
            <div class="conf-bar-wrap">
              <div class="conf-label">
                <span>${t.confidence}</span><span>${conf}%</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill fill-${sent}" style="width:${conf}%"></div>
              </div>
            </div>
          </div>`;
      }).join("")}
    </div>
    <div class="summary-bar">
      <div class="summary-label">${t.summary}</div>
      <div class="summary-counts">
        ${counts.positive > 0 ? `<div class="summary-item"><div class="summary-dot" style="background:#10B981"></div><span style="color:#10B981">${counts.positive} ${t.positive}</span></div>` : ""}
        ${counts.negative > 0 ? `<div class="summary-item"><div class="summary-dot" style="background:#EF4444"></div><span style="color:#EF4444">${counts.negative} ${t.negative}</span></div>` : ""}
        ${counts.neutral  > 0 ? `<div class="summary-item"><div class="summary-dot" style="background:#F59E0B"></div><span style="color:#F59E0B">${counts.neutral}  ${t.neutral}</span></div>`  : ""}
      </div>
    </div>`;
 
  content.innerHTML     = html;
  content.style.display = "block";
}
 
// ── HELPERS ───────────────────────────────
function sentimentIcon(sent) {
  return sent === "positive" ? "✅" : sent === "negative" ? "❌" : "➖";
}
 
function showError(msg) {
  const content = document.getElementById("resultsContent");
  const empty   = document.getElementById("resultsEmpty");
  empty.style.display   = "none";
  content.style.display = "block";
  content.innerHTML     = `<div class="error-box">⚠️ ${msg}</div>`;
}
 
// ── ENTER SHORTCUT ────────────────────────
textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && e.ctrlKey) analyzeReview();
});
 
// ── INIT ──────────────────────────────────
applyLanguage();