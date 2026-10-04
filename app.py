import json, joblib, pandas as pd, streamlit as st
from kb import answer

st.set_page_config(page_title="Heart Risk Assistant", page_icon="🫀", layout="wide")
B = joblib.load("model_bundle.joblib"); MODEL, EXPL = B["model"], B["explainer"]; M = json.load(open("metrics.json"))
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1150px}
.hero{background:linear-gradient(135deg,#7f1d3a,#d6455a);color:#fff;border-radius:18px;padding:22px 28px;margin-bottom:14px}
.hero h1{margin:0;font-size:30px}.hero p{margin:4px 0 0;opacity:.9}
.card{border-radius:16px;padding:24px;text-align:center;color:#fff}
.big{font-size:60px;font-weight:700;line-height:1}
.pill{display:inline-block;padding:3px 11px;border-radius:99px;background:rgba(127,127,127,.18);margin:2px;font-size:13px}
</style><div class="hero"><h1>🫀 Heart Risk Assistant</h1><p>Estimate your risk, understand why, and ask questions — Cleveland dataset · educational prototype</p></div>""",
            unsafe_allow_html=True)

CP = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
ECG = ["Normal", "ST-T abnormality", "LV hypertrophy"]; SL = ["Upsloping", "Flat", "Downsloping"]; TH = ["Normal", "Fixed defect", "Reversible defect"]
LABEL = dict(age="Age", sex="Sex", cp="Chest pain type", trestbps="Resting BP", chol="Cholesterol", fbs="Fasting sugar", restecg="Resting ECG",
             thalach="Max heart rate", exang="Exercise angina", oldpeak="ST depression", slope="ST slope", ca="Vessels coloured", thal="Thalassemia")

with st.sidebar:
    st.header("Patient details")
    age = st.slider("Age", 20, 90, 54); sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    cp = st.selectbox("Chest pain type", CP); trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 120, 570, 245); fbs = st.checkbox("Fasting blood sugar > 120 mg/dl")
    restecg = st.selectbox("Resting ECG", ECG); thalach = st.slider("Max heart rate achieved", 70, 210, 150)
    exang = st.checkbox("Exercise-induced angina"); oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("ST slope", SL); ca = st.slider("Major vessels coloured (0-4)", 0, 4, 0); thal = st.selectbox("Thalassemia", TH)
    go = st.button("Assess risk", type="primary", use_container_width=True)

def make_row(**o):
    r = dict(age=age, sex=int(sex == "Male"), cp=CP.index(cp), trestbps=trestbps, chol=chol, fbs=int(fbs), restecg=ECG.index(restecg),
             thalach=thalach, exang=int(exang), oldpeak=oldpeak, slope=SL.index(slope), ca=ca, thal=TH.index(thal) + 1); r.update(o)
    return pd.DataFrame([r])

def assess(row):
    p = float(MODEL.predict_proba(row)[0, 1]); pre, lr = EXPL.named_steps["p"], EXPL.named_steps["m"]
    Z = pre.transform(row); Z = Z.toarray() if hasattr(Z, "toarray") else Z; agg = {}
    for n, c in zip(pre.get_feature_names_out(), Z[0] * lr.coef_[0]):
        f = n.split("__")[1]; f = f.split("_")[0] if n.startswith("c__") else f; agg[f] = agg.get(f, 0) + c
    s = pd.Series(agg).rename(LABEL).sort_values()
    level = "Low" if p < .35 else "Moderate" if p < .65 else "High"
    return dict(p=p, level=level, s=s, up=list(s[s > .15].sort_values(ascending=False).index[:3]), down=list(s[s < -.15].index[:3]),
                inputs=make_row().iloc[0].to_dict())

if go or "res" not in st.session_state: st.session_state.res = assess(make_row())
R = st.session_state.res; color = dict(Low="#2e9e6b", Moderate="#e0a21b", High="#d6455a")[R["level"]]
t1, t2, t3 = st.tabs(["📊 Assessment", "💬 Ask the assistant", "🔬 About the model"])

with t1:
    a, b = st.columns([1, 1.6])
    with a:
        st.markdown(f'<div class="card" style="background:{color}">Estimated risk<div class="big">{R["p"]*100:.0f}%</div>'
                    f'<div style="font-size:22px;margin-top:6px">{R["level"]} risk</div></div>', unsafe_allow_html=True)
        st.progress(R["p"])
        pill = lambda L: " ".join(f'<span class="pill">{x}</span>' for x in L) or "nothing major"
        st.markdown("**Pushing risk up:** " + pill(R["up"]), unsafe_allow_html=True)
        st.markdown("**Pulling risk down:** " + pill(R["down"]), unsafe_allow_html=True)
    with b:
        st.subheader("What drives this result"); st.bar_chart(R["s"].rename("Effect on risk"), horizontal=True, color="#d6455a")
    st.subheader("What-if simulator")
    w1, w2, w3 = st.columns(3)
    nbp = w1.slider("Resting BP", 90, 200, trestbps, key="wbp"); nch = w2.slider("Cholesterol", 120, 570, chol, key="wch")
    nhr = w3.slider("Max heart rate", 70, 210, thalach, key="whr")
    np_ = float(MODEL.predict_proba(make_row(trestbps=nbp, chol=nch, thalach=nhr))[0, 1])
    st.metric("Risk with these changes", f"{np_*100:.0f}%", f"{(np_-R['p'])*100:+.0f} pts", delta_color="inverse")
    st.caption("Shows how the model responds to the change. It is a statistical association, not proof that changing a number prevents disease.")

with t2:
    st.caption("Ask about your result or heart health. Set ANTHROPIC_API_KEY to use Claude; otherwise a built-in guideline assistant answers.")
    if "chat" not in st.session_state:
        st.session_state.chat = [dict(role="assistant", content=f"Hi! Your estimated risk is **{R['p']*100:.0f}% ({R['level']})**. Ask me why, or about BP, cholesterol, exercise, diet or when to see a doctor.")]
    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    chips = st.columns(4); picked = None
    for col, q in zip(chips, ["Why is my risk this level?", "How can I lower my BP?", "What exercise is safe?", "When should I see a doctor?"]):
        if col.button(q, use_container_width=True): picked = q
    q = st.chat_input("Type your question…") or picked
    if q:
        hist = list(st.session_state.chat); st.session_state.chat.append(dict(role="user", content=q))
        with st.chat_message("user"): st.markdown(q)
        txt, src = answer(q, hist, R); st.session_state.chat.append(dict(role="assistant", content=txt))
        with st.chat_message("assistant"): st.markdown(txt); st.caption(f"answered by {src} assistant")

with t3:
    h, c = M["holdout"], M["cv"]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("CV accuracy", f"{c['accuracy']*100:.1f}%", f"±{c['accuracy_std']*100:.1f}", delta_color="off")
    x2.metric("ROC-AUC", f"{c['roc_auc']:.3f}"); x3.metric("Recall", f"{c['recall']*100:.0f}%"); x4.metric("Precision", f"{c['precision']*100:.0f}%")
    st.markdown(f"**Model:** regularized logistic regression with one-hot encoded categorical features, trained on {M['n']} Cleveland patients. "
                f"Fixed 80/20 holdout (seed {M['seed']}, n={h['n_test']}): accuracy **{h['accuracy']*100:.1f}%**, AUC {h['roc_auc']:.2f}. "
                "With only 302 patients, the repeated cross-validation estimate is more informative than a single holdout split. "
                "The 'what drives this' chart uses the fitted logistic-regression coefficients.")
    st.warning("Educational prototype trained on one 1980s hospital dataset. Not a medical device. In an emergency call 112 (India).")
