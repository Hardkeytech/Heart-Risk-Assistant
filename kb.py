"""Chatbot brain: tiny retrieval over general heart-health guidance + optional Claude (set ANTHROPIC_API_KEY)."""
import os, re

KB = [
 ("blood pressure hypertension bp mmhg", "Blood pressure", "Persistently high blood pressure strains the heart and arteries. Most guidelines aim for readings below about 130/80 mm Hg. Cutting salt, staying active, limiting alcohol and keeping a healthy weight all help; if readings stay high, a doctor can discuss medication."),
 ("cholesterol chol ldl lipid statin fat", "Cholesterol", "High LDL cholesterol builds up plaque in arteries. Fibre-rich foods, fewer saturated and trans fats, and regular exercise help. A lipid profile blood test shows your LDL, HDL and triglycerides, and your doctor decides whether medication is needed."),
 ("exercise activity walk workout fitness cardio", "Physical activity", "Adults are generally advised to get about 150 minutes of moderate activity a week, such as brisk walking, plus two strength sessions. Start gently if you have been inactive, and check with a doctor first if you have symptoms."),
 ("diet food eat nutrition salt sugar weight", "Heart-friendly eating", "Favour vegetables, fruit, pulses, whole grains, nuts and fish; cut back on salt, sugary drinks, deep-fried food and processed meat. Small, steady changes last longer than strict diets."),
 ("smoking smoke tobacco cigarette quit vape", "Smoking", "Smoking is one of the strongest changeable risk factors. Risk starts falling soon after quitting. Doctors, counselling and nicotine replacement can roughly double the chance of quitting successfully."),
 ("sugar diabetes glucose fbs fasting hba1c", "Blood sugar", "Fasting blood sugar above 120 mg/dl (the model's cut-off) can signal prediabetes or diabetes, which raises heart risk. Weight, diet and activity matter; ask a doctor about an HbA1c test."),
 ("chest pain angina cp symptoms emergency attack breathless", "Warning signs", "Chest pressure or pain, pain spreading to the arm, jaw or back, sudden breathlessness, cold sweat or fainting can signal a heart attack. Do not wait: call your local emergency number (112 in India) immediately."),
 ("ecg restecg electrocardiogram st oldpeak slope", "ECG and ST depression", "A resting ECG records the heart's electrical activity. ST depression ('oldpeak') during exercise, especially with a flat or downsloping ST slope, is a sign the heart muscle may not be getting enough blood. Only a doctor can interpret it."),
 ("thalassemia thal ca vessels fluoroscopy angiography", "Scan features", "'Major vessels coloured' counts narrowed arteries seen on fluoroscopy, and the thallium test shows blood flow to the heart. In this dataset more vessels coloured and a reversible defect are linked to higher risk."),
 ("stress sleep anxiety mental alcohol", "Stress, sleep and alcohol", "Poor sleep (under 6 hours), chronic stress and heavy drinking all push blood pressure up. Aim for 7-8 hours of sleep and build in daily relaxation or movement."),
 ("heart rate thalach pulse maximum", "Heart rate", "Maximum heart rate during exercise falls with age and with poorer fitness. In this dataset a lower peak heart rate is associated with higher risk. Fitness training can improve it gradually."),
 ("checkup doctor test screening cardiologist when", "When to see a doctor", "See a doctor soon if your estimated risk is moderate or high, if you have symptoms, or if you have a family history of early heart disease. A routine check of BP, cholesterol and sugar every year or two is sensible for adults over 40."),
]
EMERGENCY = re.compile(r"chest (pain|pressure|tight)|can.?t breathe|fainted|passing out|heart attack|left arm", re.I)

def retrieve(q, k=2):
    w = set(re.findall(r"[a-z]+", q.lower()))
    sc = sorted(((len(w & set(t.split())), n, b) for t, n, b in KB), reverse=True)
    return [(n, b) for s, n, b in sc[:k] if s > 0]

def ctx_text(c):
    if not c: return "No assessment has been run yet."
    return (f"Estimated risk {c['p']*100:.0f}% ({c['level']}). Raising risk: {', '.join(c['up']) or 'none'}. "
            f"Lowering risk: {', '.join(c['down']) or 'none'}. Inputs: {c['inputs']}")

def local_answer(q, c):
    ql = q.lower(); out = []
    if EMERGENCY.search(q):
        out.append("⚠️ **If you have these symptoms right now, call your local emergency number (112 in India) or get to a hospital. Please don't wait for an app.**")
    if re.search(r"why|explain|driver|reason|factor|result|score|risk", ql) and c:
        out.append(f"Your estimated risk is **{c['p']*100:.0f}% ({c['level']})**. "
                   f"What pushes it up: {', '.join(c['up']) or 'nothing major'}. What helps: {', '.join(c['down']) or 'nothing major'}. "
                   "These come from the model's coefficients for your inputs, so they show what the model weighs, not a diagnosis.")
    elif re.search(r"why|explain|driver|result|score", ql) and not c:
        out.append("Fill in the patient details and press **Assess risk** first, then I can explain the result.")
    for n, b in retrieve(q): out.append(f"**{n}.** {b}")
    if not out:
        out.append("I can explain your risk result and share general guidance on blood pressure, cholesterol, exercise, diet, smoking, blood sugar, warning signs and when to see a doctor. Try asking one of those.")
    out.append("_General education only, not a diagnosis. Please talk to a doctor about your results._")
    return "\n\n".join(out)

def answer(q, history, c):
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            ref = "\n".join(f"- {n}: {b}" for n, b in retrieve(q, 3))
            sys = ("You are a friendly heart-health assistant inside a risk-estimation app built on the Cleveland dataset. "
                   "Explain the user's result and give general lifestyle education only. Never diagnose, never recommend medication or doses, "
                   "and tell users with acute symptoms to call emergency services (112 in India). Be concise (under 150 words).\n"
                   f"Current assessment: {ctx_text(c)}\nReference notes:\n{ref}")
            msgs = [{"role": m["role"], "content": m["content"]} for m in history[-8:]] + [{"role": "user", "content": q}]
            r = anthropic.Anthropic(api_key=key).messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=500, system=sys, messages=msgs)
            return r.content[0].text, "Claude"
        except Exception:
            pass
    return local_answer(q, c), "built-in"
