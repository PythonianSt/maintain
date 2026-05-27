# app.py
import streamlit as st
import pandas as pd
import requests
import base64
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI

st.set_page_config(page_title="Campus Psychosomatic Signal Intake", page_icon="🧠", layout="centered")

# =========================
# GitHub helpers
# =========================
GITHUB_TOKEN = st.secrets["github"]["token"]
REPO = st.secrets["github"]["repo"]          # e.g. "username/repo"
BRANCH = st.secrets["github"].get("branch", "main")
CSV_PATH = st.secrets["github"].get("csv_path", "psychosomatic_intake.csv")

API_URL = f"https://api.github.com/repos/{REPO}/contents/{CSV_PATH}"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
MODEL = st.secrets["openai"].get("model", "gpt-5.5")

def generate_gpt_action(student_summary):
    """
    Generate a Thai action plan. Uses the newer Responses API first, then falls back
    to Chat Completions. Returns a non-empty string or raises a visible error.
    """
    prompt = f"""
คุณคือผู้ช่วยแพทย์ในสถานพยาบาลมหาวิทยาลัย
ช่วยสร้าง Action Plan ภาษาไทยแบบสุภาพ อบอุ่น และไม่ตัดสิน
สำหรับนักศึกษาที่มีอาการกาย-ใจ/psychosomatic signals

ต้องตอบเป็นภาษาไทย และต้องมี 3 หัวข้อชัดเจน:
1) ข้อความ check-in สำหรับส่งให้นักศึกษา
2) ข้อเสนอการพูดคุยปรึกษา / counseling offer
3) ช่องทางติดต่อพยาบาล / สถานพยาบาล

หลักความปลอดภัย:
- ห้ามวินิจฉัยโรค
- ห้ามกล่าวว่านักศึกษาเป็นโรคซึมเศร้า วิตกกังวล หรือ psychosomatic disorder
- ใช้คำว่า “สัญญาณที่ควรดูแลต่อเนื่อง” แทนการวินิจฉัย
- หาก suicidal item positive = True ให้เน้นความปลอดภัย พบเจ้าหน้าที่ทันที และไม่ควรอยู่ลำพัง
- ทำให้ข้อความสั้นพอที่จะ copy ส่งทาง LINE/SMS ได้

ข้อมูลนักศึกษา:
{student_summary}
"""

    instructions = (
        "You write safe, warm, non-judgmental Thai health-support messages "
        "for university students. Do not diagnose. Always include check-in, "
        "counseling offer, and nurse/infirmary contact."
    )

    # Preferred current API
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=900,
        )
        text = getattr(response, "output_text", "") or ""
        if text.strip():
            return text.strip()
    except Exception as e:
        st.warning(f"Responses API ใช้ไม่ได้ จึงลองใช้ Chat Completions แทน: {e}")

    # Fallback API
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=900,
        )
        text = response.choices[0].message.content or ""
        if text.strip():
            return text.strip()
        raise ValueError("GPT returned empty content.")
    except Exception as e:
        st.error(f"ไม่สามารถสร้าง Action Plan ได้: {e}")
        return ""


def now_bkk():
    return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")

def load_csv_from_github():
    r = requests.get(API_URL, headers=HEADERS, params={"ref": BRANCH})
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        df = pd.read_csv(StringIO(content), index_col="student_id")
        return df, data["sha"]
    elif r.status_code == 404:
        return pd.DataFrame(), None
    else:
        st.error(f"GitHub read error: {r.status_code} {r.text}")
        return pd.DataFrame(), None

def save_csv_to_github(df, sha=None):
    csv_text = df.to_csv(index=True)
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"Update psychosomatic intake CSV {now_bkk()}",
        "content": encoded,
        "branch": BRANCH
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(API_URL, headers=HEADERS, json=payload)
    if r.status_code in [200, 201]:
        return True
    else:
        st.error(f"GitHub write error: {r.status_code} {r.text}")
        return False

# =========================
# Risk interpretation
# =========================
def phq_level(score):
    if score >= 20: return "รุนแรงมาก"
    if score >= 15: return "รุนแรงปานกลางถึงมาก"
    if score >= 10: return "ปานกลาง"
    if score >= 5: return "เล็กน้อย"
    return "ต่ำ"

def gad_level(score):
    if score >= 15: return "รุนแรง"
    if score >= 10: return "ปานกลาง"
    if score >= 5: return "เล็กน้อย"
    return "ต่ำ"

def signal_level(total_signal, phq9, gad7, suicidal):
    if suicidal or phq9 >= 20 or gad7 >= 15 or total_signal >= 16:
        return "แดง", "ควรพบแพทย์/พยาบาลวันนี้ หรือส่งต่อหากมีความเสี่ยงสูง"
    if phq9 >= 10 or gad7 >= 10 or total_signal >= 10:
        return "เหลือง", "ควรนัดติดตาม ประเมินซ้ำ และให้คำแนะนำเรื่องการนอน ความเครียด และการเรียน"
    return "เขียว", "ให้คำแนะนำ self-care และติดตามตามความเหมาะสม"

# =========================
# UI
# =========================
st.title("🧠 แบบคัดกรองสัญญาณกาย-ใจ สำหรับนักศึกษา")
st.caption("Psychosomatic Signal Intake | สำหรับสถานพยาบาลในมหาวิทยาลัย")

st.warning(
    "เครื่องมือนี้เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรค "
    "หากมีความคิดทำร้ายตนเอง ทำร้ายผู้อื่น หรือรู้สึกไม่ปลอดภัย ควรติดต่อเจ้าหน้าที่ทันที"
)

with st.expander("📌 คำชี้แจง PDPA และการคุ้มครองข้อมูลส่วนบุคคล", expanded=True):
    st.write("""
ข้อมูลที่กรอกจะใช้เพื่อการคัดกรองสุขภาพ การดูแลต่อเนื่อง การนัดหมาย และการพัฒนาระบบบริการสุขภาพของสถานพยาบาลเท่านั้น  
ข้อมูลจะถูกจัดเก็บอย่างจำกัดตามความจำเป็น โดยใช้รหัสนักศึกษาเป็นดัชนีในการติดตาม  
ผู้ดูแลระบบควรจำกัดสิทธิ์การเข้าถึงข้อมูล เก็บ GitHub repository เป็น private และไม่นำข้อมูลไปเผยแพร่เป็นรายบุคคล  
การกด “ยินยอม” หมายถึงผู้กรอกเข้าใจวัตถุประสงค์ของการเก็บข้อมูลและยินยอมให้ใช้ข้อมูลเพื่อการดูแลสุขภาพภายในหน่วยงาน
""")

consent = st.checkbox("ข้าพเจ้ายินยอมให้บันทึกและใช้ข้อมูลตามวัตถุประสงค์ด้านการดูแลสุขภาพ")

if not consent:
    st.stop()

st.subheader("1) ข้อมูลทั่วไป")
student_id = st.text_input("รหัสนักศึกษา *")
faculty = st.text_input("คณะ/สาขา")
year = st.selectbox("ชั้นปี", ["", "1", "2", "3", "4", "5", "6", "อื่น ๆ"])
visit_reason = st.text_area("เหตุผลที่มารับบริการวันนี้ / อาการสำคัญ")

st.subheader("2) สัญญาณจากแบบสอบถาม")
sleep = st.slider("คุณภาพการนอน 0 = แย่มาก, 10 = ดีมาก", 0, 10, 5)
fatigue = st.slider("ความเหนื่อยล้า 0 = ไม่มี, 10 = มากที่สุด", 0, 10, 5)
class_attendance = st.slider("การเข้าเรียนช่วง 2 สัปดาห์ที่ผ่านมา (%)", 0, 100, 80)
exercise_freq = st.slider("ออกกำลังกายกี่วันต่อสัปดาห์", 0, 7, 2)
loneliness = st.slider("ความรู้สึกโดดเดี่ยว 0 = ไม่มี, 10 = มากที่สุด", 0, 10, 3)
repeated_visits_self = st.number_input("จำนวนครั้งที่มาสถานพยาบาลด้วยอาการคล้ายเดิมใน 1 เดือน", 0, 30, 0)

st.subheader("3) PHQ-9")
phq_questions = [
    "เบื่อ ไม่สนใจทำสิ่งต่าง ๆ",
    "ไม่สบายใจ ซึมเศร้า หรือท้อแท้",
    "หลับยาก หลับ ๆ ตื่น ๆ หรือหลับมากไป",
    "เหนื่อยง่าย หรือไม่ค่อยมีแรง",
    "เบื่ออาหาร หรือกินมากเกินไป",
    "รู้สึกไม่ดีกับตัวเอง รู้สึกล้มเหลว",
    "สมาธิไม่ดี เช่น อ่านหนังสือหรือดูสื่อไม่รู้เรื่อง",
    "เคลื่อนไหวหรือพูดช้าลง หรือกระสับกระส่าย",
    "คิดว่าตายไปเสียจะดีกว่า หรือคิดทำร้ายตนเอง"
]
phq_scores = []
for q in phq_questions:
    phq_scores.append(st.selectbox(q, [0, 1, 2, 3], format_func=lambda x: ["ไม่มีเลย", "บางวัน", "มากกว่า 7 วัน", "เกือบทุกวัน"][x]))
phq9 = sum(phq_scores)
suicidal = phq_scores[8] > 0

st.subheader("4) GAD-7")
gad_questions = [
    "รู้สึกกังวล กระวนกระวาย หรือเครียด",
    "ไม่สามารถหยุดหรือควบคุมความกังวลได้",
    "กังวลมากเกินไปในหลายเรื่อง",
    "ผ่อนคลายได้ยาก",
    "กระสับกระส่ายจนอยู่นิ่งได้ยาก",
    "หงุดหงิดง่าย",
    "กลัวว่าจะมีเรื่องร้ายเกิดขึ้น"
]
gad_scores = []
for q in gad_questions:
    gad_scores.append(st.selectbox(q, [0, 1, 2, 3], format_func=lambda x: ["ไม่มีเลย", "บางวัน", "มากกว่า 7 วัน", "เกือบทุกวัน"][x], key=q))
gad7 = sum(gad_scores)

# simple signal score
signal_score = 0
signal_score += max(0, 10 - sleep) // 2
signal_score += fatigue // 2
signal_score += 3 if class_attendance < 60 else 1 if class_attendance < 80 else 0
signal_score += 2 if exercise_freq == 0 else 1 if exercise_freq <= 1 else 0
signal_score += loneliness // 2
signal_score += min(4, repeated_visits_self)

color, advice = signal_level(signal_score, phq9, gad7, suicidal)

st.subheader("5) ผลคัดกรองเบื้องต้น")
st.metric("PHQ-9", phq9, phq_level(phq9))
st.metric("GAD-7", gad7, gad_level(gad7))
st.metric("Psychosomatic Signal Score", signal_score, color)

if color == "แดง":
    st.error(advice)
elif color == "เหลือง":
    st.warning(advice)
else:
    st.success(advice)

if suicidal:
    st.error("พบคำตอบเกี่ยวกับความคิดทำร้ายตนเอง ควรประเมินความปลอดภัยทันทีและไม่ควรปล่อยให้อยู่ลำพังหากมีความเสี่ยง")

note = st.text_area("บันทึกเพิ่มเติมของพยาบาล/แพทย์")

st.subheader("6) Action Plan via GPT")

student_summary = f"""
รหัสนักศึกษา: {student_id}
คณะ: {faculty}
ชั้นปี: {year}
อาการสำคัญ: {visit_reason}
Sleep score: {sleep}/10
Fatigue score: {fatigue}/10
Class attendance: {class_attendance}%
Exercise: {exercise_freq} วัน/สัปดาห์
Loneliness: {loneliness}/10
Repeated clinic visits: {repeated_visits_self}
PHQ-9: {phq9} ({phq_level(phq9)})
GAD-7: {gad7} ({gad_level(gad7)})
Suicidal item positive: {suicidal}
Psychosomatic signal score: {signal_score}
Traffic level: {color}
Advice: {advice}
"""

if st.button("🤖 สร้าง Action Plan ด้วย GPT"):
    if not student_id.strip():
        st.warning("กรุณากรอกรหัสนักศึกษาก่อนสร้าง Action Plan")
    else:
        with st.spinner("กำลังสร้างข้อความแนะนำ..."):
            action_plan = generate_gpt_action(student_summary)
            if action_plan.strip():
                st.session_state["action_plan"] = action_plan
                st.success("สร้าง Action Plan สำเร็จ")
            else:
                st.session_state["action_plan"] = ""
                st.error("Action Plan ว่างเปล่า กรุณาตรวจสอบ OpenAI model/API key หรือดูข้อความ error ด้านบน")

if st.session_state.get("action_plan", "").strip():
    st.text_area(
        "Action Plan ที่สามารถ copy ส่งต่อ/บันทึกได้",
        st.session_state["action_plan"],
        height=350
    )
else:
    st.info("กดปุ่ม 🤖 เพื่อสร้างข้อความ check-in, counseling offer และช่องทางติดต่อพยาบาล")

if st.button("💾 บันทึกข้อมูลลง GitHub CSV"):
    if not student_id.strip():
        st.error("กรุณากรอกรหัสนักศึกษา")
        st.stop()

    old_df, sha = load_csv_from_github()

    row = {
        "timestamp_bkk": now_bkk(),
        "faculty": faculty,
        "year": year,
        "visit_reason": visit_reason,
        "sleep_score": sleep,
        "fatigue_score": fatigue,
        "class_attendance_percent": class_attendance,
        "exercise_days_per_week": exercise_freq,
        "loneliness_score": loneliness,
        "repeated_visits_self_report": repeated_visits_self,
        "phq9": phq9,
        "phq9_level": phq_level(phq9),
        "gad7": gad7,
        "gad7_level": gad_level(gad7),
        "suicidal_item_positive": suicidal,
        "psychosomatic_signal_score": signal_score,
        "traffic_level": color,
        "advice": advice,
        "clinical_note": note,
        "pdpa_consent": True,
        "gpt_action_plan": st.session_state.get("action_plan", "")
    }

    new_df = pd.DataFrame([row], index=[student_id])
    new_df.index.name = "student_id"

    if old_df.empty:
        combined = new_df
    else:
        combined = pd.concat([old_df, new_df], axis=0)

    ok = save_csv_to_github(combined, sha)
    if ok:
        st.success("บันทึกข้อมูลสำเร็จแล้ว")
