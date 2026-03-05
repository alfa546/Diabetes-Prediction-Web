from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'diabetesai-admin-secret-2025-xK9mP'

# ── Admin credentials (change these!) ──
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'doctor@123'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'diabetes_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
HISTORY_FILE = os.path.join(BASE_DIR, 'patient_history.json')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(records):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(records, f, indent=2)

def get_ai_advice(pd_data, prediction, probability, api_key=None):
    glucose = pd_data['glucose']; bmi = pd_data['bmi']
    age = pd_data['age'];         bp  = pd_data['blood_pressure']
    risk = round(probability * 100, 1)

    if api_key and api_key.strip() and not api_key.startswith("YOUR"):
        try:
            import urllib.request
            prompt = f"""Patient: Age {age}, Glucose {glucose} mg/dL, BMI {bmi}, BP {bp} mmHg, Risk {risk}%, {"DIABETIC" if prediction==1 else "NOT DIABETIC"}.
Provide JSON only (no markdown) with keys: diet_en, diet_ur, exercise_en, exercise_ur, lifestyle_en, lifestyle_ur, doctor_advice_en, doctor_advice_ur, message_en, message_ur
Each key an array of 3 strings except messages (single string). Urdu in Urdu script."""
            payload = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":1500,
                "messages":[{"role":"user","content":prompt}]}).encode()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type":"application/json","x-api-key":api_key,"anthropic-version":"2023-06-01"})
            with urllib.request.urlopen(req, timeout=15) as r:
                res = json.loads(r.read())
                text = res['content'][0]['text']
                s,e = text.find('{'), text.rfind('}')+1
                if s!=-1 and e>s:
                    ai = json.loads(text[s:e]); ai['ai_powered']=True; return ai
        except Exception as ex:
            print("Claude API:", ex)

    if prediction == 1:
        return {
            "ai_powered": False,
            "diet_en":["Reduce sugar & refined carbs — glucose is "+str(glucose)+" mg/dL","Eat more vegetables, legumes, and whole grains daily","Small meals every 3-4 hours to keep blood sugar stable"],
            "diet_ur":["چینی اور میدہ کم کریں — گلوکوز "+str(glucose)+" ہے","روزانہ سبزیاں، دالیں اور اناج کھائیں","ہر 3-4 گھنٹے میں تھوڑا کھائیں"],
            "exercise_en":["Walk 30 minutes every morning — most effective habit","Light yoga or stretching 3x per week","Never sit for more than 1 hour continuously"],
            "exercise_ur":["روزانہ صبح 30 منٹ چلیں","ہفتے میں 3 بار ہلکی یوگا کریں","1 گھنٹے سے زیادہ مسلسل نہ بیٹھیں"],
            "lifestyle_en":["Monitor blood sugar weekly","Sleep 7-8 hours — poor sleep worsens diabetes","Reduce stress with deep breathing"],
            "lifestyle_ur":["ہفتے میں ایک بار بلڈ شوگر چیک کریں","7-8 گھنٹے سوئیں","گہرا سانس لے کر ذہنی سکون پائیں"],
            "doctor_advice_en":"WARNING: Please see a doctor within 1-2 weeks. Get HbA1c test done.",
            "doctor_advice_ur":"خبردار: 1-2 ہفتوں میں ڈاکٹر سے ملیں۔ HbA1c ٹیسٹ کروائیں۔",
            "message_en":"Millions manage diabetes well with the right lifestyle. You can too!",
            "message_ur":"لاکھوں لوگ صحیح طرز زندگی سے ذیابیطس کنٹرول کرتے ہیں۔ آپ بھی کر سکتے ہیں!"
        }
    else:
        return {
            "ai_powered": False,
            "diet_en":["Keep eating balanced meals — vegetables, fruits, whole grains","Limit sugary drinks and ultra-processed foods","Drink 8-10 glasses of water daily"],
            "diet_ur":["متوازن کھانا جاری رکھیں — سبزیاں، پھل، اناج","میٹھے مشروبات کم کریں","روزانہ 8-10 گلاس پانی پئیں"],
            "exercise_en":["30 minutes of exercise 5 days a week","Mix cardio with strength training","Stay active throughout the day"],
            "exercise_ur":["ہفتے میں 5 دن 30 منٹ ورزش","Cardio اور strength exercise دونوں کریں","پورے دن متحرک رہیں"],
            "lifestyle_en":["Annual diabetes screening after age 35","Maintain healthy BMI 18.5-24.9","Avoid smoking"],
            "lifestyle_ur":["35 سال کے بعد سالانہ ذیابیطس چیک کروائیں","صحت مند وزن رکھیں","سگریٹ سے پرہیز کریں"],
            "doctor_advice_en":"You are healthy! Annual routine checkup is enough.",
            "doctor_advice_ur":"آپ صحت مند ہیں! سالانہ routine checkup کافی ہے۔",
            "message_en":"Excellent results! Keep up your healthy habits!",
            "message_ur":"شاباش! صحت مند عادات جاری رکھیں!"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import pandas as pd
        d = request.json
        feats = [
            float(d.get('pregnancies',0)), float(d['glucose']),
            float(d['blood_pressure']),    float(d.get('skin_thickness',20)),
            float(d.get('insulin',80)),    float(d['bmi']),
            float(d.get('dpf',0.5)),       float(d['age'])
        ]
        feat_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        fs = scaler.transform(pd.DataFrame([feats], columns=feat_names))
        pred  = int(model.predict(fs)[0])
        prob  = float(model.predict_proba(fs)[0][1])
        levels = ["low","medium","high"]
        idx = 0 if prob<0.3 else 1 if prob<0.6 else 2
        risk_level = levels[idx]
        risk_en    = ["Low Risk","Medium Risk","High Risk"][idx]
        risk_ur    = ["کم خطرہ","درمیانہ خطرہ","زیادہ خطرہ"][idx]
        advice = get_ai_advice({'glucose':feats[1],'bmi':feats[5],'age':feats[7],'blood_pressure':feats[2]}, pred, prob, d.get('api_key',''))
        record = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "name": d.get('patient_name','Anonymous'),
            "age": feats[7], "glucose": feats[1], "bmi": feats[5],
            "blood_pressure": feats[2], "prediction": pred,
            "probability": round(prob*100,1), "risk_level": risk_level,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        }
        hist = load_history(); hist.insert(0, record); save_history(hist[:100])
        return jsonify({
            'success':True, 'prediction':pred, 'probability':round(prob*100,1),
            'risk_level':risk_level, 'risk_label_en':risk_en, 'risk_label_ur':risk_ur,
            'diabetic_en':"Diabetic" if pred else "Not Diabetic",
            'diabetic_ur':"ذیابیطس ہے" if pred else "ذیابیطس نہیں",
            'advice':advice, 'record':record, 'timestamp': record['timestamp']
        })
    except KeyError as e:
        return jsonify({'success':False,'error':f'Missing: {e}'}),400
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/analyze-diet', methods=['POST'])
def analyze_diet():
    try:
        d = request.json
        foods = d.get('foods', [])   # list of {name, quantity, unit}
        api_key = d.get('api_key', '')

        # ── Built-in sugar/nutrition database (per 100g unless noted) ──
        FOOD_DB = {
            # Pakistani/common foods — sugar_g, gi (glycemic index), carbs_g, calories
            'rice':          {'sugar':0.1,  'gi':72, 'carbs':28,  'cal':130,  'unit':'g'},
            'chawal':        {'sugar':0.1,  'gi':72, 'carbs':28,  'cal':130,  'unit':'g'},
            'roti':          {'sugar':0.3,  'gi':62, 'carbs':43,  'cal':265,  'unit':'piece', 'piece_g':35},
            'chapati':       {'sugar':0.3,  'gi':62, 'carbs':43,  'cal':265,  'unit':'piece', 'piece_g':35},
            'naan':          {'sugar':2.0,  'gi':71, 'carbs':50,  'cal':300,  'unit':'piece', 'piece_g':90},
            'bread':         {'sugar':4.9,  'gi':70, 'carbs':49,  'cal':265,  'unit':'slice', 'piece_g':30},
            'paratha':       {'sugar':0.5,  'gi':66, 'carbs':40,  'cal':330,  'unit':'piece', 'piece_g':80},
            'dal':           {'sugar':1.8,  'gi':29, 'carbs':20,  'cal':116,  'unit':'g'},
            'daal':          {'sugar':1.8,  'gi':29, 'carbs':20,  'cal':116,  'unit':'g'},
            'chicken':       {'sugar':0.0,  'gi':0,  'carbs':0,   'cal':165,  'unit':'g'},
            'mutton':        {'sugar':0.0,  'gi':0,  'carbs':0,   'cal':250,  'unit':'g'},
            'beef':          {'sugar':0.0,  'gi':0,  'carbs':0,   'cal':250,  'unit':'g'},
            'fish':          {'sugar':0.0,  'gi':0,  'carbs':0,   'cal':140,  'unit':'g'},
            'egg':           {'sugar':0.4,  'gi':0,  'carbs':0.6, 'cal':78,   'unit':'piece', 'piece_g':50},
            'anda':          {'sugar':0.4,  'gi':0,  'carbs':0.6, 'cal':78,   'unit':'piece', 'piece_g':50},
            'milk':          {'sugar':4.8,  'gi':31, 'carbs':4.8, 'cal':61,   'unit':'ml'},
            'doodh':         {'sugar':4.8,  'gi':31, 'carbs':4.8, 'cal':61,   'unit':'ml'},
            'chai':          {'sugar':6.0,  'gi':35, 'carbs':8,   'cal':45,   'unit':'cup', 'piece_g':200},
            'tea':           {'sugar':6.0,  'gi':35, 'carbs':8,   'cal':45,   'unit':'cup', 'piece_g':200},
            'sugar':         {'sugar':100,  'gi':100,'carbs':100, 'cal':387,  'unit':'g'},
            'cheeni':        {'sugar':100,  'gi':100,'carbs':100, 'cal':387,  'unit':'g'},
            'banana':        {'sugar':12.2, 'gi':51, 'carbs':23,  'cal':89,   'unit':'piece', 'piece_g':120},
            'kela':          {'sugar':12.2, 'gi':51, 'carbs':23,  'cal':89,   'unit':'piece', 'piece_g':120},
            'apple':         {'sugar':10.4, 'gi':36, 'carbs':14,  'cal':52,   'unit':'piece', 'piece_g':150},
            'mango':         {'sugar':14.8, 'gi':56, 'carbs':15,  'cal':60,   'unit':'piece', 'piece_g':200},
            'aam':           {'sugar':14.8, 'gi':56, 'carbs':15,  'cal':60,   'unit':'piece', 'piece_g':200},
            'orange':        {'sugar':9.4,  'gi':43, 'carbs':12,  'cal':47,   'unit':'piece', 'piece_g':130},
            'potato':        {'sugar':0.9,  'gi':85, 'carbs':17,  'cal':77,   'unit':'g'},
            'aloo':          {'sugar':0.9,  'gi':85, 'carbs':17,  'cal':77,   'unit':'g'},
            'biryani':       {'sugar':1.2,  'gi':70, 'carbs':30,  'cal':200,  'unit':'g'},
            'nihari':        {'sugar':0.5,  'gi':35, 'carbs':5,   'cal':220,  'unit':'g'},
            'halwa':         {'sugar':30,   'gi':80, 'carbs':45,  'cal':350,  'unit':'g'},
            'kheer':         {'sugar':15,   'gi':65, 'carbs':22,  'cal':150,  'unit':'g'},
            'mithai':        {'sugar':50,   'gi':85, 'carbs':65,  'cal':380,  'unit':'g'},
            'samosa':        {'sugar':1.5,  'gi':55, 'carbs':22,  'cal':260,  'unit':'piece', 'piece_g':70},
            'cola':          {'sugar':10.6, 'gi':65, 'carbs':10.6,'cal':41,   'unit':'ml'},
            'juice':         {'sugar':10,   'gi':60, 'carbs':10,  'cal':45,   'unit':'ml'},
            'yogurt':        {'sugar':4.7,  'gi':14, 'carbs':3.6, 'cal':59,   'unit':'g'},
            'dahi':          {'sugar':4.7,  'gi':14, 'carbs':3.6, 'cal':59,   'unit':'g'},
            'lassi':         {'sugar':8,    'gi':30, 'carbs':12,  'cal':70,   'unit':'ml'},
            'saag':          {'sugar':0.5,  'gi':15, 'carbs':3,   'cal':28,   'unit':'g'},
            'salad':         {'sugar':1.5,  'gi':10, 'carbs':3,   'cal':20,   'unit':'g'},
            'default':       {'sugar':5,    'gi':50, 'carbs':10,  'cal':100,  'unit':'g'},
        }

        analyzed = []
        total_sugar = 0
        total_carbs = 0
        total_cal   = 0
        total_gi_weighted = 0
        total_gi_carbs    = 0

        for item in foods:
            name = item.get('name','').lower().strip()
            qty  = float(item.get('quantity', 1))
            unit = item.get('unit','').lower().strip()

            db = FOOD_DB.get(name, FOOD_DB['default'])
            known = name in FOOD_DB

            # compute effective grams
            if db['unit'] in ('piece','slice','cup') and 'piece_g' in db:
                g = qty * db['piece_g']
            elif unit in ('ml','g'):
                g = qty
            elif unit in ('cup',):
                g = qty * 240
            elif unit in ('tbsp','tablespoon'):
                g = qty * 15
            elif unit in ('tsp','teaspoon'):
                g = qty * 5
            elif unit in ('piece','pcs','slice'):
                g = qty * db.get('piece_g', 100)
            else:
                g = qty * 100  # assume grams

            sugar_g = round(g / 100 * db['sugar'], 2)
            carbs_g = round(g / 100 * db['carbs'], 2)
            cal_v   = round(g / 100 * db['cal'],   1)
            gi      = db['gi']

            # risk tag
            if sugar_g > 15: risk = 'high';   risk_color = 'red'
            elif sugar_g > 5: risk = 'medium'; risk_color = 'amber'
            else:             risk = 'low';    risk_color = 'green'

            analyzed.append({
                'name': item.get('name'),
                'quantity': qty,
                'unit': unit or db['unit'],
                'grams': round(g,1),
                'sugar_g': sugar_g,
                'carbs_g': carbs_g,
                'calories': cal_v,
                'gi': gi,
                'risk': risk,
                'risk_color': risk_color,
                'known': known,
                'swap': get_swap(name, risk)
            })
            total_sugar += sugar_g
            total_carbs += carbs_g
            total_cal   += cal_v
            if carbs_g > 0:
                total_gi_weighted += gi * carbs_g
                total_gi_carbs    += carbs_g

        avg_gi = round(total_gi_weighted / total_gi_carbs) if total_gi_carbs > 0 else 50

        # overall diet risk
        if total_sugar > 50:   diet_risk = 'high';   diet_msg_en = 'Your daily sugar intake is dangerously high!';   diet_msg_ur = 'آپ کی روزانہ چینی بہت زیادہ ہے!'
        elif total_sugar > 25: diet_risk = 'medium'; diet_msg_en = 'Your sugar intake is above recommended levels.'; diet_msg_ur = 'آپ کی چینی تجویز کردہ حد سے زیادہ ہے۔'
        else:                  diet_risk = 'low';    diet_msg_en = 'Your sugar intake looks manageable!';            diet_msg_ur = 'آپ کی چینی کی مقدار قابو میں ہے!'

        # recommended: <25g sugar/day for diabetics, <50g normal
        rec_limit = 25

        # AI-powered detailed advice
        advice = get_diet_advice(analyzed, total_sugar, total_carbs, avg_gi, api_key)

        return jsonify({
            'success': True,
            'analyzed': analyzed,
            'total_sugar': round(total_sugar, 1),
            'total_carbs': round(total_carbs, 1),
            'total_calories': round(total_cal, 1),
            'avg_gi': avg_gi,
            'diet_risk': diet_risk,
            'diet_msg_en': diet_msg_en,
            'diet_msg_ur': diet_msg_ur,
            'recommended_limit': rec_limit,
            'advice': advice
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_swap(name, risk):
    """Suggest a healthier swap for high-risk foods."""
    swaps = {
        'sugar': 'Use stevia or honey (small amount)',
        'cheeni': 'Stevia ya thodi shahad use karein',
        'rice': 'Try brown rice or cauliflower rice',
        'chawal': 'Brown chawal ya phool goobi chawal try karein',
        'naan': 'Whole wheat roti instead',
        'mithai': 'Fresh fruit instead of sweets',
        'halwa': 'Dates (khajoor) in small quantity',
        'cola': 'Plain water or coconut water',
        'juice': 'Eat whole fruit instead of juice',
        'samosa': 'Baked vegetable roll instead',
        'biryani': 'Brown rice pulao with veggies',
        'kheer': 'Low-sugar kheer with stevia',
        'mango': 'Eat half portion or choose guava',
        'aam': 'Adha aam khayen ya amrood chunein',
        'banana': 'Apple or pear instead (lower GI)',
        'kela': 'Seb ya nashpati behtar hai',
        'potato': 'Sweet potato (shakarkandi) instead',
        'aloo': 'Shakarkandi behtar option hai',
        'paratha': 'Plain roti with less ghee',
        'lassi': 'Plain dahi (unsweetened) better',
        'bread': 'Whole grain or multigrain bread',
    }
    if risk == 'low': return None
    return swaps.get(name)


def get_diet_advice(analyzed, total_sugar, total_carbs, avg_gi, api_key=None):
    """Get AI diet advice — Claude API or smart fallback."""

    high_risk_foods = [a['name'] for a in analyzed if a['risk'] == 'high']
    medium_risk_foods = [a['name'] for a in analyzed if a['risk'] == 'medium']

    if api_key and api_key.strip() and not api_key.startswith('YOUR'):
        try:
            import urllib.request
            food_list = ', '.join([f"{a['quantity']} {a['unit']} {a['name']} ({a['sugar_g']}g sugar)" for a in analyzed])
            prompt = f"""A diabetes patient ate these foods today: {food_list}
Total sugar: {total_sugar}g, Total carbs: {total_carbs}g, Average GI: {avg_gi}

Provide JSON only (no markdown) with:
- summary_en: 2-sentence overall diet assessment
- summary_ur: same in Urdu script
- warnings_en: array of 3 specific warnings about high-sugar foods
- warnings_ur: same in Urdu
- swaps_en: array of 3 specific food swap suggestions
- swaps_ur: same in Urdu  
- tomorrow_en: array of 3 diet tips for tomorrow
- tomorrow_ur: same in Urdu"""
            payload = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":1200,
                "messages":[{"role":"user","content":prompt}]}).encode()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type":"application/json","x-api-key":api_key,"anthropic-version":"2023-06-01"})
            with urllib.request.urlopen(req, timeout=15) as r:
                res = json.loads(r.read())
                text = res['content'][0]['text']
                s, e = text.find('{'), text.rfind('}')+1
                if s != -1 and e > s:
                    ai = json.loads(text[s:e])
                    ai['ai_powered'] = True
                    return ai
        except Exception as ex:
            print("Diet AI error:", ex)

    # Smart fallback
    if total_sugar > 50:
        summary_en = f"Your daily sugar intake of {total_sugar}g is critically high — nearly double the safe limit for diabetics (25g/day). Immediate diet changes needed."
        summary_ur = f"آپ کی روزانہ چینی {total_sugar}g ہے جو ذیابیطس کے مریضوں کی حد (25g) سے دوگنی ہے۔ فوری تبدیلی ضروری ہے۔"
    elif total_sugar > 25:
        summary_en = f"Your sugar intake ({total_sugar}g) exceeds the diabetic-safe limit of 25g/day. Some foods need to be replaced."
        summary_ur = f"آپ کی چینی ({total_sugar}g) محفوظ حد 25g سے زیادہ ہے۔ کچھ کھانے تبدیل کریں۔"
    else:
        summary_en = f"Good news! Your sugar intake ({total_sugar}g) is within safe limits. Keep maintaining this balance."
        summary_ur = f"خوشخبری! آپ کی چینی ({total_sugar}g) محفوظ حد میں ہے۔ یہ توازن برقرار رکھیں۔"

    warnings = []
    warnings_ur = []
    if high_risk_foods:
        warnings.append(f"HIGH ALERT: {', '.join(high_risk_foods)} contain very high sugar — avoid or drastically reduce")
        warnings_ur.append(f"خطرہ: {', '.join(high_risk_foods)} میں بہت زیادہ چینی ہے — بالکل کم کریں")
    if avg_gi > 60:
        warnings.append(f"Your diet's average GI is {avg_gi} (high) — this spikes blood sugar rapidly")
        warnings_ur.append(f"آپ کی خوراک کا اوسط GI {avg_gi} ہے جو بلڈ شوگر تیزی سے بڑھاتا ہے")
    if total_carbs > 150:
        warnings.append(f"Total carbs ({total_carbs}g) are very high — aim for under 100g for diabetics")
        warnings_ur.append(f"کل carbs ({total_carbs}g) بہت زیادہ ہیں — ذیابیطس میں 100g سے کم رکھیں")
    while len(warnings) < 3:
        warnings.append("Eat smaller portions spread across 5-6 small meals instead of 3 large ones")
        warnings_ur.append("3 بڑے کھانوں کی بجائے 5-6 چھوٹے کھانے کھائیں")

    return {
        'ai_powered': False,
        'summary_en': summary_en,
        'summary_ur': summary_ur,
        'warnings_en': warnings[:3],
        'warnings_ur': warnings_ur[:3],
        'swaps_en': [
            f"Replace high-GI foods with low-GI alternatives (brown rice, whole wheat roti)",
            "Use stevia instead of sugar in chai/tea",
            "Choose whole fruits over juices — fiber slows sugar absorption"
        ],
        'swaps_ur': [
            "زیادہ GI والی چیزیں کم GI سے بدلیں (براؤن چاول، گندم کی روٹی)",
            "چائے میں چینی کی جگہ stevia استعمال کریں",
            "جوس کی بجائے پورا پھل کھائیں — فائبر چینی کو سست کرتا ہے"
        ],
        'tomorrow_en': [
            "Start tomorrow with eggs + vegetables — zero sugar, high protein breakfast",
            f"Your target: keep total sugar under 25g tomorrow (today was {total_sugar}g)",
            "Drink water before each meal — reduces appetite and sugar cravings"
        ],
        'tomorrow_ur': [
            "کل صبح انڈے اور سبزی کھائیں — صفر چینی، بھرپور پروٹین",
            f"کل کا ہدف: چینی 25g سے کم رکھیں (آج {total_sugar}g تھی)",
            "ہر کھانے سے پہلے پانی پئیں — بھوک اور چینی کی طلب کم ہوتی ہے"
        ]
    }



@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username','')
        password = request.form.get('password','')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['admin'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/history', methods=['GET'])
def history():
    if not session.get('logged_in'):
        return jsonify({'error':'Unauthorized'}), 401
    return jsonify(load_history())

@app.route('/history/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    if not session.get('logged_in'):
        return jsonify({'error':'Unauthorized'}), 401
    hist = [r for r in load_history() if r['id'] != record_id]
    save_history(hist)
    return jsonify({'success':True})

@app.route('/health')
def health_check():
    return jsonify({'status':'running','accuracy':'94.5%','records':len(load_history())})

if __name__ == '__main__':
    print("DiabetesAI v2 running at http://localhost:5000")
    print("Doctor Dashboard at http://localhost:5000/dashboard")
    app.run(debug=True, host='0.0.0.0', port=5000)
