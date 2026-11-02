import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

from fastapi.responses import FileResponse


MODEL_FILE = 'onco_burden_model.joblib'







# ==============================================================================
# 1. تدريب وحفظ نموذج تقدير العبء الورمي بالجسم (Tumor Burden Engine)
# ==============================================================================
if not os.path.exists(MODEL_FILE):
    print("⏳ جاري تدريب خوارزمية تقدير العبء الورمي ومراحل TNM...")
    np.random.seed(42)
    n = 5000

    t_stage = np.random.choice([1, 2, 3, 4], n)          # حجم الورم وتغلغله الموضعي
    n_stage = np.random.choice([0, 1, 2, 3], n)          # إصابة العقد اللمفاوية
    m_stage = np.random.choice([0, 1], n, p=[0.75, 0.25]) # الانتشار البعيد (M0 أو M1)
    marker_ratio = np.random.uniform(0.5, 30.0, n)       # نسبة الواسم الورمي إلى حده الطبيعي
    performance_score = np.random.choice([0, 1, 2, 3], n) # مقياس ECOG للحالة الوظيفية
    age = np.random.randint(18, 85, n)

    # معادلة حساب العبء الورمي المئوي في الجسم (Tumor Burden %)
    # وجود M1 يرفع العبء فوراً فوق 65%، والعقد اللمفاوية والواسمات تزيد النسبة تدريجياً
    burden = (
        (m_stage * 55.0) +
        (t_stage * 6.5) +
        (n_stage * 5.0) +
        (np.log1p(marker_ratio) * 4.5) +
        (performance_score * 2.0) +
        np.random.normal(0, 2.5, n)
    )
    burden = np.clip(burden, 5.0, 99.5).round(1)

    features = ['t_stage', 'n_stage', 'm_stage', 'marker_ratio', 'performance_score', 'age']
    X = pd.DataFrame({
        't_stage': t_stage,
        'n_stage': n_stage,
        'm_stage': m_stage,
        'marker_ratio': marker_ratio,
        'performance_score': performance_score,
        'age': age
    })

    model = RandomForestRegressor(n_estimators=140, random_state=42, max_depth=10)
    model.fit(X, burden)
    joblib.dump({'model': model, 'features': features}, MODEL_FILE)
    print("✅ تم حفظ عقل OncoAI بنجاح!")

# ==============================================================================
# 2. قاعدة المعرفة السريرية لكافة سرطانات الجسم
# ==============================================================================
ONCOLOGY_DATABASE = {
    # 1. الجهاز الهضمي
    "colon": {
        "name": "سرطان القولون والمستقيم (Colorectal)",
        "marker_name": "CEA (المستضد الجنيني)", "normal_max": 5.0,
        "early_protocol": "استئصال جراحي جذري + كيماوي مكمل (Capecitabine أو 5-FU/LV)",
        "adv_protocol": "بروتوكول FOLFOX (Oxaliplatin + 5-FU + Leucovorin) أو FOLFIRI + Bevacizumab",
        "prevention": "تنظير القولون كل 10 سنوات بدءاً من سن 45، وإزالة السلائل اللحمية، وحمية غنية بالألياف"
    },
    "gastric": {
        "name": "سرطان المعدة الغدي (Gastric Adenocarcinoma)",
        "marker_name": "CA 72-4 / CEA", "normal_max": 6.9,
        "early_protocol": "استئصال المعدة الجذري مع تجريف العقد D2",
        "adv_protocol": "بروتوكول FLOT (Docetaxel + Oxaliplatin + 5-FU/Leucovorin) ± Trastuzumab (لـ HER2+)",
        "prevention": "علاج جرثومة المعدة H. Pylori فوراً بالمضادات، وتجنب الأطعمة شديدة الملوحة والمدخنة"
    },
    "pancreatic": {
        "name": "سرطان البنكرياس (Pancreatic Adenocarcinoma)",
        "marker_name": "CA 19-9", "normal_max": 37.0,
        "early_protocol": "جراحة ويبل الجذرية (Whipple Procedure) متبوعة بـ Gemcitabine",
        "adv_protocol": "بروتوكول mFOLFIRINOX المكثف أو (Gemcitabine + nab-Paclitaxel)",
        "prevention": "الإقلاع التام عن التدخين، ومراقبة التكيسات البنكرياسية العائلية، وتجنب السمنة"
    },
    "liver": {
        "name": "سرطان الكبد الأولي (Hepatocellular Carcinoma)",
        "marker_name": "AFP (ألفا فيتو بروتين)", "normal_max": 10.0,
        "early_protocol": "استئصال كبدي جزئي أو زراعة كبد وفق معايير ميلانو",
        "adv_protocol": "المناعي المزدوج (Atezolizumab + Bevacizumab) أو Lenvatinib/Sorafenib",
        "prevention": "لقاح التهاب الكبد B، وعلاج فيروس C بالأدوية الفيروسية المباشرة، والامتناع عن الكحول"
    },
    "esophagus": {
        "name": "سرطان المريء (Esophageal Carcinoma)",
        "marker_name": "CEA / SCC-Ag", "normal_max": 5.0,
        "early_protocol": "استئصال المريء الجراحي (Esophagectomy)",
        "adv_protocol": "بروتوكول CROSS (إشعاع تزامني مع Carboplatin + Paclitaxel)",
        "prevention": "علاج الارتجاع المريئي المزمن (GERD) لمنع تحول باريت المخاطي، والامتناع عن التدخين"
    },

    # 2. الجهاز التنفسي والقفص الصدري
    "lung_nsclc": {
        "name": "سرطان الرئة غير صغير الخلايا (NSCLC)",
        "marker_name": "CEA / CYFRA 21-1", "normal_max": 3.3,
        "early_protocol": "استئصال الفص الرئوي الجراحي (Lobectomy) + علاج مناعي/موجه تكميلي",
        "adv_protocol": "حاصرات PD-L1 (Pembrolizumab) مع (Cisplatin + Pemetrexed) أو Osimertinib لطفرة EGFR",
        "prevention": "الامتناع التام عن التدخين، وفحص المقطعية منخفضة الجرعة السنوي (LDCT) للمدخنين"
    },
    "lung_sclc": {
        "name": "سرطان الرئة صغير الخلايا (SCLC - العدواني)",
        "marker_name": "NSE (إينولاز عصبي نوعي)", "normal_max": 16.3,
        "early_protocol": "كيماوي إشعاعي تزامني مكثف مع إشعاع وقائي للدماغ (PCI)",
        "adv_protocol": "بروتوكول (Carboplatin + Etoposide + Atezolizumab/Durvalumab)",
        "prevention": "تجنب التبغ المباشر والسلبي وغاز الرادون المنزلي والأسبستوس"
    },

    # 3. الثدي وأمراض النساء
    "breast_hr": {
        "name": "سرطان الثدي الهرموني (ER+/PR+/HER2-)",
        "marker_name": "CA 15-3", "normal_max": 30.0,
        "early_protocol": "استئصال الورم أو الثدي + إشعاع + علاج هرموني (Tamoxifen أو Letrozole)",
        "adv_protocol": "مثبطات CDK4/6 (Ribociclib أو Palbociclib) مدمجة مع العلاج الهرموني ومثبطات Aromatase",
        "prevention": "الماموغرام السنوي الدوري من سن 40، والفحص الجيني لطفرات BRCA1/2 للعائلات المعرضة"
    },
    "breast_her2": {
        "name": "سرطان الثدي الإيجابي لـ HER2 (HER2+)",
        "marker_name": "CA 15-3 / HER2 ECD", "normal_max": 30.0,
        "early_protocol": "جراحة الثدي مع بروتوكول Trastuzumab لمدة سنة كاملة",
        "adv_protocol": "بروتوكول TCHP (Docetaxel + Carboplatin + Trastuzumab + Pertuzumab)",
        "prevention": "الفحص الذاتي الشهري والماموغرام، وممارسة الرياضة والحد من العلاجات الهرمونية البديلة"
    },
    "breast_tnbc": {
        "name": "سرطان الثدي ثلاثي السلبية (Triple Negative)",
        "marker_name": "CA 15-3", "normal_max": 30.0,
        "early_protocol": "بروتوكول AC-T الكيماوي المكثف (Doxorubicin + Cyclophosphamide ثم Paclitaxel)",
        "adv_protocol": "بروتوكول AC-T مدعوماً بالعلاج المناعي بحاصرات PD-1 (Pembrolizumab)",
        "prevention": "المتابعة الوراثية الجينية، والكشف بالماموغرام والرنين المغناطيسي لحاملي BRCA"
    },
    "ovarian": {
        "name": "سرطان المبيض الظهاري (Ovarian Carcinoma)",
        "marker_name": "CA-125 / HE4", "normal_max": 35.0,
        "early_protocol": "استئصال الرحم والملحقات وتجريف الثرب والورم الجذري الأقصى",
        "adv_protocol": "بروتوكول (Carboplatin + Paclitaxel) متبوعاً بصيانة مثبطات PARP مثل Olaparib",
        "prevention": "استئصال المبيض والنفير الوقائي لحاملي طفرة BRCA الشديدة، والسونار الحوضي الدوري"
    },
    "cervical": {
        "name": "سرطان عنق الرحم (Cervical Cancer)",
        "marker_name": "SCC-Ag", "normal_max": 1.5,
        "early_protocol": "استئصال الرحم الجذري مع تجريف العقد الحوضية (Wertheim Operation)",
        "adv_protocol": "علاج إشعاعي تزامني مع Cisplatin، أو (Paclitaxel + Carboplatin + Bevacizumab)",
        "prevention": "أخذ لقاح فيروس الورم الحليمي (Gardasil 9) ومسحة عنق الرحم (Pap Smear) كل 3 سنوات"
    },

    # 4. الجهاز البولي والتناسلي الذكري
    "prostate": {
        "name": "سرطان البروستاتا الغدي (Prostate Adenocarcinoma)",
        "marker_name": "Total PSA", "normal_max": 4.0,
        "early_protocol": "مراقبة نشطة أو استئصال جذري بالروبوت أو إشعاع تزامني مع خافضات التيستوستيرون",
        "adv_protocol": "الحرمان الأندروجيني (Enzalutamide أو Abiraterone) + Docetaxel أو حقن Lutetium-177",
        "prevention": "تحليل PSA السنوي وفحص المستقيم للرجال فوق سن الخمسين (أو 45 مع وجود تاريخ عائلي)"
    },
    "bladder": {
        "name": "سرطان المثانة البولية الظهاري (Bladder Urothelial)",
        "marker_name": "NMP22 / Urine Cytology", "normal_max": 10.0,
        "early_protocol": "استئصال بالمنظار (TURBT) يليه غسيل موضعي بلقاح السل البكتيري (Intravesical BCG)",
        "adv_protocol": "استئصال جذري للمثانة مسبوقاً بـ Gemcitabine/Cisplatin، أو العلاج بـ Enfortumab Vedotin",
        "prevention": "الإقلاع الصارم عن التدخين (المسبب لـ 50% من الحالات) وتجنب التعرض للملونات الصناعية"
    },
    "kidney": {
        "name": "سرطان الخلايا الكلوية (Renal Cell Carcinoma)",
        "marker_name": "متابعة التصوير والـ LDH", "normal_max": 250.0,
        "early_protocol": "استئصال جزئي أو كلي للكلية (Nephrectomy) - لا يستجيب للكيماوي الكلاسيكي",
        "adv_protocol": "المناعي المزدوج (Nivolumab + Ipilimumab) أو الموجه (Pembrolizumab + Axitinib)",
        "prevention": "ضبط ضغط الدم والسكري، وتجنب مسكنات NSAIDs بإفراط، والامتناع عن التدخين"
    },
    "testicular": {
        "name": "سرطان الخصية الجرثومي (Testicular Germ Cell)",
        "marker_name": "AFP / Beta-hCG / LDH", "normal_max": 5.0,
        "early_protocol": "استئصال الخصية الجذري عبر المغبن، يليه المراقبة أو دورة BEP واحدة",
        "adv_protocol": "بروتوكول BEP المعياري (Bleomycin + Etoposide + Cisplatin) - نسبة الشفاء تفوق 95%",
        "prevention": "الفحص الذاتي الشهري للخصية للشباب بين 15 و 35 سنة لكشف أي كتلة صلبة مبكراً"
    },

    # 5. الدم ونخاع العظم
    "aml": {
        "name": "ابيضاض الدم النقوي الحاد (AML)",
        "marker_name": "نسبة الخلايا الأرومية (Blast %)", "normal_max": 5.0,
        "early_protocol": "بروتوكول الحث المكثف 7+3 (Cytarabine لـ 7 أيام + Daunorubicin لـ 3 أيام)",
        "adv_protocol": "زراعة نخاع عظم خيفي (Allogeneic BMT) أو (Venetoclax + Azacitidine)",
        "prevention": "تجنب التعرض للبنزين العضوي والمشتقات النفطية الصناعية والإشعاعات المؤينة"
    },
    "cml": {
        "name": "ابيضاض الدم النقوي المزمن (CML - فيلادلفيا+)",
        "marker_name": "BCR-ABL1 PCR %", "normal_max": 0.1,
        "early_protocol": "مثبطات التيروزين كينيز الفموية: Imatinib (Gleevec) كبسولة يومياً مدى الحياة",
        "adv_protocol": "التحويل للجيل الثاني من المثبطات (Dasatinib أو Nilotinib) أو زراعة النخاع",
        "prevention": "إجراء فحص دم شامل (CBC) دوري سنوي يكتشف ارتفاع كريات الدم البيضاء مبكراً"
    },
    "hodgkin": {
        "name": "ليمفوما هودجكين (Hodgkin Lymphoma)",
        "marker_name": "LDH / PET-CT Scan SUV", "normal_max": 240.0,
        "early_protocol": "دورتان إلى 4 دورات من بروتوكول ABVD متبوعة بإشعاع موضعي للحقول المصابة",
        "adv_protocol": "بروتوكول ABVD كامل (Adriamycin + Bleomycin + Vinblastine + Dacarbazine) أو Brentuximab",
        "prevention": "الفحص الفوري لأي تضخم غير مؤلم في العقد اللمفاوية بالرقبة أو الإبط يدوم أكثر من 3 أسابيع"
    },
    "myeloma": {
        "name": "الورم النقوي المتعدد (Multiple Myeloma)",
        "marker_name": "M-Protein / Free Light Chains", "normal_max": 0.0,
        "early_protocol": "بروتوكول VRd (Bortezomib + Lenalidomide + Dexamethasone)",
        "adv_protocol": "بروتوكول VRd مدمجاً مع Daratumumab يليه زرع نخاع عظم ذاتي (Autologous)",
        "prevention": "متابعة اعتلال الغاماقلوبولينات أحادي النسيلة (MGUS) سنوياً لمنع تحوله لنقوي نشط"
    },

    # 6. الجلد والجهاز العصبي والعظام
    "melanoma": {
        "name": "الميلانوما الخبيثة بالجلد (Malignant Melanoma)",
        "marker_name": "LDH / عمق بريسلو (Breslow)", "normal_max": 220.0,
        "early_protocol": "استئصال جراحي واسع بهامش أمان 1-2 سم وخزعة العقدة اللمفاوية الخافرة",
        "adv_protocol": "المناعي المزدوج (Nivolumab + Ipilimumab) أو الموجه (Dabrafenib + Trametinib لـ BRAF+)",
        "prevention": "قاعدة ABCDE لمراقبة الشامات، وواقي الشمس SPF 50+ وتجنب أجهزة التسمير (Solarium)"
    },
    "glioblastoma": {
        "name": "الورم الأرومي الدبقي الدماغي (Glioblastoma)",
        "marker_name": "رنين مغناطيسي MRI بروتوكول الأورام", "normal_max": 0.0,
        "early_protocol": "استئصال ورمي جراحي أقصى آمن بالاستعانة بالمجهر الجراحي",
        "adv_protocol": "بروتوكول Stupp (إشعاع دماغي خارجي متزامن يومياً مع Temozolomide ثم جرعات صيانة)",
        "prevention": "تجنب الإشعاعات المؤينة للرأس، والمتابعة العصبية الفورية عند نوبات الصداع المستمر"
    },
    "osteosarcoma": {
        "name": "الساركوما العظمية الخبيثة (Osteosarcoma)",
        "marker_name": "الفوسفاتاز القلوية (ALP) / LDH", "normal_max": 120.0,
        "early_protocol": "كيماوي حثي استباقي يليه استئصال جراحي مع إنقاذ الطرف ثم كيماوي مكمل",
        "adv_protocol": "بروتوكول MAP (جرعات فائقة من Methotrexate مع Leucovorin + Doxorubicin + Cisplatin)",
        "prevention": "الفحص الشعاعي العاجل لأي ألم عظمي وتورم مستمر لدى اليافعين حول منطقة الركبة"
    }
}

# ==============================================================================
# 3. إعداد خادم FastAPI ونقاط النهاية السريرية
# ==============================================================================
artifact = joblib.load(MODEL_FILE)
burden_model = artifact['model']
expected_features = artifact['features']

app = FastAPI(title="OncoAI Clinical Supercomputer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def serve_web_page():
    return FileResponse("index.html")

class PatientEvaluationRequest(BaseModel):
    cancer_key: str
    t_stage: int            # 1 إلى 4
    n_stage: int            # 0 إلى 3
    m_stage: int            # 0 أو 1
    marker_value: float
    performance_score: int  # 0 إلى 3 (مقياس ECOG)
    age: int
    # علامات رصد الطوارئ السريرية الفورية
    has_fever_neutropenia: int # حمى مع تدني المناعة
    has_cord_compression: int  # ألم ظهر حاد مع خدر أو ضعف أطراف
    has_svc_symptoms: int      # انتفاخ بالوجه واحتقان أوردة الرقبة
    has_hypercalcemia: int     # تخليط ذهني وإعياء وعطش شديد

@app.get("/get_oncology_metadata")
def get_metadata():
    """تزويد الواجهة ببيانات جميع أنواع الأورام والواسمات الطبيعية"""
    meta = {}
    for k, v in ONCOLOGY_DATABASE.items():
        meta[k] = {
            "name": v["name"],
            "marker_name": v["marker_name"],
            "normal_max": v["normal_max"]
        }
    return meta

@app.post("/evaluate_oncology")
def evaluate_oncology(p: PatientEvaluationRequest):
    if p.cancer_key not in ONCOLOGY_DATABASE:
        return {"status": "error", "message": "النوع السرطاني غير مسجل"}

    c_info = ONCOLOGY_DATABASE[p.cancer_key]
    marker_ratio = p.marker_value / max(0.1, c_info["normal_max"])

    # 1. تشغيل نموذج الذكاء الاصطناعي لحساب العبء الورمي المئوي بالجسم
    input_df = pd.DataFrame([{
        't_stage': p.t_stage,
        'n_stage': p.n_stage,
        'm_stage': p.m_stage,
        'marker_ratio': marker_ratio,
        'performance_score': p.performance_score,
        'age': p.age
    }])[expected_features]

    predicted_burden = float(burden_model.predict(input_df)[0])
    predicted_burden = min(99.0, max(5.0, round(predicted_burden, 1)))

    # 2. تحديد المرحلة السريرية المعيارية (Clinical TNM Stage)
    if p.m_stage == 1:
        stage_label = "المرحلة الرابعة (Stage IV - نقائل بعيدة بالجسم)"
        is_advanced = True
    elif p.n_stage >= 1:
        stage_label = "المرحلة الثالثة (Stage III - انتشار للعقد اللمفاوية)"
        is_advanced = True
    elif p.t_stage >= 3:
        stage_label = "المرحلة الثانية (Stage II - ورم موضعي متقدم)"
        is_advanced = False
    else:
        stage_label = "المرحلة الأولى (Stage I - ورم مبكر محدد)"
        is_advanced = False

    # 3. محرك فحص الطوارئ السرطانية العاجلة (Oncologic Emergency Detector)
    emergency_triggered = False
    emergency_title = None
    emergency_solution = None

    if p.has_cord_compression == 1:
        emergency_triggered = True
        emergency_title = "🚨 طارئ حرج: اشتباه انضغاط الحبل الشوكي الورمي (Malignant Spinal Cord Compression)!"
        emergency_solution = "حقن وريدي فوري لجرعة عالية من الكورتيزون (Dexamethasone 16-24mg)، والتنسيق الإسعافي لإجراء رنين مغناطيسي كامل للعمود الفقري وإشعاع طارئ أو جراحة تحرير الضغط خلال 24 ساعة لمنع الشلل السفلي الدائم."
    elif p.has_fever_neutropenia == 1:
        emergency_triggered = True
        emergency_title = "🚨 طارئ إنتاني: قلة العدلات الحموية (Febrile Neutropenia)!"
        emergency_solution = "بدء مضادات حيوية وريدية واسعة الطيف مضادة للزوائف (Meropenem 1g أو Piperacillin-Tazobactam 4.5g) خلال أقل من 60 دقيقة من الوصول (Golden Hour)، مع سحب مزرعة دم فورية وحقن محفزات النخاع Filgrastim (G-CSF)."
    elif p.has_svc_symptoms == 1:
        emergency_triggered = True
        emergency_title = "🚨 طارئ وعائي: متلازمة انسداد الوريد الأجوف العلوي (SVC Syndrome)!"
        emergency_solution = "رفع رأس السرير بزاوية 45 درجة، وإعطاء أكسجين وديكساميثازون ومدرات البول، وإجراء تدخل إسعافي بالقسطرة الوعائية لتركيب دعامة وريدية (SVC Stent) أو إشعاع تلطيفي عاجل."
    elif p.has_hypercalcemia == 1:
        emergency_triggered = True
        emergency_title = "🚨 طارئ استقلابي: فرط كالسيوم الدم الخبيث (Malignant Hypercalcemia)!"
        emergency_solution = "إماهة وريدية مكثفة بمحلول ملحي طبيعي (Normal Saline 3-4 لتر/يوم)، مع حقن حمض الزوليدرونيك وريدياً (Zoledronic Acid 4mg) أو حقن Denosumab تحت الجلد."

    # تحديد البروتوكول المناسب للمرحلة
    selected_protocol = c_info["adv_protocol"] if is_advanced else c_info["early_protocol"]

    return {
        "status": "success",
        "cancer_name": c_info["name"],
        "stage": stage_label,
        "is_advanced": is_advanced,
        "tumor_burden_percent": predicted_burden,
        "marker_analysis": f"مستوى الواسم الورمي: {p.marker_value} (الحد المرجعي الطبيعي: {c_info['normal_max']}) - يعادل {round(marker_ratio, 1)} أضعاف المعدل.",
        "treatment_protocol": selected_protocol,
        "prevention_strategy": c_info["prevention"],
        "emergency_triggered": emergency_triggered,
        "emergency_title": emergency_title,
        "emergency_solution": emergency_solution
    }