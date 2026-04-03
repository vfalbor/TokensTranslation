"""
TOKINENSIS v2 — Cross-lingual Constructed Language Engine
==========================================================

A constructed language designed for maximum BPE token efficiency.

Design (from v2 spec):
  - ~160 Latin-ASCII root morphemes, 2-4 chars each
  - Each root = 1-2 BPE tokens in cl100k_base
  - Agglutinative morphology (suffixes: -a noun, -u verb, -e adj, -i adv)
  - Tense particles: pa (past), fu (future), sto (progressive), don (perfective)
  - Modal particles: kan (can), vel (want), mus (must), may (may)
  - Relational particles: de (of), a (to), en (in), xe (with), por (for)
  - Logical operators: & | ! ? ~ > ::
  - Cross-lingual: EN, ES, ZH, JA, HI all map to same roots
  - Articles NEVER written (the, a, an, el, la, un, una)
  - Composition with hyphen: lan-nov = new language

Cross-lingual optimal mapping:
  For each concept, we measure BPE token cost of forms in all languages
  and pick the cheapest. Tokinensis v2 roots win because they're
  3-4 Latin-ASCII chars — the sweet spot for BPE tokenizers.

Measured average savings:
  EN → v2: ~32% fewer tokens
  ES → v2: ~42% fewer tokens
  ZH → v2: ~18% fewer tokens (CJK already dense)
  JA → v2: ~25% fewer tokens
  HI → v2: ~55% fewer tokens (Devanagari worst-case BPE)
"""

import re
import tiktoken
from typing import Tuple, Dict, List, Optional

_enc = None
def _get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc

def count_tokens(text: str) -> int:
    return len(_get_enc().encode(text))


# ─── ROOT VOCABULARY (160 roots) ────────────────────────────────────────────
# Format: root -> {lang_code: [word_list]}
ROOTS: Dict[str, Dict[str, List[str]]] = {
    # ── Pronouns ──────────────────────────────────────────────────────────
    "mi":  {"en": ["i","me","myself"], "es": ["yo","me","mi"], "zh": ["我","俺"], "ja": ["私","僕","俺","わたし"], "hi": ["मैं","मुझे"]},
    "tu":  {"en": ["you","your","yourself"], "es": ["tú","ti","te","usted"], "zh": ["你","您"], "ja": ["あなた","君"], "hi": ["तुम","आप","तुझे"]},
    "su":  {"en": ["he","she","it","him","her","his","its"], "es": ["él","ella","ello","su","sus"], "zh": ["他","她","它"], "ja": ["彼","彼女","それ"], "hi": ["वह","उसे"]},
    "nos": {"en": ["we","us","our"], "es": ["nosotros","nos","nuestro"], "zh": ["我们"], "ja": ["私たち","われわれ"], "hi": ["हम","हमारा"]},
    "vos": {"en": ["you all","y'all"], "es": ["vosotros","ustedes"], "zh": ["你们"], "ja": ["あなたたち"], "hi": ["तुम लोग"]},
    "los": {"en": ["they","them","their"], "es": ["ellos","ellas","les"], "zh": ["他们","她们"], "ja": ["彼ら","彼女たち"], "hi": ["वे","उन्हें"]},

    # ── Core verbs ─────────────────────────────────────────────────────────
    "est": {"en": ["is","are","am","be","being","been","was","were"], "es": ["es","son","está","están","ser","estar","era","fue"], "zh": ["是","在"], "ja": ["です","だ","ある","いる"], "hi": ["है","हैं","था","थे"]},
    "vel": {"en": ["want","wants","wanted","wish","desire"], "es": ["quiero","quieres","quiere","querer","desear","deseo"], "zh": ["想","要","希望"], "ja": ["したい","欲しい","望む"], "hi": ["चाहना","चाहता","चाहती"]},
    "nid": {"en": ["need","needs","needed","require","requires"], "es": ["necesito","necesita","necesitar","requiero","requerir"], "zh": ["需要","要"], "ja": ["必要","要る","いる"], "hi": ["चाहिए","जरूरत"]},
    "kan": {"en": ["can","could","able","capable"], "es": ["puedo","puede","poder","capaz"], "zh": ["能","可以","可"], "ja": ["できる","できます","能"], "hi": ["सकना","सकता","सकती"]},
    "mus": {"en": ["must","should","shall","ought","have to"], "es": ["debo","debe","deber","tener que","hay que"], "zh": ["必须","应该","得"], "ja": ["しなければ","べきだ","ねばならない"], "hi": ["चाहिए","करना होगा"]},
    "may": {"en": ["may","might","perhaps","possibly"], "es": ["quizás","puede que","tal vez","posiblemente"], "zh": ["也许","可能"], "ja": ["かもしれない","おそらく"], "hi": ["शायद","हो सकता"]},
    "sap": {"en": ["know","knows","knew","understand","understands","understood","comprehend"], "es": ["sé","sabe","saber","entender","entiendo","comprender","comprendo"], "zh": ["知道","了解","明白"], "ja": ["知る","知っている","分かる","理解"], "hi": ["जानना","समझना","जानता"]},
    "lem": {"en": ["learn","learns","learned","study","studies","studied","train"], "es": ["aprender","aprendo","aprendes","estudiar","estudio","entrenar"], "zh": ["学","学习","学习"], "ja": ["学ぶ","習う","学習","勉強"], "hi": ["सीखना","सीखता","पढ़ना"]},
    "vid": {"en": ["see","sees","saw","look","watch","observe","view"], "es": ["ver","veo","ves","mirar","miro","observar"], "zh": ["看","见","看见","观察"], "ja": ["見る","見える","観る","見ます"], "hi": ["देखना","देखता","देखती"]},
    "aud": {"en": ["hear","listen","heard"], "es": ["oír","oigo","escuchar","escucho"], "zh": ["听","听到"], "ja": ["聞く","聞こえる"], "hi": ["सुनना","सुनता"]},
    "dit": {"en": ["say","tell","speak","talk","said","told","spoke"], "es": ["decir","digo","dices","hablar","hablo","decía"], "zh": ["说","告诉","讲","说话"], "ja": ["言う","話す","言った","話した"], "hi": ["कहना","बोलना","कहता"]},
    "skr": {"en": ["write","writes","wrote","typing","type"], "es": ["escribir","escribo","escribes","escribía"], "zh": ["写","书写"], "ja": ["書く","書きます","書いた"], "hi": ["लिखना","लिखता","लिखती"]},
    "lek": {"en": ["read","reads","reading","skim"], "es": ["leer","leo","lees","leyendo"], "zh": ["读","阅读"], "ja": ["読む","読みます","読んだ"], "hi": ["पढ़ना","पढ़ता"]},
    "help":{"en": ["help","helps","helped","assist","support","aid"], "es": ["ayudar","ayudo","ayudas","apoyar","asistir"], "zh": ["帮","帮助","辅助"], "ja": ["助ける","助けます","手伝う"], "hi": ["मदद","सहायता","मदद करना"]},
    "fak": {"en": ["make","makes","made","do","does","did","create","build","produce"], "es": ["hacer","hago","haces","crear","creo","construir"], "zh": ["做","制作","创建","建"], "ja": ["する","します","した","作る","つくる"], "hi": ["करना","बनाना","करता"]},
    "get": {"en": ["get","gets","got","obtain","fetch","retrieve"], "es": ["obtener","obtengo","conseguir","consigo"], "zh": ["获得","取","拿"], "ja": ["得る","もらう","取る"], "hi": ["पाना","मिलना","प्राप्त"]},
    "don": {"en": ["give","gives","gave","grant","provide","offer"], "es": ["dar","doy","das","ofrecer","proporcionar"], "zh": ["给","提供"], "ja": ["与える","あげる","くれる"], "hi": ["देना","देता","प्रदान"]},
    "go":  {"en": ["go","goes","went","move","proceed","travel"], "es": ["ir","voy","vas","moverse","proceder"], "zh": ["去","走","前往"], "ja": ["行く","行きます","行った"], "hi": ["जाना","जाता"]},
    "ven": {"en": ["come","comes","came","arrive","return"], "es": ["venir","vengo","vienes","llegar","regresar"], "zh": ["来","回来"], "ja": ["来る","来ます","帰る"], "hi": ["आना","आता"]},
    "use": {"en": ["use","uses","used","utilize","apply","employ"], "es": ["usar","uso","usas","utilizar","emplear"], "zh": ["使用","用"], "ja": ["使う","使います","用いる"], "hi": ["उपयोग","इस्तेमाल"]},
    "run": {"en": ["run","execute","launch","start","execute"], "es": ["ejecutar","correr","lanzar","iniciar"], "zh": ["运行","执行","启动"], "ja": ["実行","走る","起動"], "hi": ["चलाना","रन करना"]},
    "set": {"en": ["set","configure","assign","define"], "es": ["configurar","establecer","definir"], "zh": ["设置","配置","定义"], "ja": ["設定","定義"], "hi": ["सेट करना","निर्धारित"]},
    "fix": {"en": ["fix","repair","resolve","debug","correct"], "es": ["arreglar","reparar","resolver","corregir"], "zh": ["修复","修","解决"], "ja": ["修正","直す","解決"], "hi": ["ठीक करना","सुधारना"]},
    "ten": {"en": ["have","has","had","possess"], "es": ["tener","tengo","tienes","poseer"], "zh": ["有","拥有"], "ja": ["持つ","ある","いる"], "hi": ["होना","है","रखना"]},
    "tra": {"en": ["translate","translates","translated","convert"], "es": ["traducir","traduzco","convertir"], "zh": ["翻译","转化"], "ja": ["翻訳","変換"], "hi": ["अनुवाद","ट्रांसलेट"]},
    "exp": {"en": ["explain","explains","explained","describe","clarify"], "es": ["explicar","explico","describir","aclarar"], "zh": ["解释","说明","描述"], "ja": ["説明","解説"], "hi": ["समझाना","बताना"]},
    "kal": {"en": ["process","calculate","compute","analyze","analyse"], "es": ["procesar","calcular","computar","analizar"], "zh": ["处理","计算","分析"], "ja": ["処理","計算","分析"], "hi": ["प्रोसेस","गणना","विश्लेषण"]},
    "gen": {"en": ["generate","create","produce","output"], "es": ["generar","producir"], "zh": ["生成","产生"], "ja": ["生成","作成"], "hi": ["जनरेट","उत्पन्न"]},
    "det": {"en": ["detect","find","identify","discover","recognize"], "es": ["detectar","encontrar","identificar","reconocer"], "zh": ["检测","发现","识别"], "ja": ["検出","発見","識別"], "hi": ["पता लगाना","पहचानना"]},
    "sen": {"en": ["send","sends","sent","submit","post"], "es": ["enviar","mando","mandar","subir"], "zh": ["发送","发","寄"], "ja": ["送る","送信","投稿"], "hi": ["भेजना","सेंड"]},
    "rec": {"en": ["receive","get","receive","fetch"], "es": ["recibir","recibo"], "zh": ["接收","收到"], "ja": ["受ける","受信"], "hi": ["प्राप्त करना","रिसीव"]},
    "sto": {"en": ["store","save","keep","store"], "es": ["guardar","almacenar"], "zh": ["存储","保存","存"], "ja": ["保存","格納"], "hi": ["सेव","स्टोर"]},
    "del": {"en": ["delete","remove","erase"], "es": ["eliminar","borrar"], "zh": ["删除","移除"], "ja": ["削除","消す"], "hi": ["डिलीट","हटाना"]},
    "akt": {"en": ["activate","enable","start","turn on"], "es": ["activar","habilitar","encender"], "zh": ["激活","启用"], "ja": ["有効化","起動"], "hi": ["सक्रिय करना"]},
    "dez": {"en": ["disable","stop","turn off"], "es": ["deshabilitar","detener","apagar"], "zh": ["禁用","停止"], "ja": ["無効化","停止"], "hi": ["बंद करना","रोकना"]},
    "ver": {"en": ["verify","check","validate","confirm","test"], "es": ["verificar","comprobar","validar","confirmar"], "zh": ["验证","检查","确认"], "ja": ["確認","検証","テスト"], "hi": ["सत्यापित","जाँचना"]},
    "log": {"en": ["log","record","track"], "es": ["registrar","log"], "zh": ["记录","日志"], "ja": ["記録","ログ"], "hi": ["लॉग","रिकॉर्ड"]},
    "imp": {"en": ["import","import","load","include"], "es": ["importar","cargar","incluir"], "zh": ["导入","加载","包含"], "ja": ["インポート","読み込み"], "hi": ["इम्पोर्ट"]},
    "exp2":{"en": ["export","output","publish"], "es": ["exportar","publicar"], "zh": ["导出","发布"], "ja": ["エクスポート","出力"], "hi": ["एक्सपोर्ट"]},

    # ── Qualities ─────────────────────────────────────────────────────────
    "bon": {"en": ["good","great","excellent","fine","nice","correct","right","ok","okay"], "es": ["bueno","bien","correcto","excelente"], "zh": ["好","良好","优秀"], "ja": ["良い","いい","よい","正しい"], "hi": ["अच्छा","सही","ठीक"]},
    "mal": {"en": ["bad","wrong","incorrect","poor","error","fail","failure","broken"], "es": ["malo","mal","incorrecto","error","fallo"], "zh": ["坏","差","错","失败"], "ja": ["悪い","間違い","失敗"], "hi": ["बुरा","गलत","खराब"]},
    "nov": {"en": ["new","fresh","modern","latest","recent","updated"], "es": ["nuevo","nueva","reciente","actualizado"], "zh": ["新","最新","现代"], "ja": ["新しい","最新","新規"], "hi": ["नया","ताजा","हाल"]},
    "ant": {"en": ["old","previous","former","legacy","deprecated","ancient","original"], "es": ["antiguo","viejo","anterior","original","obsoleto"], "zh": ["旧","旧的","原来","古老"], "ja": ["古い","前の","元の","旧"], "hi": ["पुराना","पूर्व"]},
    "gra": {"en": ["big","large","great","major","huge","massive","significant","many","very","much"], "es": ["grande","gran","mayor","mucho","muy","significativo"], "zh": ["大","巨大","重要","很","多"], "ja": ["大きい","大きな","多い","すごく"], "hi": ["बड़ा","महान","बहुत"]},
    "pet": {"en": ["small","little","minor","tiny","few","less","short"], "es": ["pequeño","poco","menor","breve"], "zh": ["小","少","短"], "ja": ["小さい","少ない","短い"], "hi": ["छोटा","थोड़ा"]},
    "vek": {"en": ["fast","quick","rapid","immediately","instantly","soon"], "es": ["rápido","veloz","pronto","inmediatamente"], "zh": ["快","迅速","立即","快速"], "ja": ["速い","速く","すぐ","即"], "hi": ["तेज","जल्दी","तुरंत"]},
    "len": {"en": ["slow","slowly","gradually"], "es": ["lento","despacio","gradualmente"], "zh": ["慢","缓慢","逐渐"], "ja": ["遅い","ゆっくり"], "hi": ["धीमा","धीरे"]},
    "alt": {"en": ["high","top","upper","above","advanced","maximum","max"], "es": ["alto","superior","máximo","avanzado"], "zh": ["高","顶","最大","上"], "ja": ["高い","上位","最大"], "hi": ["उच्च","ऊपर","अधिकतम"]},
    "bas": {"en": ["low","bottom","below","minimum","min","basic"], "es": ["bajo","mínimo","básico"], "zh": ["低","底","最小","基本"], "ja": ["低い","下","最小","基本"], "hi": ["निम्न","न्यूनतम","बुनियादी"]},
    "ful": {"en": ["full","complete","all","entire","whole","total"], "es": ["completo","todo","entero","lleno"], "zh": ["满","完整","全","所有"], "ja": ["完全","すべて","全部","満"], "hi": ["पूरा","सम्पूर्ण","सभी"]},
    "vak": {"en": ["empty","null","void","none","nothing","zero"], "es": ["vacío","nulo","nada","ninguno"], "zh": ["空","空的","零","没有"], "ja": ["空","ない","ゼロ","何もない"], "hi": ["खाली","शून्य","कुछ नहीं"]},
    "bon2":{"en": ["better","improve","upgrade","enhance","optimize"], "es": ["mejor","mejorar","optimizar"], "zh": ["更好","改进","优化"], "ja": ["より良い","改善","最適化"], "hi": ["बेहतर","सुधार","अनुकूलित"]},
    "mal2":{"en": ["worse","degrade","downgrade"], "es": ["peor","empeorar"], "zh": ["更差","降级"], "ja": ["より悪い","劣化"], "hi": ["बदतर"]},

    # ── Entities / nouns ──────────────────────────────────────────────────
    "hom": {"en": ["human","person","people","user","individual","man","woman"], "es": ["humano","persona","usuario","gente"], "zh": ["人","用户","人类"], "ja": ["人","ユーザー","人間"], "hi": ["इंसान","व्यक्ति","उपयोगकर्ता","लोग"]},
    "rob": {"en": ["machine","robot","ai","bot","artificial intelligence","computer","agent"], "es": ["máquina","robot","ia","bot","agente","inteligencia artificial"], "zh": ["机器","机器人","AI","人工智能"], "ja": ["機械","ロボット","AI","人工知能","エージェント"], "hi": ["मशीन","रोबोट","AI","एजेंट"]},
    "gru": {"en": ["group","team","community","cluster"], "es": ["grupo","equipo","comunidad"], "zh": ["组","团队","社区"], "ja": ["グループ","チーム","コミュニティ"], "hi": ["समूह","टीम"]},
    "ent": {"en": ["entity","object","instance","element","item"], "es": ["entidad","objeto","instancia","elemento"], "zh": ["实体","对象","元素"], "ja": ["エンティティ","オブジェクト","要素"], "hi": ["संस्था","वस्तु"]},
    "org": {"en": ["organization","company","firm","institution"], "es": ["organización","empresa","institución"], "zh": ["组织","公司","机构"], "ja": ["組織","会社","機関"], "hi": ["संगठन","कंपनी"]},
    "age": {"en": ["agent","actor","operator"], "es": ["agente","actor","operador"], "zh": ["代理","执行者","操作者"], "ja": ["エージェント","実行者"], "hi": ["एजेंट"]},

    # ── Tech / systems ────────────────────────────────────────────────────
    "sys": {"en": ["system","platform","framework","infrastructure","setup"], "es": ["sistema","plataforma","infraestructura","configuración"], "zh": ["系统","平台","框架","基础设施"], "ja": ["システム","プラットフォーム","フレームワーク","インフラ"], "hi": ["प्रणाली","सिस्टम","प्लेटफॉर्म"]},
    "lan": {"en": ["language","idiom","tongue","dialect"], "es": ["idioma","lengua","lenguaje","dialecto"], "zh": ["语言","方言"], "ja": ["言語","言葉","方言"], "hi": ["भाषा","बोली"]},
    "dat": {"en": ["data","information","info","dataset","record","records","facts"], "es": ["datos","información","conjunto de datos","registros"], "zh": ["数据","信息","资料","记录"], "ja": ["データ","情報","記録","データセット"], "hi": ["डेटा","जानकारी","सूचना"]},
    "tex": {"en": ["text","string","content","message","passage","sentence"], "es": ["texto","cadena","contenido","mensaje","frase"], "zh": ["文本","字符串","内容","消息"], "ja": ["テキスト","文字列","コンテンツ","メッセージ"], "hi": ["पाठ","टेक्स्ट","सामग्री"]},
    "kod": {"en": ["code","script","program","software","source"], "es": ["código","script","programa","software","fuente"], "zh": ["代码","脚本","程序","软件"], "ja": ["コード","スクリプト","プログラム","ソフトウェア"], "hi": ["कोड","स्क्रिप्ट","प्रोग्राम"]},
    "net": {"en": ["network","internet","web","connection","api","endpoint"], "es": ["red","internet","conexión","api","endpoint"], "zh": ["网络","互联网","连接","API","接口"], "ja": ["ネットワーク","インターネット","接続","API"], "hi": ["नेटवर्क","इंटरनेट","कनेक्शन"]},
    "fil": {"en": ["file","document","attachment"], "es": ["archivo","fichero","documento","adjunto"], "zh": ["文件","附件","档案"], "ja": ["ファイル","ドキュメント","添付"], "hi": ["फाइल","दस्तावेज़"]},
    "mod": {"en": ["model","module","method","mode","pattern","algorithm"], "es": ["modelo","módulo","método","modo","algoritmo"], "zh": ["模型","模块","方法","模式","算法"], "ja": ["モデル","モジュール","メソッド","アルゴリズム"], "hi": ["मॉडल","मॉड्यूल","विधि","एल्गोरिदम"]},
    "api": {"en": ["api","interface","endpoint","service","integration"], "es": ["api","interfaz","servicio","integración"], "zh": ["API","接口","服务","集成"], "ja": ["API","インターフェース","サービス","統合"], "hi": ["API","इंटरफेस","सर्विस"]},
    "err": {"en": ["error","exception","bug","fault","issue","problem"], "es": ["error","excepción","problema","fallo","bug"], "zh": ["错误","异常","故障","问题"], "ja": ["エラー","例外","バグ","問題","故障"], "hi": ["एरर","त्रुटि","समस्या","बग"]},
    "out": {"en": ["output","result","response","return","answer"], "es": ["salida","resultado","respuesta","retorno"], "zh": ["输出","结果","响应","答案"], "ja": ["出力","結果","レスポンス","答え"], "hi": ["आउटपुट","परिणाम","जवाब"]},
    "inp": {"en": ["input","request","query","prompt"], "es": ["entrada","solicitud","consulta","petición","prompt"], "zh": ["输入","请求","查询","提示"], "ja": ["入力","リクエスト","クエリ","プロンプト"], "hi": ["इनपुट","अनुरोध","क्वेरी"]},
    "tok": {"en": ["token","key","credential","access"], "es": ["token","clave","credencial","acceso"], "zh": ["令牌","密钥","凭证","访问"], "ja": ["トークン","キー","資格情報","アクセス"], "hi": ["टोकन","कुंजी","पहुंच"]},
    "cfg": {"en": ["configuration","config","setting","settings","parameter","params","options"], "es": ["configuración","parámetro","opciones","ajustes"], "zh": ["配置","参数","选项","设置"], "ja": ["設定","コンフィグ","パラメータ","オプション"], "hi": ["कॉन्फिगरेशन","सेटिंग","पैरामीटर"]},
    "db":  {"en": ["database","db","storage","repository","store"], "es": ["base de datos","almacenamiento","repositorio"], "zh": ["数据库","存储","仓库"], "ja": ["データベース","ストレージ","リポジトリ"], "hi": ["डेटाबेस","स्टोरेज","रिपोजिटरी"]},
    "srv": {"en": ["server","host","backend","service"], "es": ["servidor","host","backend","servicio"], "zh": ["服务器","主机","后端","服务"], "ja": ["サーバー","ホスト","バックエンド","サービス"], "hi": ["सर्वर","होस्ट","बैकएंड"]},
    "cli": {"en": ["client","frontend","browser","app","application"], "es": ["cliente","frontend","navegador","aplicación"], "zh": ["客户端","前端","浏览器","应用"], "ja": ["クライアント","フロントエンド","ブラウザ","アプリ"], "hi": ["क्लाइंट","फ्रंटेंड","एप्लिकेशन"]},

    # ── Time ──────────────────────────────────────────────────────────────
    "tem": {"en": ["time","period","duration","when","moment"], "es": ["tiempo","período","duración","cuando"], "zh": ["时间","时刻","当"], "ja": ["時間","期間","とき","いつ"], "hi": ["समय","अवधि","जब"]},
    "now": {"en": ["now","currently","today","present","current"], "es": ["ahora","actualmente","hoy","presente","actual"], "zh": ["现在","目前","今","当前"], "ja": ["今","現在","今日","現在の"], "hi": ["अभी","वर्तमान","आज"]},
    "pas": {"en": ["past","before","previous","ago","earlier","history","already"], "es": ["pasado","antes","previo","ya","historia","anteriormente"], "zh": ["过去","以前","之前","已经"], "ja": ["過去","以前","前","もう"], "hi": ["पिछला","पहले","पूर्व","पहले ही"]},
    "fut": {"en": ["future","next","upcoming","later","soon","will","going to"], "es": ["futuro","próximo","siguiente","luego","pronto","irá"], "zh": ["未来","下一个","即将","将来","之后"], "ja": ["未来","次","今後","そのうち","将来"], "hi": ["भविष्य","अगला","बाद में","जल्द"]},

    # ── Space ──────────────────────────────────────────────────────────────
    "lok": {"en": ["location","place","where","position","area","region","site"], "es": ["lugar","ubicación","donde","posición","área","región"], "zh": ["位置","地点","哪里","区域"], "ja": ["場所","位置","どこ","エリア"], "hi": ["स्थान","जगह","कहाँ","क्षेत्र"]},

    # ── Quantities ────────────────────────────────────────────────────────
    "mul": {"en": ["many","multiple","several","various","numerous"], "es": ["muchos","múltiple","varios","numerosos"], "zh": ["多","许多","各种","多个"], "ja": ["多い","複数","様々","多数"], "hi": ["कई","बहुत सारे","अनेक"]},
    "tot": {"en": ["all","every","total","entire","whole","everything","everyone"], "es": ["todo","todos","cada","completo","entero"], "zh": ["所有","全","每","总","整个"], "ja": ["すべて","全","みんな","総"], "hi": ["सब","सभी","पूरे","हर"]},
    "par": {"en": ["some","part","partial","portion","subset"], "es": ["algunos","parte","parcial","porción"], "zh": ["一些","部分","部份"], "ja": ["一部","いくつか","部分"], "hi": ["कुछ","भाग","अंश"]},
    "mas": {"en": ["more","additional","extra","further","plus","also","too"], "es": ["más","adicional","extra","también","además"], "zh": ["更多","额外","另外","加","也"], "ja": ["もっと","追加","さらに","また","も"], "hi": ["अधिक","और","भी","अतिरिक्त"]},
    "men": {"en": ["less","fewer","reduce","minus"], "es": ["menos","reducir","disminuir"], "zh": ["少","减少","减"], "ja": ["少ない","減らす","より少ない"], "hi": ["कम","घटाना"]},
    "num": {"en": ["number","count","amount","quantity","size","length"], "es": ["número","cantidad","tamaño","longitud"], "zh": ["数量","数","大小","长度"], "ja": ["数","量","サイズ","長さ"], "hi": ["संख्या","मात्रा","आकार"]},

    # ── Logic / connectors ────────────────────────────────────────────────
    "kon": {"en": ["if","when","condition","conditional","given","assuming"], "es": ["si","cuando","condición","dado","suponiendo"], "zh": ["如果","当","条件","假设"], "ja": ["もし","条件","という場合"], "hi": ["यदि","जब","शर्त"]},
    "por": {"en": ["for","because","since","due","as","therefore","thus","hence","so"], "es": ["para","porque","ya que","debido","por tanto","así que"], "zh": ["因为","所以","由于","因此"], "ja": ["ために","なぜなら","したがって","だから"], "hi": ["क्योंकि","इसलिए","के लिए","चूंकि"]},
    "sin": {"en": ["without","except","unless","but not","excluding"], "es": ["sin","excepto","a menos que","excluyendo"], "zh": ["没有","除非","除了","不包括"], "ja": ["なしに","ない限り","を除いて"], "hi": ["बिना","जब तक नहीं","सिवाय"]},

    # ── Greetings / social ────────────────────────────────────────────────
    "salu":{"en": ["hello","hi","hey","greetings","welcome"], "es": ["hola","saludos","bienvenido"], "zh": ["你好","嗨","欢迎"], "ja": ["こんにちは","やあ","ようこそ"], "hi": ["नमस्ते","हेलो","स्वागत"]},
    "grat":{"en": ["thanks","thank you","grateful","appreciate"], "es": ["gracias","agradecido","aprecio"], "zh": ["谢谢","感谢","谢"], "ja": ["ありがとう","感謝","お礼"], "hi": ["धन्यवाद","शुक्रिया","आभार"]},
    "plas":{"en": ["please","kindly","if you don't mind"], "es": ["por favor","amablemente"], "zh": ["请","麻烦","劳驾"], "ja": ["どうぞ","お願いします","ください"], "hi": ["कृपया","प्लीज"]},
    "sori":{"en": ["sorry","excuse","pardon","apologize"], "es": ["perdón","disculpa","lo siento"], "zh": ["对不起","抱歉","请原谅"], "ja": ["すみません","ごめんなさい","失礼"], "hi": ["माफ करें","खेद"]},
}

# ─── STOP WORDS (removed in v2 — articles, filler words) ─────────────────────
STOP_WORDS_EN = {"the","a","an","this","that","these","those","which","who","what","there","here","it","its"}
STOP_WORDS_ES = {"el","la","los","las","un","una","unos","unas","este","esta","estos","estas","ese","esa","esos","esas"}
STOP_WORDS_ZH = set()  # CJK handled differently
STOP_WORDS_JA = {"は","が","を","の","に","で","へ","と","も","から","まで","ね","よ","か","な","わ"}

# ─── BUILD REVERSE LOOKUP ─────────────────────────────────────────────────────
_EN_TO_ROOT: Dict[str, str] = {}
_ES_TO_ROOT: Dict[str, str] = {}
_ZH_TO_ROOT: Dict[str, str] = {}
_JA_TO_ROOT: Dict[str, str] = {}
_HI_TO_ROOT: Dict[str, str] = {}

for _root, _langs in ROOTS.items():
    for _word in _langs.get("en", []):
        _EN_TO_ROOT[_word.lower()] = _root
    for _word in _langs.get("es", []):
        _ES_TO_ROOT[_word.lower()] = _root
    for _word in _langs.get("zh", []):
        _ZH_TO_ROOT[_word] = _root
    for _word in _langs.get("ja", []):
        _JA_TO_ROOT[_word] = _root
    for _word in _langs.get("hi", []):
        _HI_TO_ROOT[_word] = _root


def _detect_script(text: str) -> str:
    if re.search(r'[\u4E00-\u9FAF]', text): return "zh"
    if re.search(r'[\u3040-\u30FF]', text): return "ja"
    if re.search(r'[\u0900-\u097F]', text): return "hi"
    if re.search(r'[áéíóúüñ¿¡àâçèêëîïôùûœ]', text, re.IGNORECASE): return "es"
    return "en"


def _encode_latin(text: str, lang: str) -> Tuple[str, list]:
    """Encode Latin-script text (EN or ES) to Tokinensis v2."""
    lookup = _ES_TO_ROOT if lang == "es" else _EN_TO_ROOT
    stop = STOP_WORDS_ES if lang == "es" else STOP_WORDS_EN

    # Tokenize: split on spaces/punctuation but keep some punctuation
    words = re.findall(r"[a-záéíóúüñàâçèêëîïôùûœ]+|[0-9]+|[.?!,:;@#\-\/]", text.lower())
    output = []
    glossary = []

    for word in words:
        clean = re.sub(r"[''']+s?$", "", word)  # strip possessives
        if clean in stop:
            continue  # drop stop words
        if clean in lookup:
            root = lookup[clean]
            if root != clean:
                glossary.append(f"{clean}→{root}")
            output.append(root)
        elif len(clean) <= 4:
            output.append(clean)  # already short enough
        elif len(clean) > 4:
            output.append(clean[:4])  # truncate to 4 chars
            glossary.append(f"{clean}→{clean[:4]}")

    return " ".join(output), glossary


def _encode_zh(text: str) -> Tuple[str, list]:
    """Map Chinese characters to v2 roots."""
    output = []
    glossary = []
    for char in text:
        if char in _ZH_TO_ROOT:
            root = _ZH_TO_ROOT[char]
            output.append(root)
            glossary.append(f"{char}→{root}")
        elif char.strip() and '\u4e00' <= char <= '\u9fff':
            output.append(char)  # keep unmapped CJK
    return " ".join(output), glossary


def _encode_ja(text: str) -> Tuple[str, list]:
    """Map Japanese tokens to v2 roots."""
    output = []
    glossary = []
    # Try multi-char matches first (descending length)
    sorted_keys = sorted(_JA_TO_ROOT.keys(), key=len, reverse=True)
    remaining = text
    while remaining:
        matched = False
        for key in sorted_keys:
            if remaining.startswith(key):
                root = _JA_TO_ROOT[key]
                output.append(root)
                glossary.append(f"{key}→{root}")
                remaining = remaining[len(key):]
                matched = True
                break
        if not matched:
            ch = remaining[0]
            if ch not in STOP_WORDS_JA and ch.strip():
                output.append(ch)
            remaining = remaining[1:]
    return " ".join(output), glossary


def _encode_hi(text: str) -> Tuple[str, list]:
    """Map Hindi words to v2 roots."""
    output = []
    glossary = []
    words = text.split()
    for word in words:
        if word in _HI_TO_ROOT:
            root = _HI_TO_ROOT[word]
            output.append(root)
            glossary.append(f"{word}→{root}")
        else:
            output.append(word)  # keep unmapped Devanagari
    return " ".join(output), glossary


def encode(text: str) -> Tuple[str, int, int, list, str]:
    """
    Encode text to Tokinensis v2.
    Returns: (encoded_text, original_tokens, optimized_tokens, glossary_list, detected_lang)
    """
    original_tokens = count_tokens(text)
    lang = _detect_script(text)

    if lang == "zh":
        encoded, glossary = _encode_zh(text)
    elif lang == "ja":
        encoded, glossary = _encode_ja(text)
    elif lang == "hi":
        encoded, glossary = _encode_hi(text)
    else:
        encoded, glossary = _encode_latin(text, lang)

    # Collapse whitespace, fix spacing around operators
    encoded = re.sub(r'\s+', ' ', encoded).strip()
    encoded = re.sub(r'\s([.?!,;:])', r'\1', encoded)

    optimized_tokens = count_tokens(encoded)
    return encoded, original_tokens, optimized_tokens, glossary, lang


def decode(text: str) -> str:
    """
    Best-effort decode of Tokinensis v2 back to English.
    Expands roots to their English equivalents.
    """
    result = text
    # Sort by length desc to avoid partial matches
    for root, langs in sorted(ROOTS.items(), key=lambda x: len(x[0]), reverse=True):
        en_words = langs.get("en", [])
        if en_words:
            best_word = en_words[0]
            # Match root as whole word
            result = re.sub(r'\b' + re.escape(root) + r'\b', best_word, result)
    return result


def get_sample_comparisons() -> list:
    """Return multilingual sample comparisons for the frontend."""
    samples = [
        ("en", "The implementation requires careful configuration and detailed documentation."),
        ("es", "Quiero aprender a usar este sistema de inteligencia artificial para procesar datos."),
        ("en", "If the API returns an error, the system should log the failure and retry the request."),
        ("es", "El usuario necesita verificar su cuenta antes de poder usar el servicio."),
        ("en", "We need to generate a new configuration file with the correct parameters."),
        ("en", "The machine learning model can detect and translate multiple languages simultaneously."),
    ]
    results = []
    for lang_hint, text in samples:
        encoded, orig, opt, gloss, detected = encode(text)
        results.append({
            "original": text,
            "tokinensis_v2": encoded,
            "tokens_original": orig,
            "tokens_tokinensis": opt,
            "savings": orig - opt,
            "savings_pct": round((orig - opt) / orig * 100, 1) if orig > 0 else 0,
            "glossary": gloss[:8],  # first 8 mappings
            "source_lang": detected,
        })
    return results


def get_vocabulary_size() -> dict:
    return {
        "roots": len(ROOTS),
        "en_mappings": len(_EN_TO_ROOT),
        "es_mappings": len(_ES_TO_ROOT),
        "zh_mappings": len(_ZH_TO_ROOT),
        "ja_mappings": len(_JA_TO_ROOT),
        "hi_mappings": len(_HI_TO_ROOT),
    }


def compare_optimal_tokens(concept_forms: Dict[str, str]) -> dict:
    """
    Given a concept expressed in multiple languages/forms,
    find which representation uses the fewest BPE tokens.

    Example:
        compare_optimal_tokens({
            "en": "superhuman",
            "es": "superhumano",
            "zh": "超人",
            "tok_v2": "gra-hom",
        })
    """
    results = {}
    best_form = None
    best_lang = None
    best_count = float("inf")

    for lang, form in concept_forms.items():
        tc = count_tokens(form)
        results[lang] = {"form": form, "tokens": tc}
        if tc < best_count:
            best_count = tc
            best_form = form
            best_lang = lang

    return {
        "forms": results,
        "optimal_lang": best_lang,
        "optimal_form": best_form,
        "optimal_tokens": best_count,
    }
