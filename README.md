🧠 Giriş: Derin Öğrenme Mimarilerine Kapsamlı Bir Bakış

Bu eğitim serisi, derin öğrenmenin matematiksel temellerinden başlayarak, modern ve karmaşık sinir ağı mimarilerine kadar uzanan geniş bir yelpazeyi ele almaktadır. Her bölümde, teorik bilginin pratik kodlamaya nasıl dönüştürüldüğü, endüstri standartlarındaki en iyi pratikler (best practices) ile gösterilmiştir. Amaç, temel sınıflandırma problemlerinden görüntü işleme ve zaman serisi tahminlemesine kadar farklı alanlarda sağlam ve uygulanabilir bir yetkinlik kazanmaktır.

🔧 Temel Sinir Ağları: Sıfırdan İnşa (Foundational NNs: Building from Scratch)

Derin öğrenmenin temel dinamiklerini anlamak amacıyla, NumPy kullanılarak sıfırdan basit bir yapay sinir ağı modeli inşa edilmiştir. Bu bölüm, kütüphanelerin soyutladığı katmanların arkasındaki mekaniği anlamak için kritik bir öneme sahiptir.

Teknik Detaylar:

İleri Yayılım (Forward Propagation) ve Geri Yayılım (Backward Propagation) algoritmalarının manuel implementasyonu.

Aktivasyon fonksiyonu olarak Sigmoid kullanımı.

Logaritmik Kayıp (Log Loss) fonksiyonu ile model hatasının hesaplanması.

Kritik Çıkarım:

Farklı öğrenme oranlarının (learning rate) modelin yakınsama hızı ve performansı üzerindeki etkisi analiz edilmiştir.

📊 Sınıflandırma Problemleri ve Model Optimizasyonu

Gerçek dünya veri setleri (Pima Diyabet, Meme Kanseri) üzerinde ikili sınıflandırma (binary classification) modelleri geliştirilmiş ve bu modellerin performansını iyileştirmeye yönelik iki temel optimizasyon stratejisi incelenmiştir.

1. Manuel Regülarizasyon ile Aşırı Öğrenme Kontrolü (Overfitting Control)

Yüksek kapasiteli modellerin ezberlemesini önlemek amacıyla, bilgiye dayalı manuel optimizasyon teknikleri uygulanmıştır.

Uygulanan Teknikler:

L2 Regularization: Ağırlıkların büyümesini cezalandırarak model karmaşıklığını sınırlar.

Batch Normalization: Katmanlar arasındaki aktivasyonları normalize ederek eğitimi stabilize eder ve hızlandırır.

Dropout: Eğitim sırasında nöronların bir kısmını rastgele kapatarak modelin genelleme yeteneğini artırır.

Early Stopping: Validasyon kaybı artmaya başladığında eğitimi durdurarak en iyi modelin korunmasını sağlar.

2. Keras Tuner ile Otomatik Hiperparametre Optimizasyonu

En iyi model mimarisini bulma sürecini otomatize etmek için Keras Tuner kütüphanesi kullanılmıştır.

Teknik Yaklaşım:

Geniş Arama Uzayı (Search Space): Katman sayısı, nöron sayısı, aktivasyon fonksiyonu, optimizer tipi, öğrenme oranı, L2 ve dropout oranları gibi birçok hiperparametre için bir arama uzayı tanımlanmıştır.

RandomSearch Stratejisi: Tanımlanan uzayda rastgele kombinasyonlar deneyerek en iyi performansı (val_loss) veren modeli bulmuştur.

🖼️ Evrişimli Sinir Ağları (CNN) ile Görüntü Sınıflandırma

Bilgisayarla Görü (Computer Vision) alanının temelini oluşturan Evrişimli Sinir Ağları (CNN), CIFAR-10 veri seti üzerinde çok sınıflı görüntü sınıflandırma görevi için uygulanmıştır.

Mimari Bileşenleri:

Conv2D: Görüntülerden hiyerarşik özellik haritaları (kenarlar, dokular, şekiller) çıkarmak için kullanılır.

MaxPooling2D: Özellik haritalarını alt-örnekleyerek (downsampling) boyutsal azaltma yapar ve en belirgin özellikleri korur.

Flatten & Dense: Çıkarılan özelliklerin son sınıflandırma katmanlarına bağlanmasını sağlar.

Eğitim Stratejisi:

sparse_categorical_crossentropy kayıp fonksiyonu, one-hot encode edilmemiş etiketler için verimli bir şekilde kullanılmıştır.

ModelCheckpoint ile eğitim sırasındaki en iyi model kaydedilmiştir.

📈 Tekrarlayan Sinir Ağları (RNN) ile Zaman Serisi Tahmini

Sıralı veriler için özel olarak tasarlanmış RNN mimarileri, elektrik tüketim verisi üzerinde zaman serisi tahminlemesi (time series forecasting) yapmak amacıyla karşılaştırmalı olarak incelenmiştir.

Model Karşılaştırması:

Simple RNN: Temel sıralı bağımlılıkları yakalamak için bir referans model.

LSTM (Long Short-Term Memory): Uzun vadeli bağımlılıkları kapı mekanizmaları ile öğrenen gelişmiş mimari.

GRU (Gated Recurrent Unit): LSTM'e benzer performans gösteren, daha verimli ve daha az karmaşık bir mimari.

Teknik Pipeline:

Pencereleme (Windowing): Zaman serisi verisi, denetimli öğrenme formatına (girdi dizileri ve hedef değerler) dönüştürülmüştür.

Tersine Ölçeklendirme (Inverse Scaling): Model tahminleri, MAE ve RMSE gibi anlamlı metriklerle değerlendirilmeden önce orijinal ölçeğe geri getirilmiştir.

🚀 Transfer Öğrenme ve İnce Ayar (Transfer Learning & Fine-Tuning)

Derin öğrenmedeki en güçlü tekniklerden biri olan Transfer Öğrenme, büyük veri setleri (ImageNet) üzerinde önceden eğitilmiş MobileNetV2 modeli kullanılarak "kediler ve köpekler" sınıflandırma problemi için uygulanmıştır.

İki Aşamalı Eğitim Stratejisi:

Özellik Çıkarımı (Feature Extraction): Önceden eğitilmiş modelin evrişimsel katmanları dondurulmuş ve sadece yeni eklenen sınıflandırma katmanları eğitilmiştir.

İnce Ayar (Fine-Tuning): Temel modelin üst katmanları çözülmüş ve tüm model, daha düşük bir öğrenme oranı ile yeniden eğitilerek önceden öğrenilmiş özelliklerin yeni göreve adapte olması sağlanmıştır.

🔧 Veri Ön İşleme ve Pipeline En İyi Pratikleri

Tüm projelerde, model performansını doğrudan etkileyen tutarlı ve modern veri hazırlama teknikleri kullanılmıştır:

Temel Ön İşleme Adımları:

Ölçeklendirme (Scaling): StandardScaler ve MinMaxScaler ile özelliklerin belirli bir aralığa getirilmesi ve veri sızıntısının önlenmesi.

Zaman Odaklı Bölümleme (Time-based Splitting): Zaman serisi verilerinde zamansal bütünlüğün korunması.

TensorFlow Veri Pipeline'ı (tf.data):

Büyük veri setlerinde verimli eğitim için .shuffle(), .batch() ve .prefetch() metodlarının etkin kullanımı.

🎯 Performans Değerlendirme ve Metrikler

Farklı problem tipleri için doğru değerlendirme metrikleri seçilmiştir:

Sınıflandırma Metrikleri: Accuracy, Loss, Precision, Recall, Confusion Matrix.

Tahminleme (Forecasting) Metrikleri: MAE (Ortalama Mutlak Hata), RMSE (Kök Ortalama Kare Hata).

Analiz Araçları: Eğitim/Validasyon kayıp ve başarım eğrilerinin Matplotlib ile görselleştirilerek aşırı öğrenme gibi problemlerin teşhis edilmesi.

🎓 Key Learnings ve Gelecek Yönelimler

Bu eğitim serisi, farklı derin öğrenme mimarilerinin ne zaman ve nasıl kullanılacağını pratik olarak göstermiştir. Gelecek adımlar için aşağıdaki ileri seviye konuların incelenmesi önerilmektedir:

Transformer Mimarileri ve Attention Mekanizması

Üretken Modeller (GANs, VAEs, Diffusion Models)

Pekiştirmeli Öğrenme (Reinforcement Learning)

Modellerin Üretime Alınması (Deployment with TF Serving, ONNX)
