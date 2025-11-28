# Sơ lược qua

Làm hệ thống tóm tắt text

# Ý tưởng

Cào dữ liệu Vnexpress , xử lí dữ liệu , xây dựng mô hình , xây dựng web => done

Ta đi từng bước từ 1 đến 4 như đang lần đường trong một mê cung chữ nghĩa, để thấy rõ từng viên gạch xây nên một hệ thống tóm tắt văn bản. Mỗi bước tôi giải thích chi tiết để bạn hiểu nguyên lý chứ không chỉ chạy theo code.

BƯỚC 1 — Thu thập & Chuẩn hóa dữ liệu

Một hệ thống tóm tắt không thể thông minh hơn dữ liệu mà nó được nuôi. Bạn cần hai phần: văn bản gốc và văn bản tóm tắt chuẩn (đối với abstractive), hoặc chỉ văn bản (đối với extractive).

A. Thu thập dữ liệu

Nguồn dữ liệu tùy bài toán:

• Bài báo, tin tức → VnExpress, Dân Trí, BBC, CNN
• Báo cáo, tài liệu khoa học
• Kho dữ liệu chuẩn như CNN/DailyMail, NEWSROOM, XSum, Gigaword, WikiHow

Nếu bạn làm tiếng Việt, bộ dữ liệu rất ít nên thường phải tự tạo bằng web crawler.

B. Làm sạch dữ liệu

Văn bản gốc bao giờ cũng dính “bụi”:
• ký tự lạ
• dấu xuống dòng lung tung
• HTML tag
• emoji
• khoảng trắng thừa
• định dạng số, dấu câu lộn xộn

Bạn xử lý bằng:

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # bỏ thẻ HTML
    text = re.sub(r'\s+', ' ', text)           # bỏ khoảng trắng thừa
    text = re.sub(r'[^\w\s.,!?]', '', text)    # bỏ ký tự lạ
    return text.strip()


Mục tiêu: văn bản sạch, liền mạch, dễ xử lý.

BƯỚC 2 — Tiền xử lý ngôn ngữ (Preprocessing)

Đây là lúc bạn “bẻ” văn bản thành đơn vị nhỏ hơn để máy đọc hiểu.

Tùy hướng tiếp cận:

A. Nếu bạn làm Extractive (truyền thống)

Bạn sẽ cần:

1. Tách câu

Để phân tích câu nào quan trọng.

import nltk
nltk.download('punkt')

sentences = nltk.sent_tokenize(text)

2. Tách từ (Tokenization)

Giúp tính TF-IDF hoặc các vector khác.

Với tiếng Việt nên dùng:
• underthesea
• pyvi
• VnCoreNLP

Ví dụ:

from underthesea import word_tokenize
tokens = word_tokenize(sentence)

3. Loại stopwords

Những từ “không thông tin”.

stopwords = set(["và", "là", "thì", "ở", ...])
tokens = [w for w in tokens if w not in stopwords]

4. Lemmatization/Stemming

Tiếng Anh dùng nhiều, tiếng Việt ít hiệu quả hơn.

B. Nếu bạn làm Abstractive (Transformer)

Tokenizer riêng của mô hình làm giúp bạn:
• không cần loại stopwords
• không cần stemming
• không tách từ thủ công

Ví dụ T5:

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
inputs = tokenizer(text, max_length=512, truncation=True)

BƯỚC 3 — Chọn chiến lược tóm tắt

Giống như chọn phong cách võ công: chém mạnh hay uốn lượn tinh tế.

A. Extractive Summarization (chọn câu quan trọng)

Máy không tạo câu mới mà lấy nguyên câu trong bài.

Cơ chế:
• đại diện mỗi câu thành vector
• tính độ quan trọng
• chọn ra top-k câu

Các kỹ thuật chính:

1. TF-IDF + Cosine Similarity

Tính tần suất từ: câu chứa từ hiếm thường quan trọng.

2. TextRank

Dựa trên PageRank: câu nào được câu khác “trỏ tới” nhiều thì quan trọng.

3. BERT Sentence Embeddings

Lấy vector câu bằng BERT, cluster để chọn câu đại diện.

B. Abstractive Summarization (tạo câu mới)

Đây là loại giống người nhất.

Mô hình phổ biến:
• T5
• BART
• PEGASUS
• mBART
• LLaMA fine-tune

Cơ chế:
• encoder đọc văn bản
• decoder sinh câu mới
• sử dụng attention để trích thông tin quan trọng

Cách làm:
• cần cặp (article, summary) để huấn luyện
• mô hình học cách diễn đạt lại thông tin

Độ khó cao hơn, nhưng kết quả đẹp hơn rất nhiều.

BƯỚC 4 — Xây mô hình

Tùy chiến lược bạn chọn ở bước 3.

A. Extractive — Ví dụ với TextRank

tách văn bản thành câu

tạo ma trận similarity giữa các câu

chạy PageRank để tính độ quan trọng

chọn top-n câu

Code mini:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sentences)

sim_matrix = cosine_similarity(tfidf)

# PageRank
scores = np.ones(len(sentences))
for _ in range(50):
    scores = 0.85 * sim_matrix.dot(scores) + 0.15

ranking = np.argsort(scores)[::-1]
summary = " ".join([sentences[i] for i in ranking[:3]])

B. Abstractive — Ví dụ với T5

Huấn luyện:

from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

inputs = tokenizer("summarize: " + article, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_length=100)
summary = tokenizer.decode(output[0], skip_special_tokens=True)


Nếu bạn muốn train riêng:
• chuẩn bị dataset
• dùng Trainer của HuggingFace
• fine-tune 2–4 giờ GPU

Kết luận ngắn

Bước 1–4 là bộ xương hoàn chỉnh:

chuẩn hóa văn bản

tách token/câu

chọn phương pháp tóm tắt

xây mô hình theo phương pháp đó

Từ bước 5 trở đi bạn sẽ tối ưu và đánh giá mô hình.

Nếu bạn cho biết bạn muốn làm extractive hay abstractive, tôi có thể viết cho bạn toàn bộ quy trình + code α→ω phù hợp dạng bài tập hoặc đồ án.
# Quy trình thực hiện

25/11 : Khang bắt đầu tìm hiểu về Vnexpress về sơ lược
26/11: Việt Tìm hiểu về tóm tắt text
26/11: Khang tìm hiểu được 2 cách tiếp cận là Extractive và Abstractive
26/11 : Khang crawl data thành công
27/11: Việt tìm hiểu chi tiết các thuật toán sẽ sử dụng trong project