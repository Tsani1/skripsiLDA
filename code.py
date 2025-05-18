import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit.components.v1 as components
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser

import os
import os
import nltk

# Buat folder lokal untuk data NLTK agar tidak kena izin akses
nltk_data_dir = "nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Tambahkan ke path NLTK
nltk.data.path.append(nltk_data_dir)

required_nltk = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords"
}

for pkg, path in required_nltk.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)


# Fungsi load CSS dengan error handling
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS '{file_path}' tidak ditemukan, abaikan styling.")

# Load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

st.markdown('<h1 class="main-title">Aplikasi LDA untuk Analisis Topik Berita Politik</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📥 Upload file CSV (wajib kolom 'Ringkasan' dan 'Tanggal')", type="csv")

st.sidebar.header("⚙️ Pengaturan Model")
custom_stopwords = st.sidebar.text_area("🛑 Stopwords Tambahan (pisahkan koma)", "")
custom_stopwords_set = set(filter(None, map(str.strip, custom_stopwords.lower().split(",")))) if custom_stopwords else set()
num_topics_manual = st.sidebar.slider("📌 Jumlah Topik Manual", min_value=3, max_value=8, value=4)
num_passes = st.sidebar.selectbox("🔁 Iterasi (passes)", [5, 10, 15, 20], index=3)

st.sidebar.header("🔧 Kontrol")
proses_btn = st.sidebar.button("🔄 Proses Teks")
latih_btn = st.sidebar.button("🔍 Latih Model LDA")
visualisasi_btn = st.sidebar.button("🌐 Tampilkan Visualisasi")

st.sidebar.markdown("---")

st.sidebar.header("📊 Pengaturan Visualisasi")
freq_option = st.sidebar.selectbox(
    "📊 Pilih Frekuensi Tren Topik",
    ["Bulanan", "Kuartalan", "Tahunan"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Pilih Topik Terbaik Otomatis")
auto_run_btn = st.sidebar.button("🔎 Cari Jumlah Topik Terbaik (3-8)")

# State untuk simpan data dan model
for key in ["raw_df", "df", "corpus", "dictionary", "lda_model", "num_topics"]:
    if key not in st.session_state:
        st.session_state[key] = None

def clean_text(text, stop_words, stemmer):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if not {"Ringkasan", "Tanggal"}.issubset(raw_df.columns):
            st.error("❌ Kolom 'Ringkasan' dan 'Tanggal' wajib ada di CSV!")
            st.stop()

        st.session_state.raw_df = raw_df.copy()

        st.markdown('<h2 class="subtitle">🗂️ Data Sebelum Preprocessing</h2>', unsafe_allow_html=True)
        st.dataframe(raw_df[["Tanggal", "Ringkasan"]].head())

        if proses_btn:
            df = raw_df[["Ringkasan", "Tanggal"]].dropna()
            df["Tanggal"] = df["Tanggal"].astype(str).str.extract(r"(\d{1,2} \w+ \d{4})")[0]
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d %b %Y", errors="coerce")
            df = df.dropna(subset=["Tanggal"])
            df["Tahun-Bulan"] = df["Tanggal"].dt.to_period("M")

            stemmer = StemmerFactory().create_stemmer()
            default_stopwords = set(stopwords.words("indonesian")).union(
                {"yang", "di", "dan", "untuk", "ini", "dalam", "tidak", "dari", "akan"}
            )
            stop_words = default_stopwords.union(custom_stopwords_set)

            df["cleaned_tokens"] = df["Ringkasan"].apply(
                lambda x: clean_text(x, stop_words, stemmer)
            )
            df = df[df["cleaned_tokens"].map(len) > 0]

            bigram = Phrases(df["cleaned_tokens"], min_count=5, threshold=10)
            bigram_mod = Phraser(bigram)
            tokens_bigram = df["cleaned_tokens"].apply(lambda x: bigram_mod[x])

            trigram = Phrases(tokens_bigram, min_count=5, threshold=10)
            trigram_mod = Phraser(trigram)
            df["cleaned_tokens"] = tokens_bigram.apply(lambda x: trigram_mod[x])

            if df.empty:
                st.error("❌ Semua data kosong setelah preprocessing.")
                st.stop()

            st.session_state.df = df
            st.success("✅ Preprocessing selesai.")
            st.markdown('<h2 class="subtitle">🧾 Data Setelah Preprocessing</h2>', unsafe_allow_html=True)
            st.dataframe(df[["Tanggal", "Ringkasan", "cleaned_tokens"]].head())

        if st.session_state.df is not None and latih_btn:
            df = st.session_state.df
            dictionary = corpora.Dictionary(df["cleaned_tokens"])
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [dictionary.doc2bow(text) for text in df["cleaned_tokens"]]

            if len(dictionary) == 0:
                st.error("❌ Kamus kosong. Periksa preprocessing.")
                st.stop()

            with st.spinner("🔄 Melatih model..."):
                lda_model = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics_manual,
                    passes=num_passes,
                    alpha="auto",
                    eta="auto",
                )

            st.session_state.corpus = corpus
            st.session_state.dictionary = dictionary
            st.session_state.lda_model = lda_model
            st.session_state.num_topics = num_topics_manual

            coherence_model = models.CoherenceModel(
                model=lda_model,
                texts=df["cleaned_tokens"],
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence_score = coherence_model.get_coherence()
            st.success(f"✅ Model selesai. Coherence Score: {coherence_score:.4f}")

            st.markdown('<h2 class="subtitle">📌 Topik dan Kata Kunci</h2>', unsafe_allow_html=True)
            for idx, topic in lda_model.print_topics(num_words=10):
                st.markdown(f"**Topik {idx + 1}:** {topic}")

            st.markdown('<h2 class="subtitle">☁️ Wordcloud per Topik</h2>', unsafe_allow_html=True)
            for t in range(num_topics_manual):
                st.markdown(f"**Topik {t + 1}**")
                topic_terms = lda_model.show_topic(t, topn=30)
                word_freq = {term: weight for term, weight in topic_terms}
                fig, ax = plt.subplots()
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        if auto_run_btn:
            df = st.session_state.df
            if df is None:
                st.warning("⚠️ Data belum diproses. Silakan proses teks terlebih dahulu.")
            else:
                dictionary = corpora.Dictionary(df["cleaned_tokens"])
                dictionary.filter_extremes(no_below=5, no_above=0.5)
                corpus = [dictionary.doc2bow(text) for text in df["cleaned_tokens"]]

                if len(dictionary) == 0:
                    st.error("❌ Kamus kosong. Periksa preprocessing.")
                    st.stop()

                best_coherence = -1
                best_model = None
                best_topics = None
                progress_bar = st.progress(0)
                for k in range(3, 9):
                    lda = models.LdaModel(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=k,
                        passes=num_passes,
                        alpha="auto",
                        eta="auto",
                    )
                    coherence_model = models.CoherenceModel(
                        model=lda,
                        texts=df["cleaned_tokens"],
                        dictionary=dictionary,
                        coherence="c_v",
                    )
                    coherence = coherence_model.get_coherence()
                    if coherence > best_coherence:
                        best_coherence = coherence
                        best_model = lda
                        best_topics = k
                    progress_bar.progress((k-2)/6)
                st.session_state.lda_model = best_model
                st.session_state.corpus = corpus
                st.session_state.dictionary = dictionary
                st.session_state.num_topics = best_topics

                st.success(f"✅ Model terbaik ditemukan dengan jumlah topik: {best_topics}, Coherence Score: {best_coherence:.4f}")

                st.markdown('<h2 class="subtitle">📌 Topik dan Kata Kunci (Model Terbaik)</h2>', unsafe_allow_html=True)
                for idx, topic in best_model.print_topics(num_words=10):
                    st.markdown(f"**Topik {idx + 1}:** {topic}")

                st.markdown('<h2 class="subtitle">☁️ Wordcloud per Topik (Model Terbaik)</h2>', unsafe_allow_html=True)
                for t in range(best_topics):
                    st.markdown(f"**Topik {t + 1}**")
                    topic_terms = best_model.show_topic(t, topn=30)
                    word_freq = {term: weight for term, weight in topic_terms}
                    fig, ax = plt.subplots()
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

        if st.session_state.lda_model is not None and visualisasi_btn:
            df = st.session_state.df
            corpus = st.session_state.corpus
            dictionary = st.session_state.dictionary
            lda_model = st.session_state.lda_model
            num_topics = st.session_state.num_topics

            # Gunakan freq_option dari sidebar
            if freq_option == "Bulanan":
                df["Period"] = df["Tanggal"].dt.to_period("M").astype(str)
            elif freq_option == "Kuartalan":
                df["Period"] = df["Tanggal"].dt.to_period("Q").astype(str)
            else:
                df["Period"] = df["Tanggal"].dt.to_period("Y").astype(str)

            topic_dist = []
            for bow in corpus:
                topic_probs = lda_model.get_document_topics(bow)
                if topic_probs:
                    topic_dist.append(max(topic_probs, key=lambda x: x[1])[0])
                else:
                    topic_dist.append(-1)

            df["Dominant_Topic"] = topic_dist
            df = df[df["Dominant_Topic"] != -1]

            topic_trends = df.groupby(["Period", "Dominant_Topic"]).size().unstack(fill_value=0)

            st.markdown(f'<h2 class="subtitle">📈 Tren Topik per {freq_option}</h2>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12,6))
            topic_trends.plot(ax=ax, marker='o')
            ax.set_ylabel("Jumlah Dokumen")
            ax.set_xlabel(freq_option)
            ax.legend([f"Topik {i+1}" for i in range(num_topics)], title="Topik")
            ax.grid(True)
            st.pyplot(fig)

            st.markdown('<h2 class="subtitle">🗺️ Visualisasi Interaktif dengan pyLDAvis</h2>', unsafe_allow_html=True)
            with st.spinner("Menyiapkan visualisasi PyLDAvis..."):
                try:
                    vis = gensimvis.prepare(lda_model, corpus, dictionary)
                    html_string = pyLDAvis.prepared_data_to_html(vis)
                    components.html(html_string, width=1300, height=800, scrolling=True)
                except Exception as e:
                    st.warning(f"Gagal menampilkan visualisasi PyLDAvis: {e}")

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {e}")
else:
    st.info("⬆ Silakan upload file CSV untuk memulai.")


