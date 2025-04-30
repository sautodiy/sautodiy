import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab
from nltk.stem.snowball import SnowballStemmer
import joblib
import streamlit as st
import os

class Processor:
    """
    Класс для автоматического поиска аналогичных товаров по их названиям.

    Предназначен для решения задачи сопоставления товаров компании с аналогами,
    найденными в списке товаров.
    
    Алгоритм включает:
    - Лемматизацию и стемминг названий для нормализации текста.
    - Векторизацию обработанных текстов.
    - Кластеризацию товаров по смысловой близости.
    - Поиск наиболее похожих товаров на основе косинусного сходства.
    """
    def __init__(self, n_clusters=500):
        """
        Инициализация модели сопоставления товаров.
    
        Параметры:
        n_clusters (int): Количество кластеров для группировки товаров.
        """
        self.n_clusters = n_clusters
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.morph_vocab = MorphVocab()
        self.stemmer = SnowballStemmer("russian")

        self.vectorizer_lemmas = CountVectorizer(max_features=1000)
        self.vectorizer_stemmed = CountVectorizer(max_features=1000)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")

        self.X_train_np = None
        self.df = None

    def save(self, path):
        """
        Сохраняет обученную модель в файл.
    
        Параметры:
        path : Путь, по которому будет сохранён файл модели (.joblib).
        """
        joblib.dump({
            'vectorizer_lemmas': self.vectorizer_lemmas,
            'vectorizer_stemmed': self.vectorizer_stemmed,
            'kmeans': self.kmeans,
            'df': self.df,
        }, path)

    @classmethod
    def load(cls, path):
        """
        Загружает модель из файла и возвращает готовый экземпляр Processor.
    
        Параметры:
        path : Путь к ранее сохранённому файлу модели.
    
        Возвращает:
        Processor : Объект модели, готовый к использованию.
        """
        data = joblib.load(path)
        proc = cls(n_clusters=data['kmeans'].n_clusters)
        proc.vectorizer_lemmas = data['vectorizer_lemmas']
        proc.vectorizer_stemmed = data['vectorizer_stemmed']
        proc.kmeans = data['kmeans']
        proc.df = data['df']

        X_lemmas = proc.vectorizer_lemmas.transform(proc.df['sku_name_lemmas'])
        X_stemmed = proc.vectorizer_stemmed.transform(proc.df['sku_name_stemmed'])
        X_sparse = hstack([X_lemmas, X_stemmed])
        proc.X_train_np = X_sparse.toarray()

        return proc


    def lemmatize(self, text):
        """
        Приводит текст к лемматизированной форме (начальные формы слов).
    
        Используется для очистки и нормализации наименований товаров.
    
        Параметры:
        text : Исходное название товара.
    
        Возвращает:
        str : Лемматизированный текст.
        """
        doc = Doc(text.lower())
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemmas.append(token.lemma)
        return ' '.join(lemmas)

    def stem_text(self, text):
        """
        Приводит текст к корневым формам слов (стемминг).
    
        Используется как дополнительная нормализация для построения признаков.
    
        Параметры:
        text : Исходное название товара.
    
        Возвращает:
        str : Стеммированная версия текста.
        """
        if not isinstance(text, str):
            return ''
        tokens = text.lower().split()
        stems = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stems)

    def fit(self, sku_names : pd.Series):
        """
        Обучает модель на списке товаров.
    
        Выполняется предобработка текстов, 
        векторизация и кластеризация. После вызова метода можно искать аналоги.
    
        Параметры:
        df : Таблица с колонкой 'sku_name', содержащей названия товаров.
        """
        self.df = pd.DataFrame()
        self.df['sku_name'] = sku_names.fillna('')
        self.df['sku_name'] = self.df['sku_name'].fillna('')
        self.df['sku_name_lemmas'] = self.df['sku_name'].apply(self.lemmatize)
        self.df['sku_name_stemmed'] = self.df['sku_name'].apply(self.stem_text)

        X_lemmas = self.vectorizer_lemmas.fit_transform(self.df['sku_name_lemmas'])
        X_stemmed = self.vectorizer_stemmed.fit_transform(self.df['sku_name_stemmed'])

        X_sparse = hstack([X_lemmas, X_stemmed])
        self.X_train_np = X_sparse.toarray()

        clusters = self.kmeans.fit_predict(self.X_train_np)
        self.df["cluster"] = clusters

    def predict(self, input_text: str, top_k=5):
        """
        Возвращает список наиболее похожих товаров по введённому названию.
    
        Модель ищет аналоги в пределах одного смыслового кластера
        и ранжирует их по косинусному сходству.
    
        Параметры:
        input_text : Название товара, для которого нужно найти аналоги.
    
        top_k : Количество аналогов, которые нужно вернуть.
    
        Возвращает:
        list of tuples : Список из пар (название товара, коэффициент сходства).
        """
        lemma_text = self.lemmatize(input_text)
        stem_text_val = self.stem_text(input_text)

        X_lemmas = self.vectorizer_lemmas.transform([lemma_text])
        X_stemmed = self.vectorizer_stemmed.transform([stem_text_val])
        X_combined = hstack([X_lemmas, X_stemmed]).toarray()

        cluster = self.kmeans.predict(X_combined)[0]
        same_cluster = self.df[self.df["cluster"] == cluster]

        if same_cluster.empty:
            return []

        candidate_vectors = self.X_train_np[same_cluster.index]
        #Корректировка top_k в зависимости от размера кластера
        actual_top_k = min(top_k, len(candidate_vectors))
        nn = NearestNeighbors(n_neighbors=actual_top_k, metric='cosine')
        nn.fit(candidate_vectors)
        distances, indices = nn.kneighbors(X_combined)
    
        results = []
        for i in range(len(indices[0])):
            real_idx = same_cluster.index[indices[0][i]]
            sim_score = 1 - distances[0][i]
            results.append((self.df.loc[real_idx, 'sku_name'], sim_score))
        return results

def main():
    # Streamlit
    st.title("Поиск похожих товаров Vink")

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Загрузка готовой модели
    st.markdown("### Загрузка предобученной модели")
    model_file = st.file_uploader("Загрузите файл модели (.joblib)", type=["joblib"])
    if model_file and st.button("Загрузить модель"):
        st.session_state.model = Processor.load(model_file)
        st.success("Модель успешно загружена")

    # Загрузка датасета
    st.markdown("### Загрузка данных и обучение")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'sku_name'", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'sku_name' not in df.columns:
            st.error("В файле не найдена колонка 'sku_name'")
        else:
            st.session_state.df = df

            if st.button("Обучить модель"):
                with st.spinner("Обучение модели..."):
                    model = Processor(n_clusters=1000)
                    model.fit(df['sku_name'])
                    st.session_state.model = model
                st.success("Обучение завершено")

    # Сохранение модели
    if st.session_state.model:
        with st.expander("💾 Сохранить обученную модель"):
            if st.button("Сохранить как .joblib"):
                path = "saved_model.joblib"
                st.session_state.model.save(path)
                with open(path, "rb") as f:
                    st.download_button("Скачать модель", f, file_name="vink_model.joblib")

    # --- Поиск похожих товаров ---
    if st.session_state.model:
        model = st.session_state.model
        query = st.text_input("Введите название товара")
        top_k = st.slider("Сколько похожих товаров показать", min_value=1, max_value=10, value=5)

        if st.button("Найти похожие") and query:
            results = model.predict(query, top_k=top_k)
            if not results:
                st.info("Похожих товаров не найдено")
            else:
                st.write("### 🔍 Похожие товары:")

                for name, score in results:
                    percent = int(score * 100)

                    # Цвет сходства
                    if score >= 0.85:
                        bar_color = "green"
                    elif score >= 0.65:
                        bar_color = "orange"
                    else:
                        bar_color = "gray"

                    # Вывод в колонках
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{name}**")
                    with col2:
                        st.markdown(
                            f"""
                            <div style="background: lightgray; border-radius: 5px; height: 20px; overflow: hidden;">
                                <div style="width: {percent}%; background: {bar_color}; height: 100%; text-align: right; padding-right: 5px; color: white;">
                                    {percent}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

if __name__ == '__main__':
    main()