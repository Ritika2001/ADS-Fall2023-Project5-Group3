from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import ImgData
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import re
from gensim.models import KeyedVectors

# Load the Word2Vec model (download if not already downloaded)

def load_model():
    if os.path.exists("./lib/data/GoogleNews-vectors-negative300.bin"):
        return KeyedVectors.load_word2vec_format('./lib/data/GoogleNews-vectors-negative300.bin', binary=True)
    else:
        return api.load('word2vec-google-news-300')

def read_and_preprocess():
    books = pd.read_csv("./lib/data/Books.csv")
    # users = pd.read_csv("./lib/data/Users.csv")
    ratings = pd.read_csv("./lib/data/Ratings.csv")
    df = pd.merge(ratings, books, on='ISBN', how='left')
    df = df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1)
    df = df.dropna()
    df.columns = df.columns.str.lower()
    df.drop_duplicates(subset=['isbn'], inplace=True)
    df.drop_duplicates(subset=['book-title'], inplace=True)
    df = df.copy()
    df['year-of-publication'] = df['year-of-publication'].astype(str)
    df['content'] = df['book-title'] + ' ' + df['book-author'] + '' + df['year-of-publication']
    df['content'] = df['content'].fillna('')
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    return books, df

books, df = read_and_preprocess()
model = load_model()

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector

# Function to compute average word vectors for all books
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

w2v_feature_array = averaged_word_vectorizer(corpus=df['tokenized_content'], model=model, num_features=300)

def get_recommendations(user_book):
    book_index = df[df['book-title'] == user_book].index[0]
    user_book_vector = w2v_feature_array[book_index].reshape(1, -1)
    similarity_scores = cosine_similarity(user_book_vector, w2v_feature_array)
    similar_books = list(enumerate(similarity_scores[0]))
    sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:20]
    sorted_similar_book_titles = []
    for i, score in sorted_similar_books:
        sorted_similar_book_titles.append(df.iloc[i]['book-title'])
    return sorted_similar_book_titles

def get_book_image(book_title):
    # Clean the book title
    cleaned_title = book_title.replace(" ", "_")
    cleaned_title = re.sub(r'\W+', '', cleaned_title)
    
    # Check if the image exists in the folder
    file_path = os.path.join(f"./lib/www/{cleaned_title}.jpg")
    if os.path.exists(file_path):
        return file_path
    
    # If image not found in the folder, find URL from DataFrame and download
    book_row = books[books['Book-Title'] == book_title]
    if not book_row.empty:
        image_url = book_row.iloc[0]['Image-URL-L']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.save(file_path, "JPEG")
            return file_path
        else:
            print(f"Failed to download image for '{book_title}'")
    else:
        print(f"Book '{book_title}' not found in the DataFrame")
    
    return None

app_ui = ui.page_fluid(
    ui.panel_title("Book Recommendation"),
    ui.navset_tab(
        ui.nav("Home"),
        ui.nav("Search",
                    ui.row(
                        ui.column(4, ui.input_text("txt", "Search", placeholder="Search book")),
                        ui.column(4, ui.input_action_button("button", "Recommend"))
                    ),
                    ui.row(
                        ui.column(4, ui.output_text("text")),
                    ),
                    ui.row(
                        ui.column(3, ui.output_image("image1"), ui.output_text("text1")),
                        ui.column(3, ui.output_image("image2"), ui.output_text("text2")),
                        ui.column(3, ui.output_image("image3"), ui.output_text("text3")),
                        ui.column(3, ui.output_image("image4"), ui.output_text("text4")),
                    ),
                    ui.row(
                        ui.column(3, ui.output_image("image5"), ui.output_text("text5")),
                        ui.column(3, ui.output_image("image6"), ui.output_text("text6")),
                        ui.column(3, ui.output_image("image7"), ui.output_text("text7")),
                        ui.column(3, ui.output_image("image8"), ui.output_text("text8"))
                    )
               )
               
    )
    
)

def server(input: Inputs, output: Outputs, session: Session):
    # def book_list():
    #     if len(input.txt())==0:
    #         user_book = "The Notebook"
    #     else:
    #         user_book = input.txt()
    #     sorted_similar_book_titles = get_recommendations(user_book)
    #     return sorted_similar_book_titles

    # book_titles_list = book_list()


    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return "Getting Recommendations for " + input.txt()

# -------------------1-----------------------------------------------
    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image1():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[0]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text1():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[0])

# -------------------2-----------------------------------------------
    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image2():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[1]
        img_src = get_book_image(book_title)
        print(img_src)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text2():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[1])
    
    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image3():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[2]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text3():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[2])


    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image4():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[3]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text4():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[3])


    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image5():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[4]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text5():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[4])


    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image6():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[5]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text6():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[5])


    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image7():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[6]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text7():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[6])


    @output
    @render.image
    @reactive.event(input.button, ignore_none=False)
    def image8():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        book_title = sorted_similar_book_titles[7]
        img_src = get_book_image(book_title)
        img: ImgData = {"src": img_src, "height": "400px"}
        return img
    
    @output
    @render.text
    @reactive.event(input.button, ignore_none=False)
    def text8():
        if len(input.txt())==0:
            user_book = "The Notebook"
        else:
            user_book = input.txt()
        sorted_similar_book_titles = get_recommendations(user_book)
        return str(sorted_similar_book_titles[7])

app = App(app_ui, server)


