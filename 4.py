# TODO popuniti kodom za problem 4

import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk


# # First run:
# !pip install nltk
# import nltk
# nltk.download()


# Funkcija za čišćenje podataka
def preprocess_text(text, stop_words, lemmatizer):
    # Pretvori u mala slova
    text = text.lower()

    # Ukloni URL-ove
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Ukloni korisnička imena i haštagove
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Ukloni RT oznake
    text = re.sub(r'^RT[\s]+', '', text)

    # Ukloni emotikone i nespecijalne karaktere
    text = re.sub(r'[:;=][\'"]?[)({\/dpoO]', '', text)
    text = re.sub(r'\\x\w{2}', '', text)  # Uklanja heksadecimalne kodove

    # Tokenizacija
    tokens = wordpunct_tokenize(text)

    # Ukloni stop-reči i interpunkcijske znakove
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Lematizacija
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Ponovno spajanje tokena u string
    return ' '.join(tokens)


# Učitavanje podataka
file_name = 'data/disaster-tweets.csv';
df = pd.read_csv(file_name)

# Priprema stop-reči i lematizatora
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Čišćenje podataka
df['clean_text'] = df['text'].apply(preprocess_text, args=(stop_words, lemmatizer))


# Kreiranje skupa reči (vokabulara)
word_freq = Counter()
for tweet in df['clean_text']:
    for word in tweet.split():
        word_freq[word] += 1

print(f"Total unique words: {len(word_freq)}")
print(f"Most common words: {word_freq.most_common(100)}")

# Filtriranje reči koje se javljaju najmanje 2 puta
filtered_vocab = [word for word, count in word_freq.items() if count >= 2]
max_vocab_size = 10000
vocab = filtered_vocab[:max_vocab_size]

# Optimizacija BoW koristeći numpy za efikasniji rad sa velikim matricama
def create_bow_features(texts, vocabulary):
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    X = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)  # Koristimo float32 za manju potrošnju memorije

    for i, text in enumerate(texts):
        word_counts = Counter(text.split())
        for word, count in word_counts.items():
            if word in word_to_index:
                X[i, word_to_index[word]] = count

    return X

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['target'], test_size=0.2, stratify=df['target'])

# Kreiranje BoW vektora
X_train_bow = create_bow_features(X_train, vocab)
X_test_bow = create_bow_features(X_test, vocab)

# Resetovanje indeksa u y_train i y_test 
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Implementacija Multinomial Naive Bayes sa optimizacijama
class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount

    def fit(self, X, Y):
        nb_examples = X.shape[0]
        self.priors = np.bincount(Y) / nb_examples
        occs = np.zeros((self.nb_classes, self.nb_words), dtype=np.float32)  # Koristimo float32 za manju potrošnju memorije
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                occs[c][w] += X[i][w]
        self.like = np.zeros((self.nb_classes, self.nb_words), dtype=np.float32)
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.pseudocount
                down = np.sum(occs[c]) + self.nb_words * self.pseudocount
                self.like[c][w] = up / down

    def predict(self, bow):
        probs = np.zeros(self.nb_classes, dtype=np.float32)  # Koristimo float32 za manju potrošnju memorije
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for w in range(self.nb_words):
                prob += bow[w] * np.log(self.like[c][w])
            probs[c] = prob
        return np.argmax(probs)

# Treniranje modela
model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)
model.fit(X_train_bow, y_train)

# Predikcija i tačnost
y_pred = np.array([model.predict(X_test_bow[i]) for i in range(len(X_test_bow))])
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Kreiranje pozitivnih i negativnih tvitova
positive_tweets = df[df['target'] == 1]['clean_text']
negative_tweets = df[df['target'] == 0]['clean_text']

# Funkcija za brojanje reči u pozitivnim i negativnim tvitovima
def get_most_common_words(tweets, min_count=10):
    word_freq = Counter()
    for tweet in tweets:
        for word in tweet.split():
            word_freq[word] += 1
    return {word: count for word, count in word_freq.items() if count >= min_count}

# Pronaći 5 najčešće korišćenih reči u pozitivnim tvitovima
positive_word_freq = get_most_common_words(positive_tweets)
negative_word_freq = get_most_common_words(negative_tweets)

# Najčešće reči u pozitivnim i negativnim tvitovima
most_common_pos = sorted(positive_word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
most_common_neg = sorted(negative_word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"Najčešće reči u pozitivnim tvitovima: {most_common_pos}")
print(f"Najčešće reči u negativnim tvitovima: {most_common_neg}")

# Računanje LR metrike
lr_metric = {}
for word in positive_word_freq:
    if word in negative_word_freq:
        lr = positive_word_freq[word] / negative_word_freq[word]
        lr_metric[word] = lr

# Pronaći 5 reči sa najvećim i najmanjim LR
top_5_lr = sorted(lr_metric.items(), key=lambda x: x[1], reverse=True)[:5]
bottom_5_lr = sorted(lr_metric.items(), key=lambda x: x[1])[:5]

print(f"5 reči sa najvećim LR: {top_5_lr}")
print(f"5 reči sa najmanjim LR: {bottom_5_lr}")


#Diskusija

# Reč "fire" je izuzetno česta u pozitivnim tvitovima, što je očekivano s obzirom na to da se odnosi na katastrofe
# ili opasnosti. S druge strane, reči poput "like" i "??" su češće u negativnim tvitovima, što može ukazivati na
# stil pisanja koji nije direktno vezan za katastrofe, već više za svakodnevnu interakciju ili nesigurnost u izražavanju.

# Reči sa visokim LR vrednostima kao što su "train", "fatal", "oil", i "evacuation" jasno ukazuju na katastrofe, nesreće
# i opasnosti, što ih čini vrlo korisnim indikatorima za detekciju tvitova koji se odnose na katastrofe.
# Reči sa niskim LR vrednostima, poput "love", "want", "let", i "??", mnogo su češće u negativnim tvitovima. Ove reči su
# uglavnom neutralne ili pozitivne i često se koriste u svakodnevnim konverzacijama, što ih čini manje relevantnim za
# prepoznavanje katastrofalnih događaja.

# LR metrika predstavlja odnos između broja pojavljivanja reči u pozitivnim i negativnim tvitovima.
# LR(reč) = broj pojavljivanja u pozitivnim tvitovima / broj pojavljivanja u negativnim tvitovima.
# Metrika LR daje uvid u to koliko je neka reč specifična za pozitivne (katastrofalne) ili negativne (nekatazrofalne) tvitove.
# Reči sa visokim LR vrednostima su korisni prediktori za detekciju katastrofalnih tvitova, jer se one češće pojavljuju u kontekstu
# nesreća i ozbiljnih događaja.
# Reči sa niskim LR vrednostima uglavnom su prisutne u negativnim tvitovima, što ukazuje na to da nisu od pomoći za prepoznavanje
# katastrofalnih događaja, već su više povezane sa svakodnevnim temama.