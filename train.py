# import tensorflow as tf
import numpy as np
import re
import pickle


def tokenize(text):
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text)


def mapping(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word


def generate_training_data(tokens, window_size_left, dataset_method):
    x = []
    y = []

    if dataset_method == "skipgram":
        for i in range(window_size_left, len(tokens) - (window_size_left)):
            for j in range(-window_size_left, window_size_left + 1):
                if j != 0:
                    x.append(tokens[i])
                    y.append(tokens[i + j])

    if dataset_method == "sliding_window":
        for i in range(len(tokens) - (window_size_left)):
            for j in range(window_size_left):
                x.append(tokens[i])
                y.append(tokens[i + j + 1])

    return x, y


def dataset_to_id(x, y, word_to_id):
    for i in range(len(x)):
        x[i] = word_to_id[x[i]]
        y[i] = word_to_id[y[i]]

    return x, y


def one_hot_encoding(data, vocab_size):
    ohe = []
    for i in range(len(data)):
        each_data = [0] * vocab_size
        each_data[data[i]] = 1
        ohe.append(each_data)

    return ohe


def init_weights(vocab_size, n_embed):
    model = {
        "w1": np.random.randn(vocab_size, n_embed),
        "w2": np.random.randn(n_embed, vocab_size),
    }

    return model


def softmax(output):
    soft = []
    for each_output in output:
        soft.append(np.exp(each_output) / np.exp(each_output).sum())

    return np.array(soft)


def forward_prop(model, x):
    outputs = {}

    outputs["embed"] = x @ model["w1"]
    outputs["output"] = outputs["embed"] @ model["w2"]
    outputs["softmax"] = softmax(outputs["output"])

    return outputs


def cross_entropy(z, y):
    return -np.sum(np.log(z) * y)


def backward(model, X, y, alpha):
    outputs = forward_prop(model, X)
    da2 = outputs["softmax"] - y
    dw2 = outputs["embed"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert dw2.shape == model["w2"].shape
    assert dw1.shape == model["w1"].shape
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2

    return cross_entropy(outputs["softmax"], y)


def training(
    text,
    window_size_left,
    learning_rate,
    n_iter,
    output_folder,
    n_embed,
    dataset_method,
):
    print("Tokenizing text......")
    tokens = tokenize(text)

    print("Creating mappings......")
    word_to_id, id_to_word = mapping(tokens)

    print("Generating dataset......")
    x, y = generate_training_data(
        tokens, window_size_left=2, dataset_method=dataset_method
    )

    print(f"window size = {window_size_left})")
    print(f"number of datapoints = {len(x)}")

    print("Mapping dataset to ID")
    x, y = dataset_to_id(x, y, word_to_id)

    x = np.array(x)
    y = np.array(y)

    x_ohe = np.array(one_hot_encoding(x, len(word_to_id)))
    y_ohe = np.array(one_hot_encoding(y, len(word_to_id)))

    model = init_weights(len(word_to_id), n_embed)

    history = [backward(model, x_ohe, y_ohe, learning_rate) for _ in range(n_iter)]

    with open(output_folder + "/model.pkl", "wb") as outfile:
        pickle.dump(model, outfile)

    with open(output_folder + "/word_to_id.pkl", "wb") as outfile:
        pickle.dump(word_to_id, outfile)

    with open(output_folder + "/id_to_word.pkl", "wb") as outfile:
        pickle.dump(id_to_word, outfile)

    return history


def inference(word, output_folder):
    with open(output_folder + "/model.pkl", "rb") as outfile:
        model = pickle.load(outfile)

    with open(output_folder + "/word_to_id.pkl", "rb") as outfile:
        word_to_id = pickle.load(outfile)

    with open(output_folder + "/id_to_word.pkl", "rb") as outfile:
        id_to_word = pickle.load(outfile)

    x = one_hot_encoding([word_to_id[word]], len(word_to_id))
    result = forward_prop(model, x)

    probabs = result["softmax"][0]

    output_words = [id_to_word[x] for x in np.argsort(probabs)]
    output_word_probs = np.sort(probabs)

    return output_words, output_word_probs


def gen_text(start_word, num_gens, output_folder):
    gen = []

    for i in range(num_gens):
        gen.append(start_word)
        output_words, _ = inference(start_word, output_folder)
        start_word = output_words[0]

    output_str = ""
    for i in range(len(gen)):
        output_str += f"{gen[i]} "

    return output_str


training_text = """Machine learning is the study of computer algorithms that \
            improve automatically through experience. It is seen as a \
            subset of artificial intelligence. Machine learning algorithms \
            build a mathematical model based on sample data, known as \
            training data, in order to make predictions or decisions without \
            being explicitly programmed to do so. Machine learning algorithms \
            are used in a wide variety of applications, such as email filtering \
            and computer vision, where it is difficult or infeasible to develop \
            conventional algorithms to perform the needed tasks."""


class Main:
    def __init__(
        self,
        window_size_left=5,
        learning_rate=0.02,
        n_iter=50,
        n_embed=500,
        output_folder="model/",
        start_word="machine",
        num_gens=200,
        is_train=True,
        dataset_method="skipgram",
        training_text=training_text,
    ) -> None:

        self.window_size_left = window_size_left
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_embed = n_embed
        self.output_folder = output_folder
        self.start_word = start_word
        self.num_gens = num_gens
        self.dataset_method = dataset_method
        self.training_text = training_text

        if is_train:
            self.train()

        self.generate_text()

    def train(self):

        training(
            self.training_text,
            self.window_size_left,
            self.learning_rate,
            self.n_iter,
            self.output_folder,
            self.n_embed,
            self.dataset_method,
        )

    def generate_text(self):
        generated = gen_text(self.start_word, self.num_gens, self.output_folder)

        return generated

    # output_words, output_word_probs = inference("learning", id_to_word)
