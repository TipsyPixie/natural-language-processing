import math
import re
from enum import Enum
from string import punctuation
from typing import Callable, Dict, List, Tuple

import nltk
import numpy
from matplotlib import pyplot, transforms
from matplotlib.patches import Ellipse
from nltk.stem import PorterStemmer, StemmerI
from nltk.tokenize import TweetTokenizer
from pandas import DataFrame

nltk.download(info_or_id="stopwords", quiet=True)
nltk.download(info_or_id="punkt", quiet=True)
nltk.download(info_or_id="twitter_samples", quiet=True)


class Sentiment(float, Enum):
    POSITIVE = 1.0
    NEGATIVE = 0.0


def tweet_to_stems(
    tweet: str, stemmer_class: Callable[..., StemmerI] = PorterStemmer
) -> List[str]:
    """
    Cleans the text, tokenizes it into separate words, removes stopwords, and converts words to stems.
    :param tweet: the text to clean up
    :param stemmer_class:
    :return: converted stems
    """

    # remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https?://.*[\r\n]*", "", tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r"#", "", tweet)

    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stemmer = stemmer_class()
    return [
        stemmer.stem(word)
        for word in tokenizer.tokenize(text=tweet)
        if word not in stopwords and word not in punctuation
    ]


def build_freqs(tweets: List[str], ys: List[float]) -> Dict[Tuple[str, float], int]:
    """
    This counts how often a word in the 'corpus' (the entire set of tweets) was associated with a positive label `1` or a negative label `0`. It then builds the `freqs` dictionary, where each key is a `(word,label)` tuple, and the value is the count of its frequency within the corpus of tweets.
    :param tweets: a list of tweets
    :param ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    :return: a dictionary mapping each (word, sentiment) pair to its frequency
    """

    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = numpy.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in tweet_to_stems(tweet=tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor: str = "none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    :param x: array_like, shape (n, ) input data.
    :param y: array_like, shape (n, ) input data.
    :param matplotlib.axes.Axes ax: The axes object to draw the ellipse into.
    :param float n_std: The number of standard deviations to determine the ellipse's radiuses.
    :param str facecolor: The face color of the graph figure.
    :param kwargs: `~matplotlib.patches.Patch` properties
    :return matplotlib.patches.Ellipse:
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_log_likelihoods(
    positive_tweets: List[str], negative_tweets: List[str]
) -> Dict[str, Dict]:
    """
    Get lambda scores
    """

    freqs = build_freqs(
        tweets=positive_tweets + negative_tweets,
        ys=numpy.append(
            numpy.ones(shape=(len(positive_tweets), 1)),
            numpy.zeros(shape=(len(negative_tweets), 1)),
        ),
    )

    word_counts = {}
    positive_total = negative_total = 0
    for word in set((key[0] for key in freqs.keys())):
        word_counts[word] = {
            "positive": freqs.get((word, 1.0), 0),
            "negative": freqs.get((word, 0.0), 0),
        }
        positive_total += word_counts[word]["positive"]
        negative_total += word_counts[word]["negative"]

    # log(P(word|positive)), log(P(word|negative))
    # with Laplacian Smoothing
    return {
        word: {
            "positive": math.log(
                (counts["positive"] + 1) / (positive_total + len(word_counts))
            ),
            "negative": math.log(
                (counts["negative"] + 1) / (negative_total + len(word_counts))
            ),
        }
        for word, counts in word_counts.items()
    }


def get_bayes_features(
    positive_tweets: List[str],
    negative_tweets: List[str],
    log_likelihoods: Dict,
) -> DataFrame:
    features = []
    for tweet in positive_tweets + negative_tweets:
        # positive, negative, sentiment
        local_feature = [0.0, 0.0]
        for stem in tweet_to_stems(tweet=tweet):
            local_feature[0] += log_likelihoods[stem]["positive"]
            local_feature[1] += log_likelihoods[stem]["negative"]
        features.append(local_feature)

    return DataFrame(data=features, columns=["positive", "negative"])


def plot_vectors(
    vectors, colors=["k", "b", "r", "m", "c"], axes=None, fname="image.svg", ax=None
):
    # scale = 1
    # scale_units = 'x'
    x_dir = []
    y_dir = []

    for i, vec in enumerate(vectors):
        x_dir.append(vec[0][0])
        y_dir.append(vec[0][1])

    if ax is None:
        fig, ax2 = pyplot.subplots()
    else:
        ax2 = ax

    if axes is None:
        x_axis = 2 + numpy.max(numpy.abs(x_dir))
        y_axis = 2 + numpy.max(numpy.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]

    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])

    for i, vec in enumerate(vectors):
        ax2.arrow(
            0,
            0,
            vec[0][0],
            vec[0][1],
            head_width=0.05 * x_axis,
            head_length=0.05 * y_axis,
            fc=colors[i],
            ec=colors[i],
        )

    if ax is None:
        pyplot.show()
        fig.savefig(fname)
