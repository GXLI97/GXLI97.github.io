---
layout: post
title: Notes on RAND-WALK - A Latent Variable Model Approach to Word Embeddings
---

In this post, I will discuss a paper published by Arora et al. in 2015, titled [RAND-WALK: A Latent Variable Model Approach to Word Embeddings][1]. This paper is interesting because the authors provide a generative model that explains the success of word embedding algorithms that are popular in practice: [pointwise mutual information (PMI)][2], [word2vec][3], and [GloVe][4].

These notes are for my personal benefit, but may be useful for someone else who wants to learn about word embeddings.

## A Crash Course in Word Embeddings

The goal of word embeddings is to take words that appear in natural language text corpus: *The quick brown fox jumps over the lazy dog.* and produce an *embedding* in a vector space, say $$\mathbb{R}^d$$. Usually, if we denote $$n$$ as the size of the entire vocabulary, we will have $$d\ll n$$.

Of course, we can do this randomly: map each word in our vocabulary to a random vector $$v\in\mathbb{R}^d$$. For example, we could map "quick" to $$[0.13, 0.29, ... ]$$. However, state of the art word embeddings do something more interesting: they find a mapping for each word that allows them to capture *relations* between words. One property is that "similar" words have large positive dot product. Another property is that you are able to solve analogies: this is where you get the popular example **King - Man + Woman = Queen**. To be more precise, when you subtract the vector representing "man" from the vector representing "king", and then add the vector representing "woman", the closest vector to the result is the vector representing "queen".

How do we generate these vectors?

Intuitively, words that appear *near each other* in a string of text should be similar. This is formalized as the **distribution hypothesis** by linguists:

> You shall know a word by the company it keeps. Firth, 1957.

A simple way to use this distribution hypothesis is to build a co-occurence matrix $$X$$, where for some suitable $$k$$ (perhaps $$k=5$$):

$$X_{w,w'} = \text{# times word w and word w' occur in a block of k words.}$$

This forms the basis of **count-based methods**. In order to get $$d$$-dimensional embeddings from $$X_{w,w'}$$, it is popular to apply some non-linearity $$\phi(X_{w,w'})$$ elementwise, then do dimensionality reduction through singular value decomposition (SVD) and take the top $$d$$ eigenvectors. Popular non-linearities $$\phi$$ that people have found improve the ability of word embeddings to solve analogy tasks include the square root, log, or the **pointwise mutual information** (PMI) that is defined as follows:

$$\text{PMI}(X_{w,w'}) := \log\frac{p(w,w')}{p(w)p(w')},$$

where we use the empirical probabilities that we observe from our corpus processing.

In addition, Pennington et al. suggest another model called GloVe which solves a weighted-SVD factorization of the count matrix $$X$$ that uses some tricks to get good embeddings. 

A second way of producing word embeddings is through neural network language models such as word2vec. I won't go into the details of how exactly word2vec is implemented, but I recommend [this tutorial][5], which is the clearest explanation of word2vec that I've found! Essentially, the word embedding in this case is the matrix found inside the neural network, which is trying to distinguish correct word-context pairs which appear in the corpus from incorrect ones which do not appear in the corpus. Even so, there has been some work showing that word2vec is *implicitly factorizing* a PMI matrix by [Levy and Goldberg, 2014][6].

## So What are the Issues?

These methods are quite mysterious. How can we explain why they work, i.e. are able to produce vectors with linear structure that lets us solve analogies?

Another pertinent question is the issue of dimensionality. Often it is the case that the embeddings are in a much smaller space than the original vocabulary size. For example, following Levy and Goldberg's 2014 paper, when they process the text, the vocabulary size is $$n=189533$$, but they reduce dimensions down to $$d=1000$$. We must be inducing a large amount of approximation error when we approximate the original matrix $$X$$ as a low-rank matrix $$\widetilde{X}$$. This approximation error may obfuscate our ability to solve analogies: instead of picking the top correct word, maybe we pick the second word. But in practice, **low dimensional embeddings are able to solve analogies well**. Why?

A popular, but wrong explanation, is that in machine learning, *simple models generalize better*. This is fallacious, as the authors point out, because our process of producing word embeddings is entirely unsupervised - the notion of "generalization" is entirely irrelevant. We are not training models to explicitly solve analogies, so there is no reason why they should be able to solve analogies, *just like there is no reason why they should be able to predict tomorrow's weather*.

## A Generative Model

In order to answer these questions, Arora et al. propose a model that explains how the text corpus is generated, in turn shedding light on the co-occurence count matrix $$X$$. They also give an explanation as to why we are able to solve analogies with word embeddings. Lastly, they also show that their generative model has some nice connections to PMI, GloVe, and word2vec.

They make the assumption that each word can be represented by a vector $$v_w\in\mathbb{R}^d$$, and the task is to recover this vector.

As the title of the paper suggests, they introduce a latent variable, called a **discourse vector** that captures what is being talked about. This vector, denoted $$c_t\in\mathbb{R}^d$$, does a "lazy" random walk on the unit sphere, which captures our intuition that as we talk, the subject of the conversation changes slowly.

At each time step, the probability that a word $$w$$ is generated follows a log-linear model that relates the vector representing the word $$v_w$$ and the current discourse vector $$c_t$$:

$$\begin{equation} 
P(w|c_t) \propto \exp(\left<v_w, c_t\right>).
\tag{1}\label{eq:one}
\end{equation}$$

Intuitively, this means that if your word vectors align closely with the current discourse vector, your word is more likely to be emitted!

After your word is produced, you update your discourse vector randomly to get a new discourse: $$c_{t+1} = c_t + \delta_t$$, then generate a word again according to the above rule.

You might have noticed that in \eqref{eq:one}, we are using $$\propto$$ instead of $$=$$, because in general when we sum over all the words, the RHS expression will not sum to 1. In order for \eqref{eq:one} to be a valid probability distribution, we have to apply a *normalization constant*. The paper calls this the **partition function $$Z_c$$**:

$$Z_c = \sum_w  \exp(\left< v_w, c\right>).$$

Using this information, they derive closed form expressions for word co-occurence probabilities, thus giving us a view into why the word co-occurence counts look the way they do. This is done by integrating out the random variable $$c_t$$ using the law of total expectation:

$$\begin{align}
p(w,w') &= \mathbb{E}_{c,c'}[P(w, w'|c,c')] \\
	&= \mathbb{E}_{c,c'}[P(w|c) P(w'|c')]
\end{align}$$

The key step in the math is that they make the assumption that the word vectors $$v_w$$ behave like random vectors that are equally strewn about in $$\mathbb{R}^d$$ space. In probability theory, this is called **isotropy**.<sup>[1](#myfootnote1)</sup> In turn, this implies that for any discourse vector $$c_t$$, the value of the partition function should concentrate around a value $$Z_0$$. (Indeed, they empirically verify this claim!)

Eventually, you can derive closed form expressions:

$$\begin{align}
\log p(w,w') &= \frac{||v_w + v_{w'}||_2^2}{2d} - 2\log Z_0 \pm \epsilon \\
\log p(w) &= \frac{||v_w||_2^2}{2d}-\log Z_0 \pm \epsilon \\
\text{PMI}(w,w') &= \frac{\left< v_w, v_{w'}\right>}{d} += O(\epsilon)
\end{align}$$

(It can be shown with some simple algebra that the PMI expression can be derived from the first two).

So this validates the use of PMI - indeed, the value of the PMI matrix encodes this vector similarity through an inner product.

## Training Objective

They have some nice expressions for co-occurence probabilities, and now they go a step further and derive a training objective for recovering the vectors by standard optimization techniques such as stochastic gradient descent (SGD).


### Squared Norm
We proceed by maximum likelihood estimation on the observed coocurence matrix $$X_{w,w'}$$:

$$
\begin{align}
l &= \log \left(\prod_{(w,w')} p(w,w')^{X_{w,w'}} \right) \\
&= \sum_{w,w'} X_{w,w'} \log p(w,w') \\
&= \sum_{w,w'} X_{w,w'} (\log{\frac{\widetilde{L}p(w,w')}{X_{w,w'}}} + \log \frac{X_{w,w'}}{\widetilde{L}})
\end{align}$$

here $$\widetilde{L} := \sum_{w,w'}X_{w,w'}$$ is sum of all the cooccurences. Then let $$\Delta_{w,w'} = \log{\frac{\widetilde{L}p(w,w')}{X_{w,w'}}}$$ (log of the ratio between expected count and observed count).

$$
\begin{align}
&= c+\sum_{w,w'} X_{w,w'} \Delta_{w,w'}
\end{align}$$

For some constant c that we don't care about because it doesn't depend on our word vectors ($$\Delta$$ does).

Instead of maximizing this directly, instead we will rewrite the expression $$\sum_{w,w'} X_{w,w'} \Delta_{w,w'}$$. To begin, let's look at $$\widetilde{L}$$:

$$
\begin{align}
\widetilde{L} &= \sum_{w,w'}\widetilde{L} p(w,w') \\
&= \sum_{w,w'} \exp(\log \widetilde{L} p(w,w')) \\
&= \sum_{w,w'} \exp(\Delta_{w,w'} + \log(X_{w,w'})) \\
&= \sum_{w,w'} X_{w,w'} \exp(\Delta_{w,w'}) \\
&= \sum_{w,w'} X_{w,w'} (1 + \Delta_{w,w'} + \Delta^2_{w,w'}/2 + O(|\Delta_{w,w'}|^3))
\end{align}$$

The paper makes the argument that the last term in the Taylor series is negligible, so we have:

$$
\begin{align}
\widetilde{L} &\approx \sum_{w,w'} X_{w,w'} (1 + \Delta_{w,w'} + \Delta^2_{w,w'}/2) \\
&=  \widetilde{L} + \sum_{w,w'} X_{w,w'} (\Delta_{w,w'} + \Delta^2_{w,w'}/2)
\end{align}
$$

So:

$$c-l \approx \sum_{w,w'} X_{w,w'} \Delta^2_{w,w'}$$

Maximizing $$l$$ corresponds to:

$$\begin{align}
\sum_{w,w'} X_{w,w'} \Delta^2_{w,w'} = \boxed{\underset{\{v_w\}, C}{\text{minimize}}
\sum_{w,w'} X_{w,w'}\left(\log X_{w,w'} - \lVert v_w+v_w'\rVert_2^2 - C\right)^2}
\end{align}$$


### Similarity to GloVe
This is very similar to GloVe! Recall that GloVe's objective function is:

$$
\underset{\{v_w\}, C}{\text{minimize}}
\sum_{w,w'} f(X_{w,w'})\left(\log X_{w,w'}  - \left<v_w, v_{w'}\right> - s_w - s_{w'} - C \right),
$$

where $$f(x) = \text{min} \{x^{3/4}, 100 \} $$. Namely, substitute $$s_w = \lVert v_w \rVert _2$$. In the GloVe, they derive this expression from the assumption of linear structure and do a lot of trial and error to get this expression (you might have noticed that the reweighting function $$f$$ is a bit strange). Here, the expression naturally falls out of the generative model.


### Similarity to word2vec CBOW
Similarly, we can connect the derived objective to word2vec continuous bag of words (CBOW) model. Recall that the CBOW model gives the probability of a word, given past words as:

$$p(w_{k+1} | \{w_i\}_{i=1}^k) \propto \exp \left(\left< v_{w_{k+1}}, \frac{1}{k}\sum_{i=1}^k v_{w_i}\right> \right)$$

If we assume that over a window of $$k$$ words, our discourse vector $$c$$ has remained constant, then it can be shown that $$\frac{1}{k}\sum_{i=1}^k v_{w_i}$$ is the Maximum-a-Posteriori (MAP) estimate of $$c$$ up to some rescaling. Thus, this expression is related to our original model \eqref{eq:one}.

## On Analogy Solving

The authors also provide an explanation for the burning question that we introduced earlier: why are embeddings able to solve analogies? They call this phenomenon "RELATIONS = LINES". 

A quick summary of other works in this context:
* GloVe starts with the *assumption* of linear structure and tries to build an objective from it, which is different from the approach of this paper where they start with a generative model and show that such a linear structure exists.
* Levy and Goldberg show that word2vec skipgram negative sampling (SGNS) vectors satisfy this linear relationship, but their argument only holds for high-dimensional embeddings.

The gist of the argument is the follows: Say we have some relation $$R$$ (such as the man $$\to$$ king relationship). Then if we have two words $$a$$ and $$b$$ that satisfy the relation, we have:

$$v_a - v_b = \alpha_{a,b} \mu_r + \eta$$

Here $$\alpha_{a,b}$$ is some scalar that depends on $$a, b$$, $$\mu_r$$ is the direction vector that encodes the relationship, and $$\eta$$ is noise. 

If we have the assumption that word vectors behave like low-dimensional random vectors, we can cast analogy solving as a *linear regression* which reduces the effect of noise $$\eta$$.

<hr>

<a name="footnote1">1</a>. I would say that the word vectors look like a spherical gaussian (which indeed is an isotropic distribution), but readers would probably visualize something in $$\mathbb{R}^2$$. It turns out that a cool, nonintuitive property of isotropic distributions in **high dimensions** is that instead of being "clumped" around the origin, they are roughly distributed uniformly on the unit sphere (hand-waving ensues). 

[1]:https://arxiv.org/abs/1502.03520
[2]: https://dl.acm.org/citation.cfm?id=89095
[3]: https://arxiv.org/abs/1301.3781
[4]: https://nlp.stanford.edu/pubs/glove.pdf
[5]: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
[6]: https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization