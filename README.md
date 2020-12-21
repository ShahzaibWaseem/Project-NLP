# CORD-19 Dataset

In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 63,000 scholarly articles, including over 51,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.


Kaggle Link: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge


Dataset Link: https://www.semanticscholar.org/cord19/download


## Steps

 ### 1. Preprocessing Data & Parts of Speech Tagging
 

The Following are the Tags used from the Penn Parts of Speech Tags.


```python
filterTags = ['VBN', 'VBD', 'JJ', 'JJS', 'JJR', 'CD', 'NN', 'NNS', 'NNP', 'NNPS']
```


|Tag|Full Form|
|--|--|
|VBN|Verb, past participle|
|VBD|Verb, past tense|
|JJ|Adjective|
|JJS|Adjective, superlative|
|JJR|Adjective, comparative|
|CD|Cardinal number|
|NN|Noun, singular or mass|
|NNS|Noun, plural|
|NNP|Proper noun, singular|
|NNPS|Proper noun, plural|

### 2. TF-IDF
```py
vectorizer = TfidfVectorizer(stop_words='english')
```

### 3. Mini-Batch and K-Means
```python
lines = ["china", "covid-19", "Japan", "America", "sars"]

kmeans = MiniBatchKMeans(n_clusters=10, batch_size=100).fit(tfidf)
kmeans.predict(vectorizer.transform(lines))

>> array([8, 3, 8, 8, 3])
```

From this we can see that keywords like COVID and SARS appear in cluster number 3. Whereas, maybe differences between countries and their comparisons between cases are in the cluster number 8.

### 4. Analysis
#### Displaying Frequently occurring words in each individual Cluster
```python
Cluster 0:
		bag3
		ppxy
		ww
		mvp40
		evp40
		vp40
		gst
		mcherry
		myc
		tagged
--------------------------------------
Cluster 1:
		patients
		hospital
		respiratory
		clinical
		year
		10
		pcr
		age
		positive
		children
--------------------------------------
Cluster 2:
		defi
		swine
		vigi
		ukcvn
		swinelineage
		maximalist
		derogations
		ventions
		tubercu
		tb
--------------------------------------
Cluster 3:
		covid
		2020
		cov
		sars
		2019
		19
		doi
		10
		health
		coronavirus
--------------------------------------
Cluster 4:
		μ1
		dea
		transfusion
		platelet
		cryoprecipitate
		transfused
		homol
		ogous
		μιη
		adrenergic
--------------------------------------
Cluster 5:
		schistosoma
		neuroschistosomiasis
		eieren
		bij
		een
		het
		nederlandse
		patiënte
		infectie
		cerebrale
--------------------------------------
Cluster 6:
		health
		based
		public
		risk
		care
		world
		pandemic
		national
		disease
		social
--------------------------------------
Cluster 7:
		por
		los
		las
		el
		una
		este
		en
		como
		la
		es
--------------------------------------
Cluster 8:
		anti
		ml
		cell
		virus
		protein
		10
		specific
		based
		cells
		non
--------------------------------------
Cluster 9:
		il
		ifn
		mediated
		induced
		cell
		anti
		dependent
		specific
		virus
		like
--------------------------------------
```

By just looking at it the cluster 5 seems to be filled with non-English Languages

-   patiënte: French
-   cerebrale: Italian
-   infectie: Dutch

And Cluster 7 looks to be filled with Spanish.

### 5. Visualization
#### Dimensionality Reduction for Plotting

```python
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42).fit(tfidf)
```

#### Plotting
##### Clustering

![Cluster](https://github.com/ShahzaibWaseem/Project-NLP/blob/master/Outputs/Plot%28cluster%29.png?raw=true)

##### t-distributed Stochastic Neighbor Embedding

t-SNE is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.

![t-SNE](https://github.com/ShahzaibWaseem/Project-NLP/blob/master/Outputs/Plot%28t-SNE%29.png?raw=true)

### 6. Searching
```python
stringToSearch=input("String to Search: ").lower()

def dictionarySearcher(data, stringToSearch):
	matchingDocuments={}
	
	for documentID in data:
		if stringToSearch in data[documentID]:
			matchingDocuments[documentID]=data[documentID]
			return matchingDocuments

matchingDocuments = dictionarySearcher(data, stringToSearch)
matchingDocumentsTitles={}
for documentID in matchingDocuments:
    file=open(os.path.join("..", "Data", "document_parses", "pdf_json", documentID + ".json"), "r")
    jsonFileData = json.load(file)
    file.close()
    matchingDocumentsTitles[documentID]=jsonFileData["metadata"]["title"]

print("The keyword \"" + stringToSearch + "\" appears in", str(len(matchingDocumentsTitles)), "Documents, and the titles are\n")
for i, documentID in enumerate(matchingDocumentsTitles):
    if matchingDocumentsTitles[documentID] == "":
        print(i, "<This Document has no Title>")
    else:
        print(i, matchingDocumentsTitles[documentID])
```

#### Output

```python
>> String to Search: pakistan
>> The keyword "pakistan" appears in 1 Documents, and the titles are 0 Felis margarita (Carnivora: Felidae)
```
The whole Article gets printed after we tell it which document to print

Title: Felis margarita (Carnivora: Felidae)

Abstract

Felis margarita Loche, 1858 is a felid commonly called the sand cat. It is 1 of 6 species in the genus Felis. One of the smallest of the wild cats, Felis margarita, is adapted behaviorally and morphologically to live in desert environments. Prey includes rodents, birds, reptiles, and arthropods. This species has a wide, but disjunct distribution through northern Africa, the Arabian Peninsula, and southwest and central Asia. F. margarita occurs at low densities throughout its range and is listed as "Near Threatened" by the International Union for Conservation of Nature and Natural Resources due to habitat degradation and its low and potentially declining population. F. margarita is bred in zoos in North America and Europe.

Body

F. m. harrisoni Hemmer, Grubb and Groves, 1976:301 . Type locality "northern edge of Umm as Samim, Oman, 21º 55' N, 55º 50' E." F. m. margarita Loche, 1858 :50. See above. F. m. scheffeli Hemmer, 1974a . Type locality "Nushki-Wűste, Westpakistan." F. m. thinobia (Ognev, 1927:356) . See above.
...