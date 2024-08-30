# MeSH Embeddings

This repository contains code & pre-trained representations for the **Medical Subject Headings** (MeSH) thesaurus. These representations were trained using the **node2vec** algorithm with default parameters. For more details about node2vec please visit [this repository](https://github.com/aditya-grover/node2vec).

> **Note**: node2vec relies on an edge list to learn node representations. The edge list for MeSH can be constructed using the tree numbers from the xml file (`descYYYY.xml`) which is available [here](ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/). In order to enable anybody to train their own MeSH representations, the edge list for `desc2020.xml` is shared along with the pre-trained representations.

The code below shows how to use the vectors in practice. Two files are needed

- the pre-trained vectors: `mesh_embeddings.txt.gz`
- the mapping between MeSH Unique Identifiers and vector ids: `mesh_ui_to_id.pickle`

```python
import gzip
import pickle

with open('mesh_ui_to_id.pickle', 'rb') as stream:
    mesh_ui_to_id = pickle.load(stream)

embeddings = {}
with gzip.open('mesh_embeddings.txt.gz', 'rt') as stream:
    n_embeddings, embedding_dim = stream.readline().strip().split()
    for line in stream:
        splitline = str(line).strip().split()
        idx = int(splitline[0])
        vector = list(map(float, splitline[1:]))
        embeddings[idx] = vector

def get_embedding_from_mesh_ui(mesh_ui):
    return embeddings[mesh_ui_to_id[mesh_ui]]

print(f'There are {n_embeddings} MeSH embeddings')
print(f'Each embedding is {embedding_dim}-dimensional')

print('MeSH UI for <Headache> is: D006261')
print('MeSH embedding for <Headache> is:', get_embedding_from_mesh_ui('D006261'))
```

```text
>>> There are 29638 MeSH embeddings
>>> Each embedding is 256-dimensional
>>> MeSH UI for <Headache> is: D006261
>>> MeSH embedding for <Headache> is: [-0.22856602, -0.32223737, -0.16364807, ...]
```
