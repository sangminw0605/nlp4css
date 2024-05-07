import numpy
from wmd import WMD

embeddings = numpy.array([[0.1, 1], [1, 0.1]], dtype=numpy.float32)
nbow = {"first":  ("#1", [0, 1], numpy.array([1.5, 0.5], dtype=numpy.float32)),
        "second": ("#2", [0, 1], numpy.array([0.75, 0.15], dtype=numpy.float32))}
calc = WMD(embeddings, nbow, vocabulary_min=2)
print(calc.nearest_neighbors("first"))