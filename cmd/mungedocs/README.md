# Documentation Mungers

Basically this is like lint/gofmt for md docs.

It basically does the following:
- iterate over all files in the given doc root.
- for each file split it into a slice (mungeLines) of lines (mungeLine)
- a mungeline has metadata about each line typically determined by a 'fast' regex.
  - metadata contains things like 'is inside a preformatted block'
  - contains a markdown header
  - has a link to another file
  - etc..
  - if you have a really slow regex with a lot of backtracking you might want to write a fast one to limit how often you run the slow one.
- each munger is then called in turn
  - they are given the mungeLines
  - they create an entirely new set of mungeLines with their modifications
  - the new set is returned
- the new set is then fed into the next munger.
- in the end we might commit the end mungeLines to the file or not (--verify)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cmd/mungedocs/README.md?pixel)]()
