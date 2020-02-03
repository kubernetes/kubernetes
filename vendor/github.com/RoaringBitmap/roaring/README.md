roaring [![Build Status](https://travis-ci.org/RoaringBitmap/roaring.png)](https://travis-ci.org/RoaringBitmap/roaring) [![Coverage Status](https://coveralls.io/repos/github/RoaringBitmap/roaring/badge.svg?branch=master)](https://coveralls.io/github/RoaringBitmap/roaring?branch=master) [![GoDoc](https://godoc.org/github.com/RoaringBitmap/roaring?status.svg)](https://godoc.org/github.com/RoaringBitmap/roaring) [![Go Report Card](https://goreportcard.com/badge/RoaringBitmap/roaring)](https://goreportcard.com/report/github.com/RoaringBitmap/roaring)
[![Build Status](https://cloud.drone.io/api/badges/RoaringBitmap/roaring/status.svg)](https://cloud.drone.io/RoaringBitmap/roaring)
=============

This is a go version of the Roaring bitmap data structure. 



Roaring bitmaps are used by several major systems such as [Apache Lucene][lucene] and derivative systems such as [Solr][solr] and
[Elasticsearch][elasticsearch], [Apache Druid (Incubating)][druid], [LinkedIn Pinot][pinot], [Netflix Atlas][atlas],  [Apache Spark][spark], [OpenSearchServer][opensearchserver], [Cloud Torrent][cloudtorrent], [Whoosh][whoosh],  [Pilosa][pilosa],  [Microsoft Visual Studio Team Services (VSTS)][vsts], and eBay's [Apache Kylin][kylin].

[lucene]: https://lucene.apache.org/
[solr]: https://lucene.apache.org/solr/
[elasticsearch]: https://www.elastic.co/products/elasticsearch
[druid]: https://druid.apache.org/
[spark]: https://spark.apache.org/
[opensearchserver]: http://www.opensearchserver.com
[cloudtorrent]: https://github.com/jpillora/cloud-torrent
[whoosh]: https://bitbucket.org/mchaput/whoosh/wiki/Home
[pilosa]: https://www.pilosa.com/
[kylin]: http://kylin.apache.org/
[pinot]: http://github.com/linkedin/pinot/wiki
[vsts]: https://www.visualstudio.com/team-services/
[atlas]: https://github.com/Netflix/atlas

Roaring bitmaps are found to work well in many important applications:

> Use Roaring for bitmap compression whenever possible. Do not use other bitmap compression methods ([Wang et al., SIGMOD 2017](http://db.ucsd.edu/wp-content/uploads/2017/03/sidm338-wangA.pdf))


The ``roaring`` Go library is used by
* [Cloud Torrent](https://github.com/jpillora/cloud-torrent): a self-hosted remote torrent client
* [runv](https://github.com/hyperhq/runv): an Hypervisor-based runtime for the Open Containers Initiative
* [InfluxDB](https://www.influxdata.com)
* [Pilosa](https://www.pilosa.com/)
* [Bleve](http://www.blevesearch.com)

This library is used in production in several systems, it is part of the [Awesome Go collection](https://awesome-go.com).


There are also  [Java](https://github.com/RoaringBitmap/RoaringBitmap) and [C/C++](https://github.com/RoaringBitmap/CRoaring) versions.  The Java, C, C++ and Go version are binary compatible: e.g,  you can save bitmaps
from a Java program and load them back in Go, and vice versa. We have a [format specification](https://github.com/RoaringBitmap/RoaringFormatSpec).


This code is licensed under Apache License, Version 2.0 (ASL2.0).

Copyright 2016-... by the authors.


### References

- Daniel Lemire, Owen Kaser, Nathan Kurz, Luca Deri, Chris O'Hara, François Saint-Jacques, Gregory Ssi-Yan-Kai, Roaring Bitmaps: Implementation of an Optimized Software Library, Software: Practice and Experience 48 (4), 2018 [arXiv:1709.07821](https://arxiv.org/abs/1709.07821)
-  Samy Chambi, Daniel Lemire, Owen Kaser, Robert Godin,
Better bitmap performance with Roaring bitmaps,
Software: Practice and Experience 46 (5), 2016.
http://arxiv.org/abs/1402.6407 This paper used data from http://lemire.me/data/realroaring2014.html
- Daniel Lemire, Gregory Ssi-Yan-Kai, Owen Kaser, Consistently faster and smaller compressed bitmaps with Roaring, Software: Practice and Experience 46 (11), 2016. http://arxiv.org/abs/1603.06549


### Dependencies

Dependencies are fetched automatically by giving the `-t` flag to `go get`.

they include
  - github.com/willf/bitset
  - github.com/mschoch/smat
  - github.com/glycerine/go-unsnap-stream
  - github.com/philhofer/fwd
  - github.com/jtolds/gls

Note that the smat library requires Go 1.6 or better.

#### Installation

  - go get -t github.com/RoaringBitmap/roaring


### Example

Here is a simplified but complete example:

```go
package main

import (
    "fmt"
    "github.com/RoaringBitmap/roaring"
    "bytes"
)


func main() {
    // example inspired by https://github.com/fzandona/goroar
    fmt.Println("==roaring==")
    rb1 := roaring.BitmapOf(1, 2, 3, 4, 5, 100, 1000)
    fmt.Println(rb1.String())

    rb2 := roaring.BitmapOf(3, 4, 1000)
    fmt.Println(rb2.String())

    rb3 := roaring.New()
    fmt.Println(rb3.String())

    fmt.Println("Cardinality: ", rb1.GetCardinality())

    fmt.Println("Contains 3? ", rb1.Contains(3))

    rb1.And(rb2)

    rb3.Add(1)
    rb3.Add(5)

    rb3.Or(rb1)

    // computes union of the three bitmaps in parallel using 4 workers  
    roaring.ParOr(4, rb1, rb2, rb3)
    // computes intersection of the three bitmaps in parallel using 4 workers  
    roaring.ParAnd(4, rb1, rb2, rb3)


    // prints 1, 3, 4, 5, 1000
    i := rb3.Iterator()
    for i.HasNext() {
        fmt.Println(i.Next())
    }
    fmt.Println()

    // next we include an example of serialization
    buf := new(bytes.Buffer)
    rb1.WriteTo(buf) // we omit error handling
    newrb:= roaring.New()
    newrb.ReadFrom(buf)
    if rb1.Equals(newrb) {
    	fmt.Println("I wrote the content to a byte stream and read it back.")
    }
    // you can iterate over bitmaps using ReverseIterator(), Iterator, ManyIterator()
}
```

If you wish to use serialization and handle errors, you might want to
consider the following sample of code:

```go
	rb := BitmapOf(1, 2, 3, 4, 5, 100, 1000)
	buf := new(bytes.Buffer)
	size,err:=rb.WriteTo(buf)
	if err != nil {
		t.Errorf("Failed writing")
	}
	newrb:= New()
	size,err=newrb.ReadFrom(buf)
	if err != nil {
		t.Errorf("Failed reading")
	}
	if ! rb.Equals(newrb) {
		t.Errorf("Cannot retrieve serialized version")
	}
```

Given N integers in [0,x), then the serialized size in bytes of
a Roaring bitmap should never exceed this bound:

`` 8 + 9 * ((long)x+65535)/65536 + 2 * N ``

That is, given a fixed overhead for the universe size (x), Roaring
bitmaps never use more than 2 bytes per integer. You can call
``BoundSerializedSizeInBytes`` for a more precise estimate.


### Documentation

Current documentation is available at http://godoc.org/github.com/RoaringBitmap/roaring

### Goroutine safety

In general, it should not generally be considered safe to access
the same bitmaps using different goroutines--they are left
unsynchronized for performance. Should you want to access
a Bitmap from more than one goroutine, you should
provide synchronization. Typically this is done by using channels to pass
the *Bitmap around (in Go style; so there is only ever one owner),
or by using `sync.Mutex` to serialize operations on Bitmaps.

### Coverage

We test our software. For a report on our test coverage, see

https://coveralls.io/github/RoaringBitmap/roaring?branch=master

### Benchmark

Type

         go test -bench Benchmark -run -
         
To run benchmarks on [Real Roaring Datasets](https://github.com/RoaringBitmap/real-roaring-datasets)
run the following:

```sh
go get github.com/RoaringBitmap/real-roaring-datasets
BENCH_REAL_DATA=1 go test -bench BenchmarkRealData -run -
```

### Iterative use

You can use roaring with gore:

- go get -u github.com/motemen/gore
- Make sure that ``$GOPATH/bin`` is in your ``$PATH``.
- go get github.com/RoaringBitmap/roaring

```go
$ gore
gore version 0.2.6  :help for help
gore> :import github.com/RoaringBitmap/roaring
gore> x:=roaring.New()
gore> x.Add(1)
gore> x.String()
"{1}"
```


### Fuzzy testing

You can help us test further the library with fuzzy testing:

         go get github.com/dvyukov/go-fuzz/go-fuzz
         go get github.com/dvyukov/go-fuzz/go-fuzz-build
         go test -tags=gofuzz -run=TestGenerateSmatCorpus
         go-fuzz-build github.com/RoaringBitmap/roaring
         go-fuzz -bin=./roaring-fuzz.zip -workdir=workdir/ -timeout=200

Let it run, and if the # of crashers is > 0, check out the reports in
the workdir where you should be able to find the panic goroutine stack
traces.

### Alternative in Go

There is a Go version wrapping the C/C++ implementation https://github.com/RoaringBitmap/gocroaring

For an alternative implementation in Go, see https://github.com/fzandona/goroar
The two versions were written independently.


### Mailing list/discussion group

https://groups.google.com/forum/#!forum/roaring-bitmaps
