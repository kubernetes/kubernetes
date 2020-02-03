The Snappy compression format in the Go programming language.

To download and install from source:
$ go get github.com/golang/snappy

Unless otherwise noted, the Snappy-Go source files are distributed
under the BSD-style license found in the LICENSE file.



Benchmarks.

The golang/snappy benchmarks include compressing (Z) and decompressing (U) ten
or so files, the same set used by the C++ Snappy code (github.com/google/snappy
and note the "google", not "golang"). On an "Intel(R) Core(TM) i7-3770 CPU @
3.40GHz", Go's GOARCH=amd64 numbers as of 2016-05-29:

"go test -test.bench=."

_UFlat0-8         2.19GB/s ± 0%  html
_UFlat1-8         1.41GB/s ± 0%  urls
_UFlat2-8         23.5GB/s ± 2%  jpg
_UFlat3-8         1.91GB/s ± 0%  jpg_200
_UFlat4-8         14.0GB/s ± 1%  pdf
_UFlat5-8         1.97GB/s ± 0%  html4
_UFlat6-8          814MB/s ± 0%  txt1
_UFlat7-8          785MB/s ± 0%  txt2
_UFlat8-8          857MB/s ± 0%  txt3
_UFlat9-8          719MB/s ± 1%  txt4
_UFlat10-8        2.84GB/s ± 0%  pb
_UFlat11-8        1.05GB/s ± 0%  gaviota

_ZFlat0-8         1.04GB/s ± 0%  html
_ZFlat1-8          534MB/s ± 0%  urls
_ZFlat2-8         15.7GB/s ± 1%  jpg
_ZFlat3-8          740MB/s ± 3%  jpg_200
_ZFlat4-8         9.20GB/s ± 1%  pdf
_ZFlat5-8          991MB/s ± 0%  html4
_ZFlat6-8          379MB/s ± 0%  txt1
_ZFlat7-8          352MB/s ± 0%  txt2
_ZFlat8-8          396MB/s ± 1%  txt3
_ZFlat9-8          327MB/s ± 1%  txt4
_ZFlat10-8        1.33GB/s ± 1%  pb
_ZFlat11-8         605MB/s ± 1%  gaviota



"go test -test.bench=. -tags=noasm"

_UFlat0-8          621MB/s ± 2%  html
_UFlat1-8          494MB/s ± 1%  urls
_UFlat2-8         23.2GB/s ± 1%  jpg
_UFlat3-8         1.12GB/s ± 1%  jpg_200
_UFlat4-8         4.35GB/s ± 1%  pdf
_UFlat5-8          609MB/s ± 0%  html4
_UFlat6-8          296MB/s ± 0%  txt1
_UFlat7-8          288MB/s ± 0%  txt2
_UFlat8-8          309MB/s ± 1%  txt3
_UFlat9-8          280MB/s ± 1%  txt4
_UFlat10-8         753MB/s ± 0%  pb
_UFlat11-8         400MB/s ± 0%  gaviota

_ZFlat0-8          409MB/s ± 1%  html
_ZFlat1-8          250MB/s ± 1%  urls
_ZFlat2-8         12.3GB/s ± 1%  jpg
_ZFlat3-8          132MB/s ± 0%  jpg_200
_ZFlat4-8         2.92GB/s ± 0%  pdf
_ZFlat5-8          405MB/s ± 1%  html4
_ZFlat6-8          179MB/s ± 1%  txt1
_ZFlat7-8          170MB/s ± 1%  txt2
_ZFlat8-8          189MB/s ± 1%  txt3
_ZFlat9-8          164MB/s ± 1%  txt4
_ZFlat10-8         479MB/s ± 1%  pb
_ZFlat11-8         270MB/s ± 1%  gaviota



For comparison (Go's encoded output is byte-for-byte identical to C++'s), here
are the numbers from C++ Snappy's

make CXXFLAGS="-O2 -DNDEBUG -g" clean snappy_unittest.log && cat snappy_unittest.log

BM_UFlat/0     2.4GB/s  html
BM_UFlat/1     1.4GB/s  urls
BM_UFlat/2    21.8GB/s  jpg
BM_UFlat/3     1.5GB/s  jpg_200
BM_UFlat/4    13.3GB/s  pdf
BM_UFlat/5     2.1GB/s  html4
BM_UFlat/6     1.0GB/s  txt1
BM_UFlat/7   959.4MB/s  txt2
BM_UFlat/8     1.0GB/s  txt3
BM_UFlat/9   864.5MB/s  txt4
BM_UFlat/10    2.9GB/s  pb
BM_UFlat/11    1.2GB/s  gaviota

BM_ZFlat/0   944.3MB/s  html (22.31 %)
BM_ZFlat/1   501.6MB/s  urls (47.78 %)
BM_ZFlat/2    14.3GB/s  jpg (99.95 %)
BM_ZFlat/3   538.3MB/s  jpg_200 (73.00 %)
BM_ZFlat/4     8.3GB/s  pdf (83.30 %)
BM_ZFlat/5   903.5MB/s  html4 (22.52 %)
BM_ZFlat/6   336.0MB/s  txt1 (57.88 %)
BM_ZFlat/7   312.3MB/s  txt2 (61.91 %)
BM_ZFlat/8   353.1MB/s  txt3 (54.99 %)
BM_ZFlat/9   289.9MB/s  txt4 (66.26 %)
BM_ZFlat/10    1.2GB/s  pb (19.68 %)
BM_ZFlat/11  527.4MB/s  gaviota (37.72 %)
