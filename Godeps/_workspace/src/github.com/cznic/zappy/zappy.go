// Copyright 2014 The zappy Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the SNAPPY-GO-LICENSE file.

/*
Package zappy implements the zappy block-based compression format.  It aims for
a combination of good speed and reasonable compression.

Zappy is a format incompatible, API compatible fork of snappy-go[1]. The C++
snappy implementation is at [2].

Reasons for the fork

The snappy compression is pretty good. Yet it has one problem built into its
format definition[3] - the maximum length of a copy "instruction" is 64 bytes.
For some specific usage patterns with long runs of repeated data, it turns out
the compression is suboptimal. For example a 1:1000 "sparseness" 64kB bit index
with only few set bits is compressed to about 3kB (about 1000 of 64B copy, 3
byte "instructions").

Format description

Zappy uses much less complicated format than snappy. Each encoded block begins
with the uvarint-encoded[4] length of the decoded data, followed by a sequence
of chunks. Chunks begin and end on byte boundaries. The chunk starts with a
varint encoded number N:

	N >= 0: N+1 literal bytes follow.

	N < 0: copy -N bytes, starting at offset M (in the following uvarint).

Performance issues

Compression rate is roughly the same as of snappy for the reference data set:

                  testdata/html: snappy   23320, zappy   22943, 0.984, orig  102400
              testdata/urls.10K: snappy  334437, zappy  355163, 1.062, orig  702087
             testdata/house.jpg: snappy  126711, zappy  126694, 1.000, orig  126958
  testdata/mapreduce-osdi-1.pdf: snappy   77227, zappy   77646, 1.005, orig   94330
              testdata/html_x_4: snappy   92350, zappy   22956, 0.249, orig  409600
               testdata/cp.html: snappy   11938, zappy   12961, 1.086, orig   24603
              testdata/fields.c: snappy    4825, zappy    5395, 1.118, orig   11150
           testdata/grammar.lsp: snappy    1814, zappy    1933, 1.066, orig    3721
           testdata/kennedy.xls: snappy  423518, zappy  440597, 1.040, orig 1029744
           testdata/alice29.txt: snappy   89550, zappy  104016, 1.162, orig  152089
          testdata/asyoulik.txt: snappy   79583, zappy   91345, 1.148, orig  125179
            testdata/lcet10.txt: snappy  238761, zappy  275488, 1.154, orig  426754
          testdata/plrabn12.txt: snappy  324567, zappy  376885, 1.161, orig  481861
                  testdata/ptt5: snappy   96350, zappy   91465, 0.949, orig  513216
                   testdata/sum: snappy   18927, zappy   20015, 1.057, orig   38240
               testdata/xargs.1: snappy    2532, zappy    2793, 1.103, orig    4227
         testdata/geo.protodata: snappy   23362, zappy   20759, 0.889, orig  118588
             testdata/kppkn.gtb: snappy   73962, zappy   87200, 1.179, orig  184320
                          TOTAL: snappy 2043734, zappy 2136254, 1.045, orig 4549067

Zappy has better RLE handling (1/1000+1 non zero bytes in each index):

     Sparse bit index      16 B: snappy       9, zappy       9, 1.000
     Sparse bit index      32 B: snappy      10, zappy      10, 1.000
     Sparse bit index      64 B: snappy      11, zappy      10, 0.909
     Sparse bit index     128 B: snappy      16, zappy      14, 0.875
     Sparse bit index     256 B: snappy      22, zappy      14, 0.636
     Sparse bit index     512 B: snappy      36, zappy      16, 0.444
     Sparse bit index    1024 B: snappy      57, zappy      18, 0.316
     Sparse bit index    2048 B: snappy     111, zappy      32, 0.288
     Sparse bit index    4096 B: snappy     210, zappy      31, 0.148
     Sparse bit index    8192 B: snappy     419, zappy      75, 0.179
     Sparse bit index   16384 B: snappy     821, zappy     138, 0.168
     Sparse bit index   32768 B: snappy    1627, zappy     232, 0.143
     Sparse bit index   65536 B: snappy    3243, zappy     451, 0.139

When compiled with CGO_ENABLED=1, zappy is now faster than snappy-go.
Old=snappy-go, new=zappy:

 benchmark                   old MB/s     new MB/s  speedup
 BenchmarkWordsDecode1e3       148.98       189.04    1.27x
 BenchmarkWordsDecode1e4       150.29       182.51    1.21x
 BenchmarkWordsDecode1e5       145.79       182.95    1.25x
 BenchmarkWordsDecode1e6       167.43       187.69    1.12x
 BenchmarkWordsEncode1e3        47.11       145.69    3.09x
 BenchmarkWordsEncode1e4        81.47       136.50    1.68x
 BenchmarkWordsEncode1e5        78.86       127.93    1.62x
 BenchmarkWordsEncode1e6        96.81       142.95    1.48x
 Benchmark_UFlat0              316.87       463.19    1.46x
 Benchmark_UFlat1              231.56       350.32    1.51x
 Benchmark_UFlat2             3656.68      8258.39    2.26x
 Benchmark_UFlat3              892.56      1270.09    1.42x
 Benchmark_UFlat4              315.84       959.08    3.04x
 Benchmark_UFlat5              211.70       301.55    1.42x
 Benchmark_UFlat6              211.59       258.29    1.22x
 Benchmark_UFlat7              209.80       272.21    1.30x
 Benchmark_UFlat8              254.59       301.70    1.19x
 Benchmark_UFlat9              163.39       192.66    1.18x
 Benchmark_UFlat10             155.46       189.70    1.22x
 Benchmark_UFlat11             170.11       198.95    1.17x
 Benchmark_UFlat12             148.32       178.78    1.21x
 Benchmark_UFlat13             359.25       579.99    1.61x
 Benchmark_UFlat14             197.27       291.33    1.48x
 Benchmark_UFlat15             185.75       248.07    1.34x
 Benchmark_UFlat16             362.74       582.66    1.61x
 Benchmark_UFlat17             222.95       240.01    1.08x
 Benchmark_ZFlat0              188.66       311.89    1.65x
 Benchmark_ZFlat1              101.46       201.34    1.98x
 Benchmark_ZFlat2               93.62       244.50    2.61x
 Benchmark_ZFlat3              102.79       243.34    2.37x
 Benchmark_ZFlat4              191.64       625.32    3.26x
 Benchmark_ZFlat5              103.09       169.39    1.64x
 Benchmark_ZFlat6              110.35       182.57    1.65x
 Benchmark_ZFlat7               89.56       190.53    2.13x
 Benchmark_ZFlat8              154.05       235.68    1.53x
 Benchmark_ZFlat9               87.58       133.51    1.52x
 Benchmark_ZFlat10              82.08       127.51    1.55x
 Benchmark_ZFlat11              91.36       138.91    1.52x
 Benchmark_ZFlat12              79.24       123.02    1.55x
 Benchmark_ZFlat13             217.04       374.26    1.72x
 Benchmark_ZFlat14             100.33       168.03    1.67x
 Benchmark_ZFlat15              80.79       160.46    1.99x
 Benchmark_ZFlat16             213.32       375.79    1.76x
 Benchmark_ZFlat17             135.37       197.13    1.46x

The package builds with CGO_ENABLED=0 as well, but the performance is worse.


 $ CGO_ENABLED=0 go test -test.run=NONE -test.bench=. > old.benchcmp
 $ CGO_ENABLED=1 go test -test.run=NONE -test.bench=. > new.benchcmp
 $ benchcmp old.benchcmp new.benchcmp
 benchmark                  old ns/op    new ns/op    delta
 BenchmarkWordsDecode1e3         9735         5288  -45.68%
 BenchmarkWordsDecode1e4       100229        55369  -44.76%
 BenchmarkWordsDecode1e5      1037611       546420  -47.34%
 BenchmarkWordsDecode1e6      9559352      5335307  -44.19%
 BenchmarkWordsEncode1e3        16206         6629  -59.10%
 BenchmarkWordsEncode1e4       140283        73161  -47.85%
 BenchmarkWordsEncode1e5      1476657       781756  -47.06%
 BenchmarkWordsEncode1e6     12702229      6997656  -44.91%
 Benchmark_UFlat0              397307       221198  -44.33%
 Benchmark_UFlat1             3890483      2008341  -48.38%
 Benchmark_UFlat2               35810        15398  -57.00%
 Benchmark_UFlat3              140850        74194  -47.32%
 Benchmark_UFlat4              814575       426783  -47.61%
 Benchmark_UFlat5              156995        81473  -48.10%
 Benchmark_UFlat6               77645        43161  -44.41%
 Benchmark_UFlat7               25415        13579  -46.57%
 Benchmark_UFlat8             6372440      3412916  -46.44%
 Benchmark_UFlat9             1453679       789956  -45.66%
 Benchmark_UFlat10            1243146       660747  -46.85%
 Benchmark_UFlat11            3903493      2146334  -45.02%
 Benchmark_UFlat12            5106250      2696144  -47.20%
 Benchmark_UFlat13            1641394       884969  -46.08%
 Benchmark_UFlat14             262206       131174  -49.97%
 Benchmark_UFlat15              32325        17047  -47.26%
 Benchmark_UFlat16             366991       204877  -44.17%
 Benchmark_UFlat17            1343988       770907  -42.64%
 Benchmark_ZFlat0              579954       329812  -43.13%
 Benchmark_ZFlat1             6564692      3504867  -46.61%
 Benchmark_ZFlat2              902029       513700  -43.05%
 Benchmark_ZFlat3              678722       384312  -43.38%
 Benchmark_ZFlat4             1197389       654361  -45.35%
 Benchmark_ZFlat5              262677       144939  -44.82%
 Benchmark_ZFlat6              111249        60876  -45.28%
 Benchmark_ZFlat7               39024        19420  -50.24%
 Benchmark_ZFlat8             8046106      4387928  -45.47%
 Benchmark_ZFlat9             2043167      1143139  -44.05%
 Benchmark_ZFlat10            1781604       980528  -44.96%
 Benchmark_ZFlat11            5478647      3078585  -43.81%
 Benchmark_ZFlat12            7245995      3929863  -45.77%
 Benchmark_ZFlat13            2432529      1371606  -43.61%
 Benchmark_ZFlat14             420315       227494  -45.88%
 Benchmark_ZFlat15              52378        26564  -49.28%
 Benchmark_ZFlat16             567047       316196  -44.24%
 Benchmark_ZFlat17            1630820       937310  -42.53%

 benchmark                   old MB/s     new MB/s  speedup
 BenchmarkWordsDecode1e3       102.71       189.08    1.84x
 BenchmarkWordsDecode1e4        99.77       180.60    1.81x
 BenchmarkWordsDecode1e5        96.38       183.01    1.90x
 BenchmarkWordsDecode1e6       104.61       187.43    1.79x
 BenchmarkWordsEncode1e3        61.70       150.85    2.44x
 BenchmarkWordsEncode1e4        71.28       136.68    1.92x
 BenchmarkWordsEncode1e5        67.72       127.92    1.89x
 BenchmarkWordsEncode1e6        78.73       142.90    1.82x
 Benchmark_UFlat0              257.73       462.93    1.80x
 Benchmark_UFlat1              180.46       349.59    1.94x
 Benchmark_UFlat2             3545.30      8244.61    2.33x
 Benchmark_UFlat3              669.72      1271.39    1.90x
 Benchmark_UFlat4              502.84       959.74    1.91x
 Benchmark_UFlat5              156.71       301.98    1.93x
 Benchmark_UFlat6              143.60       258.33    1.80x
 Benchmark_UFlat7              146.41       274.01    1.87x
 Benchmark_UFlat8              161.59       301.72    1.87x
 Benchmark_UFlat9              104.62       192.53    1.84x
 Benchmark_UFlat10             100.70       189.45    1.88x
 Benchmark_UFlat11             109.33       198.83    1.82x
 Benchmark_UFlat12              94.37       178.72    1.89x
 Benchmark_UFlat13             312.67       579.93    1.85x
 Benchmark_UFlat14             145.84       291.52    2.00x
 Benchmark_UFlat15             130.77       247.95    1.90x
 Benchmark_UFlat16             323.14       578.82    1.79x
 Benchmark_UFlat17             137.14       239.09    1.74x
 Benchmark_ZFlat0              176.57       310.48    1.76x
 Benchmark_ZFlat1              106.95       200.32    1.87x
 Benchmark_ZFlat2              140.75       247.14    1.76x
 Benchmark_ZFlat3              138.98       245.45    1.77x
 Benchmark_ZFlat4              342.08       625.95    1.83x
 Benchmark_ZFlat5               93.66       169.75    1.81x
 Benchmark_ZFlat6              100.23       183.16    1.83x
 Benchmark_ZFlat7               95.35       191.60    2.01x
 Benchmark_ZFlat8              127.98       234.68    1.83x
 Benchmark_ZFlat9               74.44       133.04    1.79x
 Benchmark_ZFlat10              70.26       127.66    1.82x
 Benchmark_ZFlat11              77.89       138.62    1.78x
 Benchmark_ZFlat12              66.50       122.62    1.84x
 Benchmark_ZFlat13             210.98       374.17    1.77x
 Benchmark_ZFlat14              90.98       168.09    1.85x
 Benchmark_ZFlat15              80.70       159.12    1.97x
 Benchmark_ZFlat16             209.13       375.04    1.79x
 Benchmark_ZFlat17             113.02       196.65    1.74x
 $

Build tags

If a constraint 'purego' appears in the build constraints [5] then a pure Go
version is built regardless of the $CGO_ENABLED value.

	$ touch zappy.go ; go install -tags purego github.com/cznic/zappy # for example

Information sources

... referenced from the above documentation.

 [1]: http://code.google.com/p/snappy-go/
 [2]: http://code.google.com/p/snappy/
 [3]: http://code.google.com/p/snappy/source/browse/trunk/format_description.txt
 [4]: http://golang.org/pkg/encoding/binary/
 [5]: http://golang.org/pkg/go/build/#hdr-Build_Constraints
*/
package zappy
