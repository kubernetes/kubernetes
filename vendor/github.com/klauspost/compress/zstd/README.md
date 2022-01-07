# zstd 

[Zstandard](https://facebook.github.io/zstd/) is a real-time compression algorithm, providing high compression ratios. 
It offers a very wide range of compression / speed trade-off, while being backed by a very fast decoder.
A high performance compression algorithm is implemented. For now focused on speed. 

This package provides [compression](#Compressor) to and [decompression](#Decompressor) of Zstandard content. 

This package is pure Go and without use of "unsafe". 

The `zstd` package is provided as open source software using a Go standard license.

Currently the package is heavily optimized for 64 bit processors and will be significantly slower on 32 bit processors.

## Installation

Install using `go get -u github.com/klauspost/compress`. The package is located in `github.com/klauspost/compress/zstd`.

[![Go Reference](https://pkg.go.dev/badge/github.com/klauspost/compress/zstd.svg)](https://pkg.go.dev/github.com/klauspost/compress/zstd)

## Compressor

### Status: 

STABLE - there may always be subtle bugs, a wide variety of content has been tested and the library is actively 
used by several projects. This library is being [fuzz-tested](https://github.com/klauspost/compress-fuzz) for all updates.

There may still be specific combinations of data types/size/settings that could lead to edge cases, 
so as always, testing is recommended.  

For now, a high speed (fastest) and medium-fast (default) compressor has been implemented. 

* The "Fastest" compression ratio is roughly equivalent to zstd level 1. 
* The "Default" compression ratio is roughly equivalent to zstd level 3 (default).
* The "Better" compression ratio is roughly equivalent to zstd level 7.
* The "Best" compression ratio is roughly equivalent to zstd level 11.

In terms of speed, it is typically 2x as fast as the stdlib deflate/gzip in its fastest mode. 
The compression ratio compared to stdlib is around level 3, but usually 3x as fast.

 
### Usage

An Encoder can be used for either compressing a stream via the
`io.WriteCloser` interface supported by the Encoder or as multiple independent
tasks via the `EncodeAll` function.
Smaller encodes are encouraged to use the EncodeAll function.
Use `NewWriter` to create a new instance that can be used for both.

To create a writer with default options, do like this:

```Go
// Compress input to output.
func Compress(in io.Reader, out io.Writer) error {
    enc, err := zstd.NewWriter(out)
    if err != nil {
        return err
    }
    _, err = io.Copy(enc, in)
    if err != nil {
        enc.Close()
        return err
    }
    return enc.Close()
}
```

Now you can encode by writing data to `enc`. The output will be finished writing when `Close()` is called.
Even if your encode fails, you should still call `Close()` to release any resources that may be held up.  

The above is fine for big encodes. However, whenever possible try to *reuse* the writer.

To reuse the encoder, you can use the `Reset(io.Writer)` function to change to another output. 
This will allow the encoder to reuse all resources and avoid wasteful allocations. 

Currently stream encoding has 'light' concurrency, meaning up to 2 goroutines can be working on part 
of a stream. This is independent of the `WithEncoderConcurrency(n)`, but that is likely to change 
in the future. So if you want to limit concurrency for future updates, specify the concurrency
you would like.

You can specify your desired compression level using `WithEncoderLevel()` option. Currently only pre-defined 
compression settings can be specified.

#### Future Compatibility Guarantees

This will be an evolving project. When using this package it is important to note that both the compression efficiency and speed may change.

The goal will be to keep the default efficiency at the default zstd (level 3). 
However the encoding should never be assumed to remain the same, 
and you should not use hashes of compressed output for similarity checks.

The Encoder can be assumed to produce the same output from the exact same code version.
However, the may be modes in the future that break this, 
although they will not be enabled without an explicit option.   

This encoder is not designed to (and will probably never) output the exact same bitstream as the reference encoder.

Also note, that the cgo decompressor currently does not [report all errors on invalid input](https://github.com/DataDog/zstd/issues/59),
[omits error checks](https://github.com/DataDog/zstd/issues/61), [ignores checksums](https://github.com/DataDog/zstd/issues/43) 
and seems to ignore concatenated streams, even though [it is part of the spec](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frames).

#### Blocks

For compressing small blocks, the returned encoder has a function called `EncodeAll(src, dst []byte) []byte`.

`EncodeAll` will encode all input in src and append it to dst.
This function can be called concurrently, but each call will only run on a single goroutine.

Encoded blocks can be concatenated and the result will be the combined input stream.
Data compressed with EncodeAll can be decoded with the Decoder, using either a stream or `DecodeAll`.

Especially when encoding blocks you should take special care to reuse the encoder. 
This will effectively make it run without allocations after a warmup period. 
To make it run completely without allocations, supply a destination buffer with space for all content.   

```Go
import "github.com/klauspost/compress/zstd"

// Create a writer that caches compressors.
// For this operation type we supply a nil Reader.
var encoder, _ = zstd.NewWriter(nil)

// Compress a buffer. 
// If you have a destination buffer, the allocation in the call can also be eliminated.
func Compress(src []byte) []byte {
    return encoder.EncodeAll(src, make([]byte, 0, len(src)))
} 
```

You can control the maximum number of concurrent encodes using the `WithEncoderConcurrency(n)` 
option when creating the writer.

Using the Encoder for both a stream and individual blocks concurrently is safe. 

### Performance

I have collected some speed examples to compare speed and compression against other compressors.

* `file` is the input file.
* `out` is the compressor used. `zskp` is this package. `zstd` is the Datadog cgo library. `gzstd/gzkp` is gzip standard and this library.
* `level` is the compression level used. For `zskp` level 1 is "fastest", level 2 is "default"; 3 is "better", 4 is "best".
* `insize`/`outsize` is the input/output size.
* `millis` is the number of milliseconds used for compression.
* `mb/s` is megabytes (2^20 bytes) per second.

```
Silesia Corpus:
http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip

This package:
file    out     level   insize      outsize     millis  mb/s
silesia.tar zskp    1   211947520   73101992    643     313.87
silesia.tar zskp    2   211947520   67504318    969     208.38
silesia.tar zskp    3   211947520   64595893    2007    100.68
silesia.tar zskp    4   211947520   60995370    8825    22.90

cgo zstd:
silesia.tar zstd    1   211947520   73605392    543     371.56
silesia.tar zstd    3   211947520   66793289    864     233.68
silesia.tar zstd    6   211947520   62916450    1913    105.66
silesia.tar zstd    9   211947520   60212393    5063    39.92

gzip, stdlib/this package:
silesia.tar gzstd   1   211947520   80007735    1654    122.21
silesia.tar gzkp    1   211947520   80136201    1152    175.45

GOB stream of binary data. Highly compressible.
https://files.klauspost.com/compress/gob-stream.7z

file        out     level   insize  outsize     millis  mb/s
gob-stream  zskp    1   1911399616  235022249   3088    590.30
gob-stream  zskp    2   1911399616  205669791   3786    481.34
gob-stream  zskp    3   1911399616  175034659   9636    189.17
gob-stream  zskp    4   1911399616  165609838   50369   36.19

gob-stream  zstd    1   1911399616  249810424   2637    691.26
gob-stream  zstd    3   1911399616  208192146   3490    522.31
gob-stream  zstd    6   1911399616  193632038   6687    272.56
gob-stream  zstd    9   1911399616  177620386   16175   112.70

gob-stream  gzstd   1   1911399616  357382641   10251   177.82
gob-stream  gzkp    1   1911399616  359753026   5438    335.20

The test data for the Large Text Compression Benchmark is the first
10^9 bytes of the English Wikipedia dump on Mar. 3, 2006.
http://mattmahoney.net/dc/textdata.html

file    out level   insize      outsize     millis  mb/s
enwik9  zskp    1   1000000000  343848582   3609    264.18
enwik9  zskp    2   1000000000  317276632   5746    165.97
enwik9  zskp    3   1000000000  292243069   12162   78.41
enwik9  zskp    4   1000000000  262183768   82837   11.51

enwik9  zstd    1   1000000000  358072021   3110    306.65
enwik9  zstd    3   1000000000  313734672   4784    199.35
enwik9  zstd    6   1000000000  295138875   10290   92.68
enwik9  zstd    9   1000000000  278348700   28549   33.40

enwik9  gzstd   1   1000000000  382578136   9604    99.30
enwik9  gzkp    1   1000000000  383825945   6544    145.73

Highly compressible JSON file.
https://files.klauspost.com/compress/github-june-2days-2019.json.zst

file                        out level   insize      outsize     millis  mb/s
github-june-2days-2019.json zskp    1   6273951764  699045015   10620   563.40
github-june-2days-2019.json zskp    2   6273951764  617881763   11687   511.96
github-june-2days-2019.json zskp    3   6273951764  524340691   34043   175.75
github-june-2days-2019.json zskp    4   6273951764  470320075   170190  35.16

github-june-2days-2019.json zstd    1   6273951764  766284037   8450    708.00
github-june-2days-2019.json zstd    3   6273951764  661889476   10927   547.57
github-june-2days-2019.json zstd    6   6273951764  642756859   22996   260.18
github-june-2days-2019.json zstd    9   6273951764  601974523   52413   114.16

github-june-2days-2019.json gzstd   1   6273951764  1164400847  29948   199.79
github-june-2days-2019.json gzkp    1   6273951764  1125417694  21788   274.61

VM Image, Linux mint with a few installed applications:
https://files.klauspost.com/compress/rawstudio-mint14.7z

file                    out level   insize      outsize     millis  mb/s
rawstudio-mint14.tar    zskp    1   8558382592  3667489370  20210   403.84
rawstudio-mint14.tar    zskp    2   8558382592  3364592300  31873   256.07
rawstudio-mint14.tar    zskp    3   8558382592  3158085214  77675   105.08
rawstudio-mint14.tar    zskp    4   8558382592  2965110639  857750  9.52

rawstudio-mint14.tar    zstd    1   8558382592  3609250104  17136   476.27
rawstudio-mint14.tar    zstd    3   8558382592  3341679997  29262   278.92
rawstudio-mint14.tar    zstd    6   8558382592  3235846406  77904   104.77
rawstudio-mint14.tar    zstd    9   8558382592  3160778861  140946  57.91

rawstudio-mint14.tar    gzstd   1   8558382592  3926257486  57722   141.40
rawstudio-mint14.tar    gzkp    1   8558382592  3962605659  45113   180.92

CSV data:
https://files.klauspost.com/compress/nyc-taxi-data-10M.csv.zst

file                    out level   insize      outsize     millis  mb/s
nyc-taxi-data-10M.csv   zskp    1   3325605752  641339945   8925    355.35
nyc-taxi-data-10M.csv   zskp    2   3325605752  591748091   11268   281.44
nyc-taxi-data-10M.csv   zskp    3   3325605752  530289687   25239   125.66
nyc-taxi-data-10M.csv   zskp    4   3325605752  476268884   135958  23.33

nyc-taxi-data-10M.csv   zstd    1   3325605752  687399637   8233    385.18
nyc-taxi-data-10M.csv   zstd    3   3325605752  598514411   10065   315.07
nyc-taxi-data-10M.csv   zstd    6   3325605752  570522953   20038   158.27
nyc-taxi-data-10M.csv   zstd    9   3325605752  517554797   64565   49.12

nyc-taxi-data-10M.csv   gzstd   1   3325605752  928656485   23876   132.83
nyc-taxi-data-10M.csv   gzkp    1   3325605752  922257165   16780   189.00
```

## Decompressor

Staus: STABLE - there may still be subtle bugs, but a wide variety of content has been tested.

This library is being continuously [fuzz-tested](https://github.com/klauspost/compress-fuzz),
kindly supplied by [fuzzit.dev](https://fuzzit.dev/). 
The main purpose of the fuzz testing is to ensure that it is not possible to crash the decoder, 
or run it past its limits with ANY input provided.  
 
### Usage

The package has been designed for two main usages, big streams of data and smaller in-memory buffers. 
There are two main usages of the package for these. Both of them are accessed by creating a `Decoder`.

For streaming use a simple setup could look like this:

```Go
import "github.com/klauspost/compress/zstd"

func Decompress(in io.Reader, out io.Writer) error {
    d, err := zstd.NewReader(in)
    if err != nil {
        return err
    }
    defer d.Close()
    
    // Copy content...
    _, err = io.Copy(out, d)
    return err
}
```

It is important to use the "Close" function when you no longer need the Reader to stop running goroutines. 
See "Allocation-less operation" below.

For decoding buffers, it could look something like this:

```Go
import "github.com/klauspost/compress/zstd"

// Create a reader that caches decompressors.
// For this operation type we supply a nil Reader.
var decoder, _ = zstd.NewReader(nil)

// Decompress a buffer. We don't supply a destination buffer,
// so it will be allocated by the decoder.
func Decompress(src []byte) ([]byte, error) {
    return decoder.DecodeAll(src, nil)
} 
```

Both of these cases should provide the functionality needed. 
The decoder can be used for *concurrent* decompression of multiple buffers. 
It will only allow a certain number of concurrent operations to run. 
To tweak that yourself use the `WithDecoderConcurrency(n)` option when creating the decoder.   

### Dictionaries

Data compressed with [dictionaries](https://github.com/facebook/zstd#the-case-for-small-data-compression) can be decompressed.

Dictionaries are added individually to Decoders.
Dictionaries are generated by the `zstd --train` command and contains an initial state for the decoder.
To add a dictionary use the `WithDecoderDicts(dicts ...[]byte)` option with the dictionary data.
Several dictionaries can be added at once.

The dictionary will be used automatically for the data that specifies them.
A re-used Decoder will still contain the dictionaries registered.

When registering multiple dictionaries with the same ID, the last one will be used.

It is possible to use dictionaries when compressing data.

To enable a dictionary use `WithEncoderDict(dict []byte)`. Here only one dictionary will be used 
and it will likely be used even if it doesn't improve compression. 

The used dictionary must be used to decompress the content.

For any real gains, the dictionary should be built with similar data. 
If an unsuitable dictionary is used the output may be slightly larger than using no dictionary.
Use the [zstd commandline tool](https://github.com/facebook/zstd/releases) to build a dictionary from sample data.
For information see [zstd dictionary information](https://github.com/facebook/zstd#the-case-for-small-data-compression). 

For now there is a fixed startup performance penalty for compressing content with dictionaries. 
This will likely be improved over time. Just be aware to test performance when implementing.  

### Allocation-less operation

The decoder has been designed to operate without allocations after a warmup. 

This means that you should *store* the decoder for best performance. 
To re-use a stream decoder, use the `Reset(r io.Reader) error` to switch to another stream.
A decoder can safely be re-used even if the previous stream failed.

To release the resources, you must call the `Close()` function on a decoder.
After this it can *no longer be reused*, but all running goroutines will be stopped.
So you *must* use this if you will no longer need the Reader.

For decompressing smaller buffers a single decoder can be used.
When decoding buffers, you can supply a destination slice with length 0 and your expected capacity.
In this case no unneeded allocations should be made. 

### Concurrency

The buffer decoder does everything on the same goroutine and does nothing concurrently.
It can however decode several buffers concurrently. Use `WithDecoderConcurrency(n)` to limit that.

The stream decoder operates on

* One goroutine reads input and splits the input to several block decoders.
* A number of decoders will decode blocks.
* A goroutine coordinates these blocks and sends history from one to the next.

So effectively this also means the decoder will "read ahead" and prepare data to always be available for output.

Since "blocks" are quite dependent on the output of the previous block stream decoding will only have limited concurrency.

In practice this means that concurrency is often limited to utilizing about 2 cores effectively.
 
 
### Benchmarks

These are some examples of performance compared to [datadog cgo library](https://github.com/DataDog/zstd).

The first two are streaming decodes and the last are smaller inputs. 
 
```
BenchmarkDecoderSilesia-8                          3     385000067 ns/op     550.51 MB/s        5498 B/op          8 allocs/op
BenchmarkDecoderSilesiaCgo-8                       6     197666567 ns/op    1072.25 MB/s      270672 B/op          8 allocs/op

BenchmarkDecoderEnwik9-8                           1    2027001600 ns/op     493.34 MB/s       10496 B/op         18 allocs/op
BenchmarkDecoderEnwik9Cgo-8                        2     979499200 ns/op    1020.93 MB/s      270672 B/op          8 allocs/op

Concurrent performance:

BenchmarkDecoder_DecodeAllParallel/kppkn.gtb.zst-16                28915         42469 ns/op    4340.07 MB/s         114 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/geo.protodata.zst-16           116505          9965 ns/op    11900.16 MB/s         16 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/plrabn12.txt.zst-16              8952        134272 ns/op    3588.70 MB/s         915 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/lcet10.txt.zst-16               11820        102538 ns/op    4161.90 MB/s         594 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/asyoulik.txt.zst-16             34782         34184 ns/op    3661.88 MB/s          60 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/alice29.txt.zst-16              27712         43447 ns/op    3500.58 MB/s          99 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/html_x_4.zst-16                 62826         18750 ns/op    21845.10 MB/s        104 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/paper-100k.pdf.zst-16          631545          1794 ns/op    57078.74 MB/s          2 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/fireworks.jpeg.zst-16         1690140           712 ns/op    172938.13 MB/s         1 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/urls.10K.zst-16                 10432        113593 ns/op    6180.73 MB/s        1143 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/html.zst-16                    113206         10671 ns/op    9596.27 MB/s          15 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallel/comp-data.bin.zst-16          1530615           779 ns/op    5229.49 MB/s           0 B/op          0 allocs/op

BenchmarkDecoder_DecodeAllParallelCgo/kppkn.gtb.zst-16             65217         16192 ns/op    11383.34 MB/s         46 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/geo.protodata.zst-16        292671          4039 ns/op    29363.19 MB/s          6 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/plrabn12.txt.zst-16          26314         46021 ns/op    10470.43 MB/s        293 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/lcet10.txt.zst-16            33897         34900 ns/op    12227.96 MB/s        205 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/asyoulik.txt.zst-16         104348         11433 ns/op    10949.01 MB/s         20 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/alice29.txt.zst-16           75949         15510 ns/op    9805.60 MB/s          32 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/html_x_4.zst-16             173910          6756 ns/op    60624.29 MB/s         37 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/paper-100k.pdf.zst-16       923076          1339 ns/op    76474.87 MB/s          1 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/fireworks.jpeg.zst-16       922920          1351 ns/op    91102.57 MB/s          2 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/urls.10K.zst-16              27649         43618 ns/op    16096.19 MB/s        407 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/html.zst-16                 279073          4160 ns/op    24614.18 MB/s          6 B/op          0 allocs/op
BenchmarkDecoder_DecodeAllParallelCgo/comp-data.bin.zst-16        749938          1579 ns/op    2581.71 MB/s           0 B/op          0 allocs/op
```

This reflects the performance around May 2020, but this may be out of date.

## Zstd inside ZIP files

It is possible to use zstandard to compress individual files inside zip archives.
While this isn't widely supported it can be useful for internal files.

To support the compression and decompression of these files you must register a compressor and decompressor.

It is highly recommended registering the (de)compressors on individual zip Reader/Writer and NOT
use the global registration functions. The main reason for this is that 2 registrations from 
different packages will result in a panic.

It is a good idea to only have a single compressor and decompressor, since they can be used for multiple zip
files concurrently, and using a single instance will allow reusing some resources.

See [this example](https://pkg.go.dev/github.com/klauspost/compress/zstd#example-ZipCompressor) for 
how to compress and decompress files inside zip archives.

# Contributions

Contributions are always welcome. 
For new features/fixes, remember to add tests and for performance enhancements include benchmarks.

For general feedback and experience reports, feel free to open an issue or write me on [Twitter](https://twitter.com/sh0dan).

This package includes the excellent [`github.com/cespare/xxhash`](https://github.com/cespare/xxhash) package Copyright (c) 2016 Caleb Spare.
