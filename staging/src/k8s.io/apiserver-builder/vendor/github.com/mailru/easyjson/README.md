# easyjson [![Build Status](https://travis-ci.org/mailru/easyjson.svg?branch=master)](https://travis-ci.org/mailru/easyjson)

easyjson allows to (un-)marshal JSON golang structs without the use of reflection by generating marshaller code.  

One of the aims of the library is to keep generated code simple enough so that it can be easily optimized or fixed. Another goal is to provide users with ability to customize the generated code not available in 'encoding/json', such as generating snake_case names or enabling 'omitempty' behavior by default.

## usage
```
go get github.com/mailru/easyjson/...
easyjson -all <file>.go
```

This will generate `<file>_easyjson.go` with marshaller/unmarshaller methods for structs. `GOPATH` variable needs to be set up correctly, since the generation invokes a `go run` on a temporary file (this is a really convenient approach to code generation borrowed from https://github.com/pquerna/ffjson).

## options
```
Usage of .root/bin/easyjson:
  -all
        generate un-/marshallers for all structs in a file
  -build_tags string
        build tags to add to generated file
  -leave_temps
        do not delete temporary files
  -no_std_marshalers
        don't generate MarshalJSON/UnmarshalJSON methods
  -noformat
        do not run 'gofmt -w' on output file
  -omit_empty
        omit empty fields by default
  -snake_case
        use snake_case names instead of CamelCase by default
  -stubs
        only generate stubs for marshallers/unmarshallers methods
```

Using `-all` will generate (un-)marshallers for all structs in the file. By default, structs need to have a line beginning with `easyjson:json` in their docstring, e.g.:
```
//easyjson:json
struct A{}
```

`-snake_case` tells easyjson to generate snake\_case field names by default (unless explicitly overriden by a field tag). The CamelCase to snake\_case conversion algorithm should work in most cases (e.g. HTTPVersion will be converted to http_version). There can be names like JSONHTTPRPC where the conversion will return an unexpected result (jsonhttprpc without underscores),  but such names require a dictionary to do the conversion and may be ambiguous.

`-build_tags` will add corresponding build tag line for the generated file.
## marshaller/unmarshaller interfaces

easyjson generates MarshalJSON/UnmarshalJSON methods that are compatible with interfaces from 'encoding/json'. They are usable with 'json.Marshal' and 'json.Unmarshal' functions, however actually using those will result in significantly worse performance compared to custom interfaces.

`MarshalEasyJSON` / `UnmarshalEasyJSON` methods are generated for faster parsing using custom Lexer/Writer structs (`jlexer.Lexer`  and  `jwriter.Writer`). The method signature is defined in `easyjson.Marshaler` / `easyjson.Unmarshaler` interfaces. These interfaces allow to avoid using any unnecessary reflection or type assertions during parsing. Functions can be used manually or with `easyjson.Marshal<...>` and `easyjson.Unmarshal<...>` helper methods. 

`jwriter.Writer` struct in addition to function for returning the data as a single slice also has methods to return the size and to send the data to an `io.Writer`. This is aimed at a typical HTTP use-case, when you want to know the `Content-Length` before actually starting to send the data.

There are helpers in the top-level package for marhsaling/unmarshaling the data using custom interfaces to and from writers, including a helper for `http.ResponseWriter`.

## custom types
If `easyjson.Marshaler` / `easyjson.Unmarshaler` interfaces are implemented by a type involved in JSON parsing, the type will be marshaled/unmarshaled using these methods.  `easyjson.Optional` interface allows for a custom type to integrate with 'omitempty' logic. 

As an example, easyjson includes an `easyjson.RawMessage` analogous to `json.RawMessage`.

Also, there are 'optional' wrappers for primitive types in `easyjson/opt` package. These are useful in the case when it is necessary to distinguish between missing and default value for the type. Wrappers allow to avoid pointers and extra heap allocations in such cases.
 
## memory pooling

The library uses a custom buffer which allocates data in increasing chunks (128-32768 bytes). Chunks of 512 bytes and larger are reused with the help of `sync.Pool`. The maximum size of a chunk is bounded to reduce redundancy in memory allocation and to make the chunks more reusable in the case of large buffer sizes.

The buffer code is in `easyjson/buffer` package the exact values can be tweaked by a `buffer.Init()` call before the first serialization.

## limitations
* The library is at an early stage, there are likely to be some bugs and some features of 'encoding/json' may not be supported. Please report such cases, so that they may be fixed sooner.
* Object keys are case-sensitive (unlike encodin/json). Case-insentive behavior will be implemented as an option (case-insensitive matching is slower).
* Unsafe package is used by the code. While a non-unsafe version of easyjson can be made in the future, using unsafe package simplifies a lot of code by allowing no-copy []byte to string conversion within the library. This is used only during parsing and all the returned values are allocated properly.
* Floats are currently formatted with default precision for 'strconv' package. It is obvious that it is not always the correct way to handle it, but there aren't enough use-cases for floats at hand to do anything better.
* During parsing, parts of JSON that are skipped over are not syntactically validated more than required to skip matching parentheses.
* No true streaming support for encoding/decoding. For many use-cases and protocols, data length is typically known on input and needs to be known before sending the data.

## benchmarks
Most benchmarks were done using a sample 13kB JSON (9k if serialized back trimming the whitespace) from https://dev.twitter.com/rest/reference/get/search/tweets. The sample is very close to real-world data, quite structured and contains a variety of different types.

For small request benchmarks, an 80-byte portion of the regular sample was used.

For large request marshalling benchmarks, a struct containing 50 regular samples was used, making a ~500kB output JSON.

Benchmarks are available in the repository and are run on 'make'.

### easyjson vs. encoding/json

easyjson seems to be 5-6 times faster than the default json serialization for unmarshalling, 3-4 times faster for non-concurrent marshalling. Concurrent marshalling is 6-7x faster if marshalling to a writer.

### easyjson vs. ffjson

easyjson uses the same approach for code generation as ffjson, but a significantly different approach to lexing and generated code. This allows easyjson to be 2-3x faster for unmarshalling and 1.5-2x faster for non-concurrent unmarshalling. 

ffjson seems to behave weird if used concurrently: for large request pooling hurts performance instead of boosting it, it also does not quite scale well. These issues are likely to be fixable and until that comparisons might vary from version to version a lot.

easyjson is similar in performance for small requests and 2-5x times faster for large ones if used with a writer.

### easyjson vs. go/codec

github.com/ugorji/go/codec library provides compile-time helpers for JSON generation. In this case, helpers are not exactly marshallers as they are encoding-independent.

easyjson is generally ~2x faster for non-concurrent benchmarks and about 3x faster for concurrent encoding (without marshalling to a writer). Unsafe option for generated helpers was used.

As an attempt to measure marshalling performance of 'go/codec' (as opposed to allocations/memcpy/writer interface invocations), a benchmark was done with resetting lenght of a byte slice rather than resetting the whole slice to nil. However, the optimization in this exact form may not be applicable in practice, since the memory is not freed between marshalling operations.

### easyjson vs 'ujson' python module
ujson is using C code for parsing, so it is interesting to see how plain golang compares to that. It is imporant to note that the resulting object for python is slower to access, since the library parses JSON object into dictionaries.

easyjson seems to be slightly faster for unmarshalling (finally!) and 2-3x faster for marshalling.

### benchmark figures
The data was measured on 4 February, 2016 using current ffjson and golang 1.6. Data for go/codec was added on 4 March 2016, benchmarked on the same machine.

#### Unmarshalling
| lib    | json size | MB/s | allocs/op | B/op
|--------|-----------|------|-----------|-------
|standard| regular   | 22   | 218       | 10229
|standard| small     | 9.7  | 14        | 720
|--------|-----------|------|-----------|-------
|easyjson| regular   | 125  | 128       | 9794
|easyjson| small     | 67   | 3         | 128
|--------|-----------|------|-----------|-------
|ffjson  | regular   | 66   | 141       | 9985
|ffjson  | small     | 17.6 | 10        | 488
|--------|-----------|------|-----------|-------
|codec   | regular   | 55   | 434       | 19299
|codec   | small     | 29   | 7         | 336
|--------|-----------|------|-----------|-------
|ujson   | regular   | 103  | N/A       | N/A

#### Marshalling, one goroutine.
| lib      | json size | MB/s | allocs/op | B/op
|----------|-----------|------|-----------|-------
|standard  | regular   | 75   | 9         | 23256
|standard  | small     | 32   | 3         | 328
|standard  | large     | 80   | 17        | 1.2M
|----------|-----------|------|-----------|-------
|easyjson  | regular   | 213  | 9         | 10260
|easyjson* | regular   | 263  | 8         | 742
|easyjson  | small     | 125  | 1         | 128
|easyjson  | large     | 212  | 33        | 490k
|easyjson* | large     | 262  | 25        | 2879
|----------|-----------|------|-----------|-------
|ffjson    | regular   | 122  | 153       | 21340
|ffjson**  | regular   | 146  | 152       | 4897
|ffjson    | small     | 36   | 5         | 384
|ffjson**  | small     | 64   | 4         | 128
|ffjson    | large     | 134  | 7317      | 818k
|ffjson**  | large     | 125  | 7320      | 827k
|----------|-----------|------|-----------|-------
|codec     | regular   | 80   | 17        | 33601
|codec***  | regular   | 108  | 9         | 1153
|codec     | small     | 42   | 3         | 304
|codec***  | small     | 56   | 1         | 48
|codec     | large     | 73   | 483       | 2.5M
|codec***  | large     | 103  | 451       | 66007
|----------|-----------|------|-----------|-------
|ujson     | regular   | 92   | N/A       | N/A
\* marshalling to a writer,
\*\* using `ffjson.Pool()`,
\*\*\* reusing output slice instead of resetting it to nil

#### Marshalling, concurrent.
| lib      | json size | MB/s  | allocs/op | B/op
|----------|-----------|-------|-----------|-------
|standard  | regular   | 252   | 9         | 23257
|standard  | small     | 124   | 3         | 328
|standard  | large     | 289   | 17        | 1.2M
|----------|-----------|-------|-----------|-------
|easyjson  | regular   | 792   | 9         | 10597
|easyjson* | regular   | 1748  | 8         | 779
|easyjson  | small     | 333   | 1         | 128
|easyjson  | large     | 718   | 36        | 548k
|easyjson* | large     | 2134  | 25        | 4957
|----------|-----------|------|-----------|-------
|ffjson    | regular   | 301  | 153       | 21629
|ffjson**  | regular   | 707  | 152       | 5148
|ffjson    | small     | 62   | 5         | 384
|ffjson**  | small     | 282  | 4         | 128
|ffjson    | large     | 438  | 7330      | 1.0M
|ffjson**  | large     | 131  | 7319      | 820k
|----------|-----------|------|-----------|-------
|codec     | regular   | 183  | 17        | 33603
|codec***  | regular   | 671  | 9         | 1157
|codec     | small     | 147  | 3         | 304
|codec***  | small     | 299  | 1         | 48
|codec     | large     | 190  | 483       | 2.5M
|codec***  | large     | 752  | 451       | 77574
\* marshalling to a writer,
\*\* using `ffjson.Pool()`,
\*\*\* reusing output slice instead of resetting it to nil



