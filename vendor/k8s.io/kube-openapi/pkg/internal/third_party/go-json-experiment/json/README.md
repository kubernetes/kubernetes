# JSON Serialization (v2)

[![GoDev](https://img.shields.io/static/v1?label=godev&message=reference&color=00add8)](https://pkg.go.dev/github.com/go-json-experiment/json)
[![Build Status](https://github.com/go-json-experiment/json/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/go-json-experiment/json/actions)

This module hosts an experimental implementation of v2 `encoding/json`.
The API is unstable and breaking changes will regularly be made.
Do not depend on this in publicly available modules.

## Goals and objectives

* **Mostly backwards compatible:** If possible, v2 should aim to be _mostly_
compatible with v1 in terms of both API and default behavior to ease migration.
For example, the `Marshal` and `Unmarshal` functions are the most widely used
declarations in the v1 package. It seems sensible for equivalent functionality
in v2 to be named the same and have the same signature.
Behaviorally, we should aim for 95% to 99% backwards compatibility.
We do not aim for 100% compatibility since we want the freedom to break
certain behaviors that are now considered to have been a mistake.
We may provide options that can bring the v2 implementation to 100% compatibility,
but it will not be the default.

* **More flexible:** There is a
[long list of feature requests](https://github.com/golang/go/issues?q=is%3Aissue+is%3Aopen+encoding%2Fjson+in%3Atitle).
We should aim to provide the most flexible features that addresses most usages.
We do not want to over fit the v2 API to handle every possible use case.
Ideally, the features provided should be orthogonal in nature such that
any combination of features results in as few surprising edge cases as possible.

* **More performant:** JSON serialization is widely used and any bit of extra
performance gains will be greatly appreciated. Some rarely used behaviors of v1
may be dropped in favor of better performance. For example,
despite `Encoder` and `Decoder` operating on an `io.Writer` and `io.Reader`,
they do not operate in a truly streaming manner,
leading to a loss in performance. The v2 implementation should aim to be truly
streaming by default (see [#33714](https://golang.org/issue/33714)).

* **Easy to use (hard to misuse):** The v2 API should aim to make
the common case easy and the less common case at least possible.
The API should avoid behavior that goes contrary to user expectation,
which may result in subtle bugs (see [#36225](https://golang.org/issue/36225)).

* **v1 and v2 maintainability:** Since the v1 implementation must stay forever,
it would be beneficial if v1 could be implemented under the hood with v2,
allowing for less maintenance burden in the future. This probably implies that
behavioral changes in v2 relative to v1 need to be exposed as options.

* **Avoid unsafe:** Standard library packages generally avoid the use of
package `unsafe` even if it could provide a performance boost.
We aim to preserve this property.

## Expectations

While this module aims to possibly be the v2 implementation of `encoding/json`,
there is no guarantee that this outcome will occur. As with any major change
to the Go standard library, this will eventually go through the
[Go proposal process](https://github.com/golang/proposal#readme).
At the present moment, this is still in the design and experimentation phase
and is not ready for a formal proposal.

There are several possible outcomes from this experiment:
1. We determine that a v2 `encoding/json` would not provide sufficient benefit
over the existing v1 `encoding/json` package. Thus, we abandon this effort.
2. We propose a v2 `encoding/json` design, but it is rejected in favor of some
other design that is considered superior.
3. We propose a v2 `encoding/json` design, but rather than adding an entirely
new v2 `encoding/json` package, we decide to merge its functionality into
the existing v1 `encoding/json` package.
4. We propose a v2 `encoding/json` design and it is accepted, resulting in
its addition to the standard library.
5. Some other unforeseen outcome (among the infinite number of possibilities).

## Development

This module is primarily developed by
[@dsnet](https://github.com/dsnet),
[@mvdan](https://github.com/mvdan), and
[@johanbrandhorst](https://github.com/johanbrandhorst)
with feedback provided by
[@rogpeppe](https://github.com/rogpeppe),
[@ChrisHines](https://github.com/ChrisHines), and
[@rsc](https://github.com/rsc).

Discussion about semantics occur semi-regularly, where a
[record of past meetings can be found here](https://docs.google.com/document/d/1rovrOTd-wTawGMPPlPuKhwXaYBg9VszTXR9AQQL5LfI/edit?usp=sharing).

## Design overview

This package aims to provide a clean separation between syntax and semantics.
Syntax deals with the structural representation of JSON (as specified in
[RFC 4627](https://tools.ietf.org/html/rfc4627),
[RFC 7159](https://tools.ietf.org/html/rfc7159),
[RFC 7493](https://tools.ietf.org/html/rfc7493),
[RFC 8259](https://tools.ietf.org/html/rfc8259), and
[RFC 8785](https://tools.ietf.org/html/rfc8785)).
Semantics deals with the meaning of syntactic data as usable application data.

The `Encoder` and `Decoder` types are streaming tokenizers concerned with the
packing or parsing of JSON data. They operate on `Token` and `RawValue` types
which represent the common data structures that are representable in JSON.
`Encoder` and `Decoder` do not aim to provide any interpretation of the data.

Functions like `Marshal`, `MarshalFull`, `MarshalNext`, `Unmarshal`,
`UnmarshalFull`, and `UnmarshalNext` provide semantic meaning by correlating
any arbitrary Go type with some JSON representation of that type (as stored in
data types like `[]byte`, `io.Writer`, `io.Reader`, `Encoder`, or `Decoder`).

![API overview](api.png)

This diagram provides a high-level overview of the v2 `json` package.
Purple blocks represent types, while blue blocks represent functions or methods.
The arrows and their direction represent the approximate flow of data.
The bottom half of the diagram contains functionality that is only concerned
with syntax, while the upper half contains functionality that assigns
semantic meaning to syntactic data handled by the bottom half.

In contrast to v1 `encoding/json`, options are represented as separate types
rather than being setter methods on the `Encoder` or `Decoder` types.

## Behavior changes

The v2 `json` package changes the default behavior of `Marshal` and `Unmarshal`
relative to the v1 `json` package to be more sensible.
Some of these behavior changes have options and workarounds to opt into
behavior similar to what v1 provided.

This table shows an overview of the changes:

| v1 | v2 | Details |
| -- | -- | ------- |
| JSON object members are unmarshaled into a Go struct using a **case-insensitive name match**. | JSON object members are unmarshaled into a Go struct using a **case-sensitive name match**. | [CaseSensitivity](/diff_test.go#:~:text=TestCaseSensitivity) |
| When marshaling a Go struct, a struct field marked as `omitempty` is omitted if **the field value is an empty Go value**, which is defined as false, 0, a nil pointer, a nil interface value, and any empty array, slice, map, or string. | When marshaling a Go struct, a struct field marked as `omitempty` is omitted if **the field value would encode as an empty JSON value**, which is defined as a JSON null, or an empty JSON string, object, or array. | [OmitEmptyOption](/diff_test.go#:~:text=TestOmitEmptyOption) |
| The `string` option **does affect** Go bools. | The `string` option **does not affect** Go bools. | [StringOption](/diff_test.go#:~:text=TestStringOption) |
| The `string` option **does not recursively affect** sub-values of the Go field value. | The `string` option **does recursively affect** sub-values of the Go field value. | [StringOption](/diff_test.go#:~:text=TestStringOption) |
| The `string` option **sometimes accepts** a JSON null escaped within a JSON string. | The `string` option **never accepts** a JSON null escaped within a JSON string. | [StringOption](/diff_test.go#:~:text=TestStringOption) |
| A nil Go slice is marshaled as a **JSON null**. | A nil Go slice is marshaled as an **empty JSON array**. | [NilSlicesAndMaps](/diff_test.go#:~:text=TestNilSlicesAndMaps) |
| A nil Go map is marshaled as a **JSON null**. | A nil Go map is marshaled as an **empty JSON object**. | [NilSlicesAndMaps](/diff_test.go#:~:text=TestNilSlicesAndMaps) |
| A Go array may be unmarshaled from a **JSON array of any length**. | A Go array must be unmarshaled from a **JSON array of the same length**. | [Arrays](/diff_test.go#:~:text=Arrays) |
| A Go byte array is represented as a **JSON array of JSON numbers**. | A Go byte array is represented as a **Base64-encoded JSON string**. | [ByteArrays](/diff_test.go#:~:text=TestByteArrays) |
| `MarshalJSON` and `UnmarshalJSON` methods declared on a pointer receiver are **inconsistently called**. | `MarshalJSON` and `UnmarshalJSON` methods declared on a pointer receiver are **consistently called**. | [PointerReceiver](/diff_test.go#:~:text=TestPointerReceiver) |
| A Go map is marshaled in a **deterministic order**. | A Go map is marshaled in a **non-deterministic order**. | [MapDeterminism](/diff_test.go#:~:text=TestMapDeterminism) |
| JSON strings are encoded **with HTML-specific characters being escaped**. | JSON strings are encoded **without any characters being escaped** (unless necessary). | [EscapeHTML](/diff_test.go#:~:text=TestEscapeHTML) |
| When marshaling, invalid UTF-8 within a Go string **are silently replaced**. | When marshaling, invalid UTF-8 within a Go string **results in an error**. | [InvalidUTF8](/diff_test.go#:~:text=TestInvalidUTF8) |
| When unmarshaling, invalid UTF-8 within a JSON string **are silently replaced**. | When unmarshaling, invalid UTF-8 within a JSON string **results in an error**. | [InvalidUTF8](/diff_test.go#:~:text=TestInvalidUTF8) |
| When marshaling, **an error does not occur** if the output JSON value contains objects with duplicate names. | When marshaling, **an error does occur** if the output JSON value contains objects with duplicate names. | [DuplicateNames](/diff_test.go#:~:text=TestDuplicateNames) |
| When unmarshaling, **an error does not occur** if the input JSON value contains objects with duplicate names. | When unmarshaling, **an error does occur** if the input JSON value contains objects with duplicate names. | [DuplicateNames](/diff_test.go#:~:text=TestDuplicateNames) |
| Unmarshaling a JSON null into a non-empty Go value **inconsistently clears the value or does nothing**. | Unmarshaling a JSON null into a non-empty Go value **always clears the value**. | [MergeNull](/diff_test.go#:~:text=TestMergeNull) |
| Unmarshaling a JSON value into a non-empty Go value **follows inconsistent and bizarre behavior**. | Unmarshaling a JSON value into a non-empty Go value **always merges if the input is an object, and otherwise replaces**.  | [MergeComposite](/diff_test.go#:~:text=TestMergeComposite) |
| A `time.Duration` is represented as a **JSON number containing the decimal number of nanoseconds**. | A `time.Duration` is represented as a **JSON string containing the formatted duration (e.g., "1h2m3.456s")**. | [TimeDurations](/diff_test.go#:~:text=TestTimeDurations) |
| Unmarshaling a JSON number into a Go float beyond its representation **results in an error**. | Unmarshaling a JSON number into a Go float beyond its representation **uses the closest representable value (e.g., ±`math.MaxFloat`)**. | [MaxFloats](/diff_test.go#:~:text=TestMaxFloats) |
| A Go struct with only unexported fields **can be serialized**. | A Go struct with only unexported fields **cannot be serialized**. | [EmptyStructs](/diff_test.go#:~:text=TestEmptyStructs) |
| A Go struct that embeds an unexported struct type **can sometimes be serialized**. | A Go struct that embeds an unexported struct type **cannot be serialized**. | [EmbedUnexported](/diff_test.go#:~:text=TestEmbedUnexported) |

See [diff_test.go](/diff_test.go) for details about every change.

## Performance

One of the goals of the v2 module is to be more performant than v1.

Each of the charts below show the performance across
several different JSON implementations:

* `JSONv1` is `encoding/json` at `v1.18.2`
* `JSONv2` is `github.com/go-json-experiment/json` at `v0.0.0-20220524042235-dd8be80fc4a7`
* `JSONIterator` is `github.com/json-iterator/go` at `v1.1.12`
* `SegmentJSON` is `github.com/segmentio/encoding/json` at `v0.3.5`
* `GoJSON` is `github.com/goccy/go-json` at `v0.9.7`
* `SonicJSON` is `github.com/bytedance/sonic` at `v1.3.0`

Benchmarks were run across various datasets:

* `CanadaGeometry` is a GeoJSON (RFC 7946) representation of Canada.
  It contains many JSON arrays of arrays of two-element arrays of numbers.
* `CITMCatalog` contains many JSON objects using numeric names.
* `SyntheaFHIR` is sample JSON data from the healthcare industry.
  It contains many nested JSON objects with mostly string values,
  where the set of unique string values is relatively small.
* `TwitterStatus` is the JSON response from the Twitter API.
  It contains a mix of all different JSON kinds, where string values
  are a mix of both single-byte ASCII and multi-byte Unicode.
* `GolangSource` is a simple tree representing the Go source code.
  It contains many nested JSON objects, each with the same schema.
* `StringUnicode` contains many strings with multi-byte Unicode runes.

All of the implementations other than `JSONv1` and `JSONv2` make
extensive use of `unsafe`. As such, we expect those to generally be faster,
but at the cost of memory and type safety. `SonicJSON` goes a step even further
and uses just-in-time compilation to generate machine code specialized
for the Go type being marshaled or unmarshaled.
Also, `SonicJSON` does not validate JSON strings for valid UTF-8,
and so gains a notable performance boost on datasets with multi-byte Unicode.
Benchmarks are performed based on the default marshal and unmarshal behavior
of each package. Note that `JSONv2` aims to be safe and correct by default,
which may not be the most performant strategy.

`JSONv2` has several semantic changes relative to `JSONv1` that
impacts performance:

1.  When marshaling, `JSONv2` no longer sorts the keys of a Go map.
    This will improve performance.
2.  When marshaling or unmarshaling, `JSONv2` always checks
    to make sure JSON object names are unique.
    This will hurt performance, but is more correct.
3.  When marshaling or unmarshaling, `JSONv2` always
    shallow copies the underlying value for a Go interface and
    shallow copies the key and value for entries in a Go map.
    This is done to keep the value as addressable so that `JSONv2` can
    call methods and functions that operate on a pointer receiver.
    This will hurt performance, but is more correct.

All of the charts are unit-less since the values are normalized
relative to `JSONv1`, which is why `JSONv1` always has a value of 1.
A lower value is better (i.e., runs faster).

Benchmarks were performed on an AMD Ryzen 9 5900X.

The code for the benchmarks is located at
https://github.com/go-json-experiment/jsonbench.

### Marshal Performance

#### Concrete types

![Benchmark Marshal Concrete](benchmark-marshal-concrete.png)

* This compares marshal performance when serializing
  [from concrete types](/testdata_test.go).
* The `JSONv1` implementation is close to optimal (without the use of `unsafe`).
* Relative to `JSONv1`, `JSONv2` is generally as fast or slightly faster.
* Relative to `JSONIterator`, `JSONv2` is up to 1.3x faster.
* Relative to `SegmentJSON`, `JSONv2` is up to 1.8x slower.
* Relative to `GoJSON`, `JSONv2` is up to 2.0x slower.
* Relative to `SonicJSON`, `JSONv2` is about 1.8x to 3.2x slower
  (ignoring `StringUnicode` since `SonicJSON` does not validate UTF-8).
* For `JSONv1` and `JSONv2`, marshaling from concrete types is
  mostly limited by the performance of Go reflection.

#### Interface types

![Benchmark Marshal Interface](benchmark-marshal-interface.png)

* This compares marshal performance when serializing from
  `any`, `map[string]any`, and `[]any` types.
* Relative to `JSONv1`, `JSONv2` is about 1.5x to 4.2x faster.
* Relative to `JSONIterator`, `JSONv2` is about 1.1x to 2.4x faster.
* Relative to `SegmentJSON`, `JSONv2` is about 1.2x to 1.8x faster.
* Relative to `GoJSON`, `JSONv2` is about 1.1x to 2.5x faster.
* Relative to `SonicJSON`, `JSONv2` is up to 1.5x slower
  (ignoring `StringUnicode` since `SonicJSON` does not validate UTF-8).
* `JSONv2` is faster than the alternatives.
  One advantange is because it does not sort the keys for a `map[string]any`,
  while alternatives (except `SonicJSON` and `JSONIterator`) do sort the keys.

#### RawValue types

![Benchmark Marshal Rawvalue](benchmark-marshal-rawvalue.png)

* This compares performance when marshaling from a `json.RawValue`.
  This mostly exercises the underlying encoder and
  hides the cost of Go reflection.
* Relative to `JSONv1`, `JSONv2` is about 3.5x to 7.8x faster.
* `JSONIterator` is blazingly fast because
  [it does not validate whether the raw value is valid](https://go.dev/play/p/bun9IXQCKRe)
  and simply copies it to the output.
* Relative to `SegmentJSON`, `JSONv2` is about 1.5x to 2.7x faster.
* Relative to `GoJSON`, `JSONv2` is up to 2.2x faster.
* Relative to `SonicJSON`, `JSONv2` is up to 1.5x faster.
* Aside from `JSONIterator`, `JSONv2` is generally the fastest.

### Unmarshal Performance

#### Concrete types

![Benchmark Unmarshal Concrete](benchmark-unmarshal-concrete.png)

* This compares unmarshal performance when deserializing
  [into concrete types](/testdata_test.go).
* Relative to `JSONv1`, `JSONv2` is about 1.8x to 5.7x faster.
* Relative to `JSONIterator`, `JSONv2` is about 1.1x to 1.6x slower.
* Relative to `SegmentJSON`, `JSONv2` is up to 2.5x slower.
* Relative to `GoJSON`, `JSONv2` is about 1.4x to 2.1x slower.
* Relative to `SonicJSON`, `JSONv2` is up to 4.0x slower
  (ignoring `StringUnicode` since `SonicJSON` does not validate UTF-8).
* For `JSONv1` and `JSONv2`, unmarshaling into concrete types is
  mostly limited by the performance of Go reflection.

#### Interface types

![Benchmark Unmarshal Interface](benchmark-unmarshal-interface.png)

* This compares unmarshal performance when deserializing into
  `any`, `map[string]any`, and `[]any` types.
* Relative to `JSONv1`, `JSONv2` is about 1.tx to 4.3x faster.
* Relative to `JSONIterator`, `JSONv2` is up to 1.5x faster.
* Relative to `SegmentJSON`, `JSONv2` is about 1.5 to 3.7x faster.
* Relative to `GoJSON`, `JSONv2` is up to 1.3x faster.
* Relative to `SonicJSON`, `JSONv2` is up to 1.5x slower
  (ignoring `StringUnicode` since `SonicJSON` does not validate UTF-8).
* Aside from `SonicJSON`, `JSONv2` is generally just as fast
  or faster than all the alternatives.

#### RawValue types

![Benchmark Unmarshal Rawvalue](benchmark-unmarshal-rawvalue.png)

* This compares performance when unmarshaling into a `json.RawValue`.
  This mostly exercises the underlying decoder and
  hides away most of the cost of Go reflection.
* Relative to `JSONv1`, `JSONv2` is about 8.3x to 17.0x faster.
* Relative to `JSONIterator`, `JSONv2` is up to 2.0x faster.
* Relative to `SegmentJSON`, `JSONv2` is up to 1.6x faster or 1.7x slower.
* Relative to `GoJSON`, `JSONv2` is up to 1.9x faster or 2.1x slower.
* Relative to `SonicJSON`, `JSONv2` is up to 2.0x faster
  (ignoring `StringUnicode` since `SonicJSON` does not validate UTF-8).
* `JSONv1` takes a
  [lexical scanning approach](https://talks.golang.org/2011/lex.slide#1),
  which performs a virtual function call for every byte of input.
  In contrast, `JSONv2` makes heavy use of iterative and linear parsing logic
  (with extra complexity to resume parsing when encountering segmented buffers).
* `JSONv2` is comparable to the alternatives that use `unsafe`.
  Generally it is faster, but sometimes it is slower.
