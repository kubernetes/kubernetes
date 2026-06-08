# JSON Serialization (v2)

[![GoDev](https://img.shields.io/static/v1?label=godev&message=reference&color=00add8)](https://pkg.go.dev/github.com/go-json-experiment/json)
[![Build Status](https://github.com/go-json-experiment/json/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/go-json-experiment/json/actions)

This module hosts an experimental implementation of v2 `encoding/json`.
The API is unstable and breaking changes will regularly be made.
Do not depend on this in publicly available modules.

Any commits that make breaking API or behavior changes will be marked
with the string "WARNING: " near the top of the commit message.
It is your responsibility to inspect the list of commit changes
when upgrading the module. Not all breaking changes will lead to build failures.

A [proposal to include this module in Go as `encoding/json/v2` and `encoding/json/jsontext`](https://github.com/golang/go/issues/71497) has been started on the Go Github project on 2025-01-30. Please provide your feedback there.

## Goals and objectives

* **Mostly backwards compatible:** If possible, v2 should aim to be _mostly_
compatible with v1 in terms of both API and default behavior to ease migration.
For example, the `Marshal` and `Unmarshal` functions are the most widely used
declarations in the v1 package. It seems sensible for equivalent functionality
in v2 to be named the same and have a mostly compatible signature.
Behaviorally, we should aim for 95% to 99% backwards compatibility.
We do not aim for 100% compatibility since we want the freedom to break
certain behaviors that are now considered to have been a mistake.
Options exist that can bring the v2 implementation to 100% compatibility,
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
packing or parsing of JSON data. They operate on `Token` and `Value` types
which represent the common data structures that are representable in JSON.
`Encoder` and `Decoder` do not aim to provide any interpretation of the data.

Functions like `Marshal`, `MarshalWrite`, `MarshalEncode`, `Unmarshal`,
`UnmarshalRead`, and `UnmarshalDecode` provide semantic meaning by correlating
any arbitrary Go type with some JSON representation of that type (as stored in
data types like `[]byte`, `io.Writer`, `io.Reader`, `Encoder`, or `Decoder`).

![API overview](api.png)

This diagram provides a high-level overview of the v2 `json` and `jsontext` packages.
Purple blocks represent types, while blue blocks represent functions or methods.
The arrows and their direction represent the approximate flow of data.
The bottom half of the diagram contains functionality that is only concerned
with syntax (implemented by the `jsontext` package),
while the upper half contains functionality that assigns
semantic meaning to syntactic data handled by the bottom half
(as implemented by the v2 `json` package).

In contrast to v1 `encoding/json`, options are represented as separate types
rather than being setter methods on the `Encoder` or `Decoder` types.
Some options affects JSON serialization at the syntactic layer,
while others affect it at the semantic layer.
Some options only affect JSON when decoding,
while others affect JSON while encoding.

## Behavior changes

The v2 `json` package changes the default behavior of `Marshal` and `Unmarshal`
relative to the v1 `json` package to be more sensible.
Some of these behavior changes have options and workarounds to opt into
behavior similar to what v1 provided.

This table shows an overview of the changes:

| v1 | v2 | Details |
| -- | -- | ------- |
| JSON object members are unmarshaled into a Go struct using a **case-insensitive name match**. | JSON object members are unmarshaled into a Go struct using a **case-sensitive name match**. | [CaseSensitivity](/v1/diff_test.go#:~:text=TestCaseSensitivity) |
| When marshaling a Go struct, a struct field marked as `omitempty` is omitted if **the field value is an empty Go value**, which is defined as false, 0, a nil pointer, a nil interface value, and any empty array, slice, map, or string. | When marshaling a Go struct, a struct field marked as `omitempty` is omitted if **the field value would encode as an empty JSON value**, which is defined as a JSON null, or an empty JSON string, object, or array. | [OmitEmptyOption](/v1/diff_test.go#:~:text=TestOmitEmptyOption) |
| The `string` option **does affect** Go strings and bools. | The `string` option **does not affect** Go strings or bools. | [StringOption](/v1/diff_test.go#:~:text=TestStringOption) |
| The `string` option **does not recursively affect** sub-values of the Go field value. | The `string` option **does recursively affect** sub-values of the Go field value. | [StringOption](/v1/diff_test.go#:~:text=TestStringOption) |
| The `string` option **sometimes accepts** a JSON null escaped within a JSON string. | The `string` option **never accepts** a JSON null escaped within a JSON string. | [StringOption](/v1/diff_test.go#:~:text=TestStringOption) |
| A nil Go slice is marshaled as a **JSON null**. | A nil Go slice is marshaled as an **empty JSON array**. | [NilSlicesAndMaps](/v1/diff_test.go#:~:text=TestNilSlicesAndMaps) |
| A nil Go map is marshaled as a **JSON null**. | A nil Go map is marshaled as an **empty JSON object**. | [NilSlicesAndMaps](/v1/diff_test.go#:~:text=TestNilSlicesAndMaps) |
| A Go array may be unmarshaled from a **JSON array of any length**. | A Go array must be unmarshaled from a **JSON array of the same length**. | [Arrays](/v1/diff_test.go#:~:text=Arrays) |
| A Go byte array is represented as a **JSON array of JSON numbers**. | A Go byte array is represented as a **Base64-encoded JSON string**. | [ByteArrays](/v1/diff_test.go#:~:text=TestByteArrays) |
| `MarshalJSON` and `UnmarshalJSON` methods declared on a pointer receiver are **inconsistently called**. | `MarshalJSON` and `UnmarshalJSON` methods declared on a pointer receiver are **consistently called**. | [PointerReceiver](/v1/diff_test.go#:~:text=TestPointerReceiver) |
| A Go map is marshaled in a **deterministic order**. | A Go map is marshaled in a **non-deterministic order**. | [MapDeterminism](/v1/diff_test.go#:~:text=TestMapDeterminism) |
| JSON strings are encoded **with HTML-specific characters being escaped**. | JSON strings are encoded **without any characters being escaped** (unless necessary). | [EscapeHTML](/v1/diff_test.go#:~:text=TestEscapeHTML) |
| When marshaling, invalid UTF-8 within a Go string **are silently replaced**. | When marshaling, invalid UTF-8 within a Go string **results in an error**. | [InvalidUTF8](/v1/diff_test.go#:~:text=TestInvalidUTF8) |
| When unmarshaling, invalid UTF-8 within a JSON string **are silently replaced**. | When unmarshaling, invalid UTF-8 within a JSON string **results in an error**. | [InvalidUTF8](/v1/diff_test.go#:~:text=TestInvalidUTF8) |
| When marshaling, **an error does not occur** if the output JSON value contains objects with duplicate names. | When marshaling, **an error does occur** if the output JSON value contains objects with duplicate names. | [DuplicateNames](/v1/diff_test.go#:~:text=TestDuplicateNames) |
| When unmarshaling, **an error does not occur** if the input JSON value contains objects with duplicate names. | When unmarshaling, **an error does occur** if the input JSON value contains objects with duplicate names. | [DuplicateNames](/v1/diff_test.go#:~:text=TestDuplicateNames) |
| Unmarshaling a JSON null into a non-empty Go value **inconsistently clears the value or does nothing**. | Unmarshaling a JSON null into a non-empty Go value **always clears the value**. | [MergeNull](/v1/diff_test.go#:~:text=TestMergeNull) |
| Unmarshaling a JSON value into a non-empty Go value **follows inconsistent and bizarre behavior**. | Unmarshaling a JSON value into a non-empty Go value **always merges if the input is an object, and otherwise replaces**.  | [MergeComposite](/v1/diff_test.go#:~:text=TestMergeComposite) |
| A `time.Duration` is represented as a **JSON number containing the decimal number of nanoseconds**. | A `time.Duration` has no default representation in v2 (see [#71631](https://golang.org/issue/71631)) and results in an error. | |
| A Go struct with only unexported fields **can be serialized**. | A Go struct with only unexported fields **cannot be serialized**. | [EmptyStructs](/v1/diff_test.go#:~:text=TestEmptyStructs) |

See [diff_test.go](/v1/diff_test.go) for details about every change.

## Performance

One of the goals of the v2 module is to be more performant than v1,
but not at the expense of correctness.
In general, v2 is at performance parity with v1 for marshaling,
but dramatically faster for unmarshaling.

See https://github.com/go-json-experiment/jsonbench for benchmarks
comparing v2 with v1 and a number of other popular JSON implementations.
