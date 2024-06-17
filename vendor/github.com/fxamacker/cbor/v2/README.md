# CBOR Codec in Go

<!-- [![](https://github.com/fxamacker/images/raw/master/cbor/v2.5.0/fxamacker_cbor_banner.png)](#cbor-library-in-go) -->

[fxamacker/cbor](https://github.com/fxamacker/cbor) is a library for encoding and decoding [CBOR](https://www.rfc-editor.org/info/std94) and [CBOR Sequences](https://www.rfc-editor.org/rfc/rfc8742.html).

CBOR is a [trusted alternative](https://www.rfc-editor.org/rfc/rfc8949.html#name-comparison-of-other-binary-) to JSON, MessagePack, Protocol Buffers, etc.&nbsp; CBOR is an Internet&nbsp;Standard defined by [IETF&nbsp;STD&nbsp;94 (RFC&nbsp;8949)](https://www.rfc-editor.org/info/std94) and is designed to be relevant for decades.

`fxamacker/cbor` is used in projects by Arm Ltd., Cisco, Dapper Labs, EdgeX&nbsp;Foundry, Fraunhofer&#8209;AISEC, Let's&nbsp;Encrypt (ISRG), Linux&nbsp;Foundation, Microsoft, Mozilla, Oasis&nbsp;Protocol, Tailscale, Teleport, [and&nbsp;others](https://github.com/fxamacker/cbor#who-uses-fxamackercbor).

See [Quick&nbsp;Start](#quick-start) and [Releases](https://github.com/fxamacker/cbor/releases/).  🆕 `UnmarshalFirst` and `DiagnoseFirst` can decode CBOR Sequences.

## fxamacker/cbor

[![](https://github.com/fxamacker/cbor/workflows/ci/badge.svg)](https://github.com/fxamacker/cbor/actions?query=workflow%3Aci)
[![](https://github.com/fxamacker/cbor/workflows/cover%20%E2%89%A596%25/badge.svg)](https://github.com/fxamacker/cbor/actions?query=workflow%3A%22cover+%E2%89%A596%25%22)
[![CodeQL](https://github.com/fxamacker/cbor/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fxamacker/cbor/actions/workflows/codeql-analysis.yml)
[![](https://img.shields.io/badge/fuzzing-passing-44c010)](#fuzzing-and-code-coverage)
[![Go Report Card](https://goreportcard.com/badge/github.com/fxamacker/cbor)](https://goreportcard.com/report/github.com/fxamacker/cbor)

`fxamacker/cbor` is a CBOR codec in full conformance with [IETF STD&nbsp;94 (RFC&nbsp;8949)](https://www.rfc-editor.org/info/std94). It also supports CBOR Sequences ([RFC&nbsp;8742](https://www.rfc-editor.org/rfc/rfc8742.html)) and Extended Diagnostic Notation ([Appendix G of RFC&nbsp;8610](https://www.rfc-editor.org/rfc/rfc8610.html#appendix-G)).

Features include full support for CBOR tags, [Core Deterministic Encoding](https://www.rfc-editor.org/rfc/rfc8949.html#name-core-deterministic-encoding), duplicate map key detection, etc.

Design balances trade-offs between security, speed, concurrency, encoded data size, usability, etc.

<details><summary>Highlights</summary><p/>

__🚀&nbsp; Speed__

Encoding and decoding is fast without using Go's `unsafe` package.  Slower settings are opt-in.  Default limits allow very fast and memory efficient rejection of malformed CBOR data.

__🔒&nbsp; Security__

Decoder has configurable limits that defend against malicious inputs.  Duplicate map key detection is supported.  By contrast, `encoding/gob` is [not designed to be hardened against adversarial inputs](https://pkg.go.dev/encoding/gob#hdr-Security).

Codec passed multiple confidential security assessments in 2022.  No vulnerabilities found in subset of codec in a [nonconfidential security assessment](https://github.com/veraison/go-cose/blob/v1.0.0-rc.1/reports/NCC_Microsoft-go-cose-Report_2022-05-26_v1.0.pdf) prepared by NCC&nbsp;Group for Microsoft&nbsp;Corporation.

__🗜️&nbsp; Data Size__

Struct tags (`toarray`, `keyasint`, `omitempty`) automatically reduce size of encoded structs. Encoding optionally shrinks float64→32→16 when values fit.

__:jigsaw:&nbsp; Usability__

API is mostly same as `encoding/json` plus interfaces that simplify concurrency for CBOR options.  Encoding and decoding modes can be created at startup and reused by any goroutines.

Presets include Core Deterministic Encoding, Preferred Serialization, CTAP2 Canonical CBOR, etc.

__📆&nbsp;  Extensibility__

Features include CBOR [extension points](https://www.rfc-editor.org/rfc/rfc8949.html#section-7.1) (e.g. CBOR tags) and extensive settings.  API has interfaces that allow users to create custom encoding and decoding without modifying this library.

<hr/>

</details>

### Secure Decoding with Configurable Settings

`fxamacker/cbor` has configurable limits, etc. that defend against malicious CBOR data.

By contrast, `encoding/gob` is [not designed to be hardened against adversarial inputs](https://pkg.go.dev/encoding/gob#hdr-Security).

<details><summary>Example decoding with encoding/gob 💥 fatal error (out of memory)</summary><p/>

```Go
// Example of encoding/gob having "fatal error: runtime: out of memory"
// while decoding 181 bytes.
package main
import (
	"bytes"
	"encoding/gob"
	"encoding/hex"
	"fmt"
)

// Example data is from https://github.com/golang/go/issues/24446
// (shortened to 181 bytes).
const data = "4dffb503010102303001ff30000109010130010800010130010800010130" +
	"01ffb80001014a01ffb60001014b01ff860001013001ff860001013001ff" +
	"860001013001ff860001013001ffb80000001eff850401010e3030303030" +
	"30303030303030303001ff3000010c0104000016ffb70201010830303030" +
	"3030303001ff3000010c000030ffb6040405fcff00303030303030303030" +
	"303030303030303030303030303030303030303030303030303030303030" +
	"30"

type X struct {
	J *X
	K map[string]int
}

func main() {
	raw, _ := hex.DecodeString(data)
	decoder := gob.NewDecoder(bytes.NewReader(raw))

	var x X
	decoder.Decode(&x) // fatal error: runtime: out of memory
	fmt.Println("Decoding finished.")
}
```

<hr/>

</details>

`fxamacker/cbor` is fast at rejecting malformed CBOR data.  E.g. attempts to  
decode 10 bytes of malicious CBOR data to `[]byte` (with default settings):

| Codec | Speed (ns/op) | Memory | Allocs |
| :---- | ------------: | -----: | -----: |
| fxamacker/cbor 2.5.0 | 44 ± 5% | 32 B/op | 2 allocs/op |
| ugorji/go 1.2.11 | 5353261 ± 4% | 67111321 B/op |  13 allocs/op |

<details><summary>Benchmark details</summary><p/>

Latest comparison used:
- Input: `[]byte{0x9B, 0x00, 0x00, 0x42, 0xFA, 0x42, 0xFA, 0x42, 0xFA, 0x42}`
- go1.19.10, linux/amd64, i5-13600K (disabled all e-cores, DDR4 @2933)
- go test -bench=. -benchmem -count=20

#### Prior comparisons

| Codec | Speed (ns/op) | Memory | Allocs |
| :---- | ------------: | -----: | -----: |
| fxamacker/cbor 2.5.0-beta2 | 44.33 ± 2% | 32 B/op | 2 allocs/op |
| fxamacker/cbor 0.1.0 - 2.4.0 | ~44.68 ± 6% | 32 B/op |  2 allocs/op |
| ugorji/go 1.2.10 | 5524792.50 ± 3% | 67110491 B/op |  12 allocs/op |
| ugorji/go 1.1.0 - 1.2.6 | 💥 runtime: | out of memory: | cannot allocate |

- Input: `[]byte{0x9B, 0x00, 0x00, 0x42, 0xFA, 0x42, 0xFA, 0x42, 0xFA, 0x42}`
- go1.19.6, linux/amd64, i5-13600K (DDR4)
- go test -bench=. -benchmem -count=20

<hr/>

</details>

### Smaller Encodings with Struct Tags

Struct tags (`toarray`, `keyasint`, `omitempty`) reduce encoded size of structs.

<details><summary>Example encoding 3-level nested Go struct to 1 byte CBOR</summary><p/>

https://go.dev/play/p/YxwvfPdFQG2

```Go
// Example encoding nested struct (with omitempty tag)
// - encoding/json:  18 byte JSON
// - fxamacker/cbor:  1 byte CBOR
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/fxamacker/cbor/v2"
)

type GrandChild struct {
	Quux int `json:",omitempty"`
}

type Child struct {
	Baz int        `json:",omitempty"`
	Qux GrandChild `json:",omitempty"`
}

type Parent struct {
	Foo Child `json:",omitempty"`
	Bar int   `json:",omitempty"`
}

func cb() {
	results, _ := cbor.Marshal(Parent{})
	fmt.Println("hex(CBOR): " + hex.EncodeToString(results))

	text, _ := cbor.Diagnose(results) // Diagnostic Notation
	fmt.Println("DN: " + text)
}

func js() {
	results, _ := json.Marshal(Parent{})
	fmt.Println("hex(JSON): " + hex.EncodeToString(results))

	text := string(results) // JSON
	fmt.Println("JSON: " + text)
}

func main() {
	cb()
	fmt.Println("-------------")
	js()
}
```

Output (DN is Diagnostic Notation):
```
hex(CBOR): a0
DN: {}
-------------
hex(JSON): 7b22466f6f223a7b22517578223a7b7d7d7d
JSON: {"Foo":{"Qux":{}}}
```

<hr/>

</details>

Example using different struct tags together:

![alt text](https://github.com/fxamacker/images/raw/master/cbor/v2.3.0/cbor_struct_tags_api.svg?sanitize=1 "CBOR API and Go Struct Tags")

API is mostly same as `encoding/json`, plus interfaces that simplify concurrency for CBOR options.

## Quick Start

__Install__: `go get github.com/fxamacker/cbor/v2` and `import "github.com/fxamacker/cbor/v2"`.

### Key Points

This library can encode and decode CBOR (RFC 8949) and CBOR Sequences (RFC 8742).

- __CBOR data item__ is a single piece of CBOR data and its structure may contain 0 or more nested data items.
- __CBOR sequence__ is a concatenation of 0 or more encoded CBOR data items.

Configurable limits and options can be used to balance trade-offs.

- Encoding and decoding modes are created from options (settings).
- Modes can be created at startup and reused.
- Modes are safe for concurrent use.

### Default Mode

Package level functions only use this library's default settings.  
They provide the "default mode" of encoding and decoding.

```go
// API matches encoding/json for Marshal, Unmarshal, Encode, Decode, etc.
b, err = cbor.Marshal(v)        // encode v to []byte b
err = cbor.Unmarshal(b, &v)     // decode []byte b to v
decoder = cbor.NewDecoder(r)    // create decoder with io.Reader r
err = decoder.Decode(&v)        // decode a CBOR data item to v

// v2.5.0 added new functions that return remaining bytes.

// UnmarshalFirst decodes first CBOR data item and returns remaining bytes.
rest, err = cbor.UnmarshalFirst(b, &v)   // decode []byte b to v

// DiagnoseFirst translates first CBOR data item to text and returns remaining bytes.
text, rest, err = cbor.DiagnoseFirst(b)  // decode []byte b to Diagnostic Notation text

// NOTE: Unmarshal returns ExtraneousDataError if there are remaining bytes,
// but new funcs UnmarshalFirst and DiagnoseFirst do not.
```

__IMPORTANT__: 👉  CBOR settings allow trade-offs between speed, security, encoding size, etc.

- Different CBOR libraries may use different default settings.
- CBOR-based formats or protocols usually require specific settings.

For example, WebAuthn uses "CTAP2 Canonical CBOR" which is available as a preset.

### Presets

Presets can be used as-is or as a starting point for custom settings.

```go
// EncOptions is a struct of encoder settings.
func CoreDetEncOptions() EncOptions              // RFC 8949 Core Deterministic Encoding
func PreferredUnsortedEncOptions() EncOptions    // RFC 8949 Preferred Serialization
func CanonicalEncOptions() EncOptions            // RFC 7049 Canonical CBOR
func CTAP2EncOptions() EncOptions                // FIDO2 CTAP2 Canonical CBOR
```

Presets are used to create custom modes.

### Custom Modes

Modes are created from settings. Once created, modes have immutable settings.

💡 Create the mode at startup and reuse it. It is safe for concurrent use.

```Go
// Create encoding mode.
opts := cbor.CoreDetEncOptions()   // use preset options as a starting point
opts.Time = cbor.TimeUnix          // change any settings if needed
em, err := opts.EncMode()          // create an immutable encoding mode

// Reuse the encoding mode. It is safe for concurrent use.

// API matches encoding/json.
b, err := em.Marshal(v)            // encode v to []byte b
encoder := em.NewEncoder(w)        // create encoder with io.Writer w
err := encoder.Encode(v)           // encode v to io.Writer w
```

Default mode and custom modes automatically apply struct tags.

### Struct Tags

Struct tags (`toarray`, `keyasint`, `omitempty`) reduce encoded size of structs.

<details><summary>Example encoding 3-level nested Go struct to 1 byte CBOR</summary><p/>

https://go.dev/play/p/YxwvfPdFQG2

```Go
// Example encoding nested struct (with omitempty tag)
// - encoding/json:  18 byte JSON
// - fxamacker/cbor:  1 byte CBOR
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/fxamacker/cbor/v2"
)

type GrandChild struct {
	Quux int `json:",omitempty"`
}

type Child struct {
	Baz int        `json:",omitempty"`
	Qux GrandChild `json:",omitempty"`
}

type Parent struct {
	Foo Child `json:",omitempty"`
	Bar int   `json:",omitempty"`
}

func cb() {
	results, _ := cbor.Marshal(Parent{})
	fmt.Println("hex(CBOR): " + hex.EncodeToString(results))

	text, _ := cbor.Diagnose(results) // Diagnostic Notation
	fmt.Println("DN: " + text)
}

func js() {
	results, _ := json.Marshal(Parent{})
	fmt.Println("hex(JSON): " + hex.EncodeToString(results))

	text := string(results) // JSON
	fmt.Println("JSON: " + text)
}

func main() {
	cb()
	fmt.Println("-------------")
	js()
}
```

Output (DN is Diagnostic Notation):
```
hex(CBOR): a0
DN: {}
-------------
hex(JSON): 7b22466f6f223a7b22517578223a7b7d7d7d
JSON: {"Foo":{"Qux":{}}}
```

<hr/>

</details>

<details><summary>Example using several struct tags</summary><p/>
	
![alt text](https://github.com/fxamacker/images/raw/master/cbor/v2.3.0/cbor_struct_tags_api.svg?sanitize=1 "CBOR API and Go Struct Tags")

</details>

Struct tags simplify use of CBOR-based protocols that require CBOR arrays or maps with integer keys.

### CBOR Tags

CBOR tags are specified in a `TagSet`.

Custom modes can be created with a `TagSet` to handle CBOR tags.
 
```go
em, err := opts.EncMode()                  // no CBOR tags
em, err := opts.EncModeWithTags(ts)        // immutable CBOR tags
em, err := opts.EncModeWithSharedTags(ts)  // mutable shared CBOR tags
```

`TagSet` and modes using it are safe for concurrent use.  Equivalent API is available for `DecMode`.

<details><summary>Example using TagSet and TagOptions</summary><p/>

```go
// Use signedCWT struct defined in "Decoding CWT" example.

// Create TagSet (safe for concurrency).
tags := cbor.NewTagSet()
// Register tag COSE_Sign1 18 with signedCWT type.
tags.Add(	
	cbor.TagOptions{EncTag: cbor.EncTagRequired, DecTag: cbor.DecTagRequired}, 
	reflect.TypeOf(signedCWT{}), 
	18)

// Create DecMode with immutable tags.
dm, _ := cbor.DecOptions{}.DecModeWithTags(tags)

// Unmarshal to signedCWT with tag support.
var v signedCWT
if err := dm.Unmarshal(data, &v); err != nil {
	return err
}

// Create EncMode with immutable tags.
em, _ := cbor.EncOptions{}.EncModeWithTags(tags)

// Marshal signedCWT with tag number.
if data, err := cbor.Marshal(v); err != nil {
	return err
}
```

</details>

### Functions and Interfaces

<details><summary>Functions and interfaces at a glance</summary><p/>

Common functions with same API as `encoding/json`:  
- `Marshal`, `Unmarshal`
- `NewEncoder`, `(*Encoder).Encode`
- `NewDecoder`, `(*Decoder).Decode`

NOTE: `Unmarshal` will return `ExtraneousDataError` if there are remaining bytes
because RFC 8949 treats CBOR data item with remaining bytes as malformed.
- 💡 Use `UnmarshalFirst` to decode first CBOR data item and return any remaining bytes.

Other useful functions: 
- `Diagnose`, `DiagnoseFirst` produce human-readable [Extended Diagnostic Notation](https://www.rfc-editor.org/rfc/rfc8610.html#appendix-G) from CBOR data.
- `UnmarshalFirst` decodes first CBOR data item and return any remaining bytes.
- `Wellformed` returns true if the the CBOR data item is well-formed.

Interfaces identical or comparable to Go `encoding` packages include:  
`Marshaler`, `Unmarshaler`, `BinaryMarshaler`, and `BinaryUnmarshaler`.

The `RawMessage` type can be used to delay CBOR decoding or precompute CBOR encoding.

</details>

### Security Tips

🔒 Use Go's `io.LimitReader` to limit size when decoding very large or indefinite size data.

Default limits may need to be increased for systems handling very large data (e.g. blockchains).

`DecOptions` can be used to modify default limits for `MaxArrayElements`, `MaxMapPairs`, and `MaxNestedLevels`.

## Status

v2.6.0 (February 2024) adds important new features, optimizations, and bug fixes. It is especially useful to systems that need to convert data between CBOR and JSON.  New options and optimizations improve handling of bignum, integers, maps, and strings.

For more details, see [release notes](https://github.com/fxamacker/cbor/releases).

### Prior Release

v2.5.0 was released on Sunday, August 13, 2023 with new features and important bug fixes.  It is fuzz tested and production quality after extended beta [v2.5.0-beta](https://github.com/fxamacker/cbor/releases/tag/v2.5.0-beta) (Dec 2022) -> [v2.5.0](https://github.com/fxamacker/cbor/releases/tag/v2.5.0) (Aug 2023).

__IMPORTANT__:  👉 Before upgrading from v2.4 or older release, please read the notable changes highlighted in the release notes.  v2.5.0 is a large release with bug fixes to error handling for extraneous data in `Unmarshal`, etc. that should be reviewed before upgrading.

See [v2.5.0 release notes](https://github.com/fxamacker/cbor/releases/tag/v2.5.0) for list of new features, improvements, and bug fixes.

See ["Version and API Changes"](https://github.com/fxamacker/cbor#versions-and-api-changes) section for more info about version numbering, etc.

<!--
<details><summary>👉 Benchmark Comparison: v2.4.0 vs v2.5.0</summary><p/>

TODO: Update to v2.4.0 vs 2.5.0 (not beta2).

Comparison of v2.4.0 vs v2.5.0-beta2 provided by @448 (edited to fit width).

PR [#382](https://github.com/fxamacker/cbor/pull/382) returns buffer to pool in `Encode()`. It adds a bit of overhead to `Encode()` but `NewEncoder().Encode()` is a lot faster and uses less memory as shown here:

```
$ benchstat bench-v2.4.0.log bench-f9e6291.log 
goos: linux
goarch: amd64
pkg: github.com/fxamacker/cbor/v2
cpu: 12th Gen Intel(R) Core(TM) i7-12700H
                                                     │ bench-v2.4.0.log │  bench-f9e6291.log                  │
                                                     │      sec/op      │   sec/op     vs base                │
NewEncoderEncode/Go_bool_to_CBOR_bool-20                   236.70n ± 2%   58.04n ± 1%  -75.48% (p=0.000 n=10)
NewEncoderEncode/Go_uint64_to_CBOR_positive_int-20         238.00n ± 2%   63.93n ± 1%  -73.14% (p=0.000 n=10)
NewEncoderEncode/Go_int64_to_CBOR_negative_int-20          238.65n ± 2%   64.88n ± 1%  -72.81% (p=0.000 n=10)
NewEncoderEncode/Go_float64_to_CBOR_float-20               242.00n ± 2%   63.00n ± 1%  -73.97% (p=0.000 n=10)
NewEncoderEncode/Go_[]uint8_to_CBOR_bytes-20               245.60n ± 1%   68.55n ± 1%  -72.09% (p=0.000 n=10)
NewEncoderEncode/Go_string_to_CBOR_text-20                 243.20n ± 3%   68.39n ± 1%  -71.88% (p=0.000 n=10)
NewEncoderEncode/Go_[]int_to_CBOR_array-20                 563.0n ± 2%    378.3n ± 0%  -32.81% (p=0.000 n=10)
NewEncoderEncode/Go_map[string]string_to_CBOR_map-20       2.043µ ± 2%    1.906µ ± 2%   -6.75% (p=0.000 n=10)
geomean                                                    349.7n         122.7n       -64.92%

                                                     │ bench-v2.4.0.log │    bench-f9e6291.log                │
                                                     │       B/op       │    B/op     vs base                 │
NewEncoderEncode/Go_bool_to_CBOR_bool-20                     128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_uint64_to_CBOR_positive_int-20           128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_int64_to_CBOR_negative_int-20            128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_float64_to_CBOR_float-20                 128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_[]uint8_to_CBOR_bytes-20                 128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_string_to_CBOR_text-20                   128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_[]int_to_CBOR_array-20                   128.0 ± 0%     0.0 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_map[string]string_to_CBOR_map-20         544.0 ± 0%   416.0 ± 0%   -23.53% (p=0.000 n=10)
geomean                                                      153.4                    ?                       ¹ ²
¹ summaries must be >0 to compute geomean
² ratios must be >0 to compute geomean

                                                     │ bench-v2.4.0.log │    bench-f9e6291.log                │
                                                     │    allocs/op     │ allocs/op   vs base                 │
NewEncoderEncode/Go_bool_to_CBOR_bool-20                     2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_uint64_to_CBOR_positive_int-20           2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_int64_to_CBOR_negative_int-20            2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_float64_to_CBOR_float-20                 2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_[]uint8_to_CBOR_bytes-20                 2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_string_to_CBOR_text-20                   2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_[]int_to_CBOR_array-20                   2.000 ± 0%   0.000 ± 0%  -100.00% (p=0.000 n=10)
NewEncoderEncode/Go_map[string]string_to_CBOR_map-20         28.00 ± 0%   26.00 ± 0%    -7.14% (p=0.000 n=10)
geomean                                                      2.782                    ?                       ¹ ²
¹ summaries must be >0 to compute geomean
² ratios must be >0 to compute geomean
```

</details>
-->

## Who uses fxamacker/cbor

`fxamacker/cbor` is used in projects by Arm Ltd., Berlin Institute of Health at Charité, Chainlink, Cisco, Confidential Computing Consortium, ConsenSys, Dapper&nbsp;Labs, EdgeX&nbsp;Foundry, F5, FIDO Alliance, Fraunhofer&#8209;AISEC, Let's Encrypt (ISRG), Linux&nbsp;Foundation, Matrix.org, Microsoft, Mozilla, National&nbsp;Cybersecurity&nbsp;Agency&nbsp;of&nbsp;France (govt), Netherlands (govt), Oasis Protocol, Smallstep, Tailscale, Taurus SA, Teleport, TIBCO, and others.

`fxamacker/cbor` passed multiple confidential security assessments.  A [nonconfidential security assessment](https://github.com/veraison/go-cose/blob/v1.0.0-rc.1/reports/NCC_Microsoft-go-cose-Report_2022-05-26_v1.0.pdf) (prepared by NCC Group for Microsoft Corporation) includes a subset of fxamacker/cbor v2.4.0 in its scope.

## Standards

`fxamacker/cbor` is a CBOR codec in full conformance with [IETF STD&nbsp;94 (RFC&nbsp;8949)](https://www.rfc-editor.org/info/std94). It also supports CBOR Sequences ([RFC&nbsp;8742](https://www.rfc-editor.org/rfc/rfc8742.html)) and Extended Diagnostic Notation ([Appendix G of RFC&nbsp;8610](https://www.rfc-editor.org/rfc/rfc8610.html#appendix-G)).

Notable CBOR features include:

| CBOR Feature  | Description  |
| :--- | :--- |
| CBOR tags | API supports built-in and user-defined tags.  |
| Preferred serialization | Integers encode to fewest bytes. Optional float64 → float32 → float16. |
| Map key sorting | Unsorted, length-first (Canonical CBOR), and bytewise-lexicographic (CTAP2). |
| Duplicate map keys | Always forbid for encoding and option to allow/forbid for decoding.   |
| Indefinite length data | Option to allow/forbid for encoding and decoding. |
| Well-formedness | Always checked and enforced. |
| Basic validity checks | Optionally check UTF-8 validity and duplicate map keys. |
| Security considerations | Prevent integer overflow and resource exhaustion (RFC 8949 Section 10). |

Known limitations are noted in the [Limitations section](#limitations). 

Go nil values for slices, maps, pointers, etc. are encoded as CBOR null.  Empty slices, maps, etc. are encoded as empty CBOR arrays and maps.

Decoder checks for all required well-formedness errors, including all "subkinds" of syntax errors and too little data.

After well-formedness is verified, basic validity errors are handled as follows:

* Invalid UTF-8 string: Decoder has option to check and return invalid UTF-8 string error. This check is enabled by default.
* Duplicate keys in a map: Decoder has options to ignore or enforce rejection of duplicate map keys.

When decoding well-formed CBOR arrays and maps, decoder saves the first error it encounters and continues with the next item.  Options to handle this differently may be added in the future.

By default, decoder treats time values of floating-point NaN and Infinity as if they are CBOR Null or CBOR Undefined.

__Click to expand topic:__

<details>
 <summary>Duplicate Map Keys</summary><p>

This library provides options for fast detection and rejection of duplicate map keys based on applying a Go-specific data model to CBOR's extended generic data model in order to determine duplicate vs distinct map keys. Detection relies on whether the CBOR map key would be a duplicate "key" when decoded and applied to the user-provided Go map or struct. 

`DupMapKeyQuiet` turns off detection of duplicate map keys. It tries to use a "keep fastest" method by choosing either "keep first" or "keep last" depending on the Go data type.

`DupMapKeyEnforcedAPF` enforces detection and rejection of duplidate map keys. Decoding stops immediately and returns `DupMapKeyError` when the first duplicate key is detected. The error includes the duplicate map key and the index number. 

APF suffix means "Allow Partial Fill" so the destination map or struct can contain some decoded values at the time of error. It is the caller's responsibility to respond to the `DupMapKeyError` by discarding the partially filled result if that's required by their protocol.

</details>

<details>
 <summary>Tag Validity</summary><p>

This library checks tag validity for built-in tags (currently tag numbers 0, 1, 2, 3, and 55799):

* Inadmissible type for tag content 
* Inadmissible value for tag content

Unknown tag data items (not tag number 0, 1, 2, 3, or 55799) are handled in two ways:

* When decoding into an empty interface, unknown tag data item will be decoded into `cbor.Tag` data type, which contains tag number and tag content.  The tag content will be decoded into the default Go data type for the CBOR data type.
* When decoding into other Go types, unknown tag data item is decoded into the specified Go type.  If Go type is registered with a tag number, the tag number can optionally be verified.

Decoder also has an option to forbid tag data items (treat any tag data item as error) which is specified by protocols such as CTAP2 Canonical CBOR.  

For more information, see [decoding options](#decoding-options-1) and [tag options](#tag-options).

</details>

## Limitations

If any of these limitations prevent you from using this library, please open an issue along with a link to your project.

* CBOR `Undefined` (0xf7) value decodes to Go's `nil` value.  CBOR `Null` (0xf6) more closely matches Go's `nil`.
* CBOR map keys with data types not supported by Go for map keys are ignored and an error is returned after continuing to decode remaining items.  
* When decoding registered CBOR tag data to interface type, decoder creates a pointer to registered Go type matching CBOR tag number.  Requiring a pointer for this is a Go limitation. 

## Fuzzing and Code Coverage

__Code coverage__ is always 95% or higher (with `go test -cover`) when tagging a release.

__Coverage-guided fuzzing__ must pass billions of execs using before tagging a release.  Fuzzing is done using nonpublic code which may eventually get merged into this project.  Until then, reports like OpenSSF&nbsp;Scorecard can't detect fuzz tests being used by this project.

<hr>

## Versions and API Changes
This project uses [Semantic Versioning](https://semver.org), so the API is always backwards compatible unless the major version number changes.  

These functions have signatures identical to encoding/json and their API will continue to match `encoding/json` even after major new releases:  
`Marshal`, `Unmarshal`, `NewEncoder`, `NewDecoder`, `(*Encoder).Encode`, and `(*Decoder).Decode`.

Exclusions from SemVer:
- Newly added API documented as "subject to change".
- Newly added API in the master branch that has never been tagged in non-beta release.
- If function parameters are unchanged, bug fixes that change behavior (e.g. return error for edge case was missed in prior version).  We try to highlight these in the release notes and add extended beta period.  E.g. [v2.5.0-beta](https://github.com/fxamacker/cbor/releases/tag/v2.5.0-beta) (Dec 2022) -> [v2.5.0](https://github.com/fxamacker/cbor/releases/tag/v2.5.0) (Aug 2023).

This project avoids breaking changes to behavior of encoding and decoding functions unless required to improve conformance with supported RFCs (e.g. RFC 8949, RFC 8742, etc.)  Visible changes that don't improve conformance to standards are typically made available as new opt-in settings or new functions.

## Code of Conduct 

This project has adopted the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).  Contact [faye.github@gmail.com](mailto:faye.github@gmail.com) with any questions or comments.

## Contributing

Please open an issue before beginning work on a PR.  The improvement may have already been considered, etc.

For more info, see [How to Contribute](CONTRIBUTING.md).

## Security Policy

Security fixes are provided for the latest released version of fxamacker/cbor.

For the full text of the Security Policy, see [SECURITY.md](SECURITY.md).

## Acknowledgements

Many thanks to all the contributors on this project!

I'm especially grateful to Bastian Müller and Dieter Shirley for suggesting and collaborating on CBOR stream mode, and much more.

I'm very grateful to Stefan Tatschner, Yawning Angel, Jernej Kos, x448, ZenGround0, and Jakob Borg for their contributions or support in the very early days.

This library clearly wouldn't be possible without Carsten Bormann authoring CBOR RFCs.

Special thanks to Laurence Lundblade and Jeffrey Yasskin for their help on IETF mailing list or at [7049bis](https://github.com/cbor-wg/CBORbis).

Huge thanks to The Go Authors for creating a fun and practical programming language with batteries included!

This library uses `x448/float16` which used to be included.  As a standalone package, `x448/float16` is useful to other projects as well.

## License

Copyright © 2019-2024 [Faye Amacker](https://github.com/fxamacker).

fxamacker/cbor is licensed under the MIT License.  See [LICENSE](LICENSE) for the full license text.

<hr>
