// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

/*
Package cbor is a modern CBOR codec (RFC 8949 & RFC 8742) with CBOR tags,
Go struct tag options (toarray/keyasint/omitempty/omitzero), Core Deterministic Encoding,
CTAP2, Canonical CBOR, float64->32->16, and duplicate map key detection.

Encoding options allow "preferred serialization" by encoding integers and floats
to their smallest forms (e.g. float16) when values fit.

Struct tag options "keyasint", "toarray", "omitempty", and "omitzero" reduce encoding size
and reduce programming effort.

For example, "toarray" tag makes struct fields encode to CBOR array elements.  And
"keyasint" makes a field encode to an element of CBOR map with specified int key.

Latest docs can be viewed at https://github.com/fxamacker/cbor#cbor-library-in-go

# Basics

The Quick Start guide is at https://github.com/fxamacker/cbor#quick-start

Function signatures identical to encoding/json include:

	Marshal, Unmarshal, NewEncoder, NewDecoder, (*Encoder).Encode, (*Decoder).Decode

Standard interfaces include:

	BinaryMarshaler, BinaryUnmarshaler, Marshaler, and Unmarshaler

Diagnostic functions translate CBOR data item into Diagnostic Notation:

	Diagnose, DiagnoseFirst

Functions that simplify using CBOR Sequences (RFC 8742) include:

	UnmarshalFirst

Custom encoding and decoding is possible by implementing standard interfaces for
user-defined Go types.

Codec functions are available at package-level (using defaults options) or by
creating modes from options at runtime.

"Mode" in this API means definite way of encoding (EncMode) or decoding (DecMode).

EncMode and DecMode interfaces are created from EncOptions or DecOptions structs.

	em, err := cbor.EncOptions{...}.EncMode()
	em, err := cbor.CanonicalEncOptions().EncMode()
	em, err := cbor.CTAP2EncOptions().EncMode()

Modes use immutable options to avoid side-effects and simplify concurrency. Behavior of
modes won't accidentally change at runtime after they're created.

Modes are intended to be reused and are safe for concurrent use.

EncMode and DecMode Interfaces

	// EncMode interface uses immutable options and is safe for concurrent use.
	type EncMode interface {
		Marshal(v interface{}) ([]byte, error)
		NewEncoder(w io.Writer) *Encoder
		EncOptions() EncOptions  // returns copy of options
	}

	// DecMode interface uses immutable options and is safe for concurrent use.
	type DecMode interface {
		Unmarshal(data []byte, v interface{}) error
		NewDecoder(r io.Reader) *Decoder
		DecOptions() DecOptions  // returns copy of options
	}

Using Default Encoding Mode

	b, err := cbor.Marshal(v)

	encoder := cbor.NewEncoder(w)
	err = encoder.Encode(v)

Using Default Decoding Mode

	err := cbor.Unmarshal(b, &v)

	decoder := cbor.NewDecoder(r)
	err = decoder.Decode(&v)

Using Default Mode of UnmarshalFirst to Decode CBOR Sequences

	// Decode the first CBOR data item and return remaining bytes:
	rest, err = cbor.UnmarshalFirst(b, &v)   // decode []byte b to v

Using Extended Diagnostic Notation (EDN) to represent CBOR data

	// Translate the first CBOR data item into text and return remaining bytes.
	text, rest, err = cbor.DiagnoseFirst(b)  // decode []byte b to text

Creating and Using Encoding Modes

	// Create EncOptions using either struct literal or a function.
	opts := cbor.CanonicalEncOptions()

	// If needed, modify encoding options
	opts.Time = cbor.TimeUnix

	// Create reusable EncMode interface with immutable options, safe for concurrent use.
	em, err := opts.EncMode()

	// Use EncMode like encoding/json, with same function signatures.
	b, err := em.Marshal(v)
	// or
	encoder := em.NewEncoder(w)
	err := encoder.Encode(v)

	// NOTE: Both em.Marshal(v) and encoder.Encode(v) use encoding options
	// specified during creation of em (encoding mode).

# CBOR Options

Predefined Encoding Options: https://github.com/fxamacker/cbor#predefined-encoding-options

Encoding Options: https://github.com/fxamacker/cbor#encoding-options

Decoding Options: https://github.com/fxamacker/cbor#decoding-options

# Struct Tags

Struct tags like `cbor:"name,omitempty"` and `json:"name,omitempty"` work as expected.
If both struct tags are specified then `cbor` is used.

Struct tag options like "keyasint", "toarray", "omitempty", and "omitzero" make it easy to use
very compact formats like COSE and CWT (CBOR Web Tokens) with structs.

The "omitzero" option omits zero values from encoding, matching
[stdlib encoding/json behavior](https://pkg.go.dev/encoding/json#Marshal).
When specified in the `cbor` tag, the option is always honored.
When specified in the `json` tag, the option is honored when building with Go 1.24+.

For example, "toarray" makes struct fields encode to array elements.  And "keyasint"
makes struct fields encode to elements of CBOR map with int keys.

https://raw.githubusercontent.com/fxamacker/images/master/cbor/v2.0.0/cbor_easy_api.png

Struct tag options are listed at https://github.com/fxamacker/cbor#struct-tags-1

# Tests and Fuzzing

Over 375 tests are included in this package. Cover-guided fuzzing is handled by
a private fuzzer that replaced fxamacker/cbor-fuzz years ago.
*/
package cbor
