package json

import (
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
)

// decodeOptions are the options used whenever this package parses JSON.
// Invalid UTF-8 is tolerated (the offending bytes are replaced with the
// Unicode replacement character) rather than rejected, matching the
// behaviour of the JSON parser this package used previously.
var allowInvalidUTF8 = jsontext.AllowInvalidUTF8(true)

// allowDuplicates allows tolerating duplicates when decoding or encoding.
var allowDuplicates = jsontext.AllowDuplicateNames(true)

// v1Merging matches json/v1 merging behavior for repeated keys
var v1Merging = jsonv1.MergeWithLegacySemantics(true)

// ReadValueToAnyMergingDuplicates reads the next value to an `any` type, returning any error.
// Sets options AllowInvalidUTF8=true, AllowDuplicateNames=true, MergeWithLegacySemantics=true.
func ReadValueToAnyMergingDuplicates(dec *jsontext.Decoder) (any, error) {
	data, err := dec.ReadValue()
	if err != nil {
		return nil, err
	}
	return UnmarshalToAnyMergingDuplicates(data)
}

// UnmarshalToAnyMergingDuplicates unmarshals data to an `any` type, returning any error.
// Sets options AllowInvalidUTF8=true, AllowDuplicateNames=true, MergeWithLegacySemantics=true.
func UnmarshalToAnyMergingDuplicates(data []byte) (any, error) {
	// try the non-duplicate path first, since we don't expect duplicates and json/v2 has an optimized non-merging version
	var v any
	if err := json.Unmarshal(data, &v, allowInvalidUTF8); !errors.Is(err, jsontext.ErrDuplicateName) {
		return v, err
	}
	// only if we got an ErrDuplicateName should we fall back to the slower unmarshal allowing duplicates and merging
	v = nil
	return v, json.Unmarshal(data, &v, allowInvalidUTF8, allowDuplicates, v1Merging)
}
