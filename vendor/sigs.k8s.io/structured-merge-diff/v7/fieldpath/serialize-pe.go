/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fieldpath

import (
	"bytes"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"strconv"
	"strings"

	internaljson "sigs.k8s.io/structured-merge-diff/v7/internal/json"
	"sigs.k8s.io/structured-merge-diff/v7/value"
)

var ErrUnknownPathElementType = errors.New("unknown path element type")

// allowInvalidUTF8 tolerates invalid utf8 (the offending bytes are replaced with the
// Unicode replacement character) rather than rejecting, matching the
// behaviour of the JSON parser this package used previously.
var allowInvalidUTF8 = jsontext.AllowInvalidUTF8(true)

// allowDuplicates allows tolerating duplicates when decoding or encoding.
var allowDuplicates = jsontext.AllowDuplicateNames(true)

const (
	// Field indicates that the content of this path element is a field's name
	peField = 'f'

	// Value indicates that the content of this path element is a field's value
	peValue = 'v'

	// Index indicates that the content of this path element is an index in an array
	peIndex = 'i'

	// Key indicates that the content of this path element is a key value map
	peKey = 'k'

	// Separator separates the type of a path element from the contents
	peSeparator = ':'
)

var (
	peFieldSepBytes = []byte{peField, peSeparator}
	peValueSepBytes = []byte{peValue, peSeparator}
	peIndexSepBytes = []byte{peIndex, peSeparator}
	peKeySepBytes   = []byte{peKey, peSeparator}
)

// DeserializePathElement parses a serialized path element
func DeserializePathElement(s string) (PathElement, error) {
	if len(s) < 2 {
		return PathElement{}, errors.New("key must be 2 characters long")
	}
	typeSep0, typeSep1 := s[0], s[1]
	if typeSep1 != peSeparator {
		return PathElement{}, fmt.Errorf("missing colon: %v", s)
	}
	switch typeSep0 {
	case peFieldSepBytes[0]:
		str := s[2:]
		return PathElement{
			FieldName: &str,
		}, nil
	case peValueSepBytes[0]:
		v, err := internaljson.UnmarshalToAnyMergingDuplicates([]byte(s[2:]))
		if err != nil {
			return PathElement{}, err
		}
		interfaceValue := value.NewValueInterface(v)
		return PathElement{Value: &interfaceValue}, nil
	case peKeySepBytes[0]:
		var fields value.FieldList
		// preserve json-iterator behavior of tolerating duplicates (duplicates accumulate)
		if err := json.UnmarshalRead(strings.NewReader(s[2:]), &fields, allowInvalidUTF8, allowDuplicates); err != nil {
			return PathElement{}, err
		}
		if fields == nil {
			// preserve json-iterator behavior of always returning a non-nil field list
			fields = value.FieldList{}
		}
		return PathElement{Key: &fields}, nil
	case peIndexSepBytes[0]:
		i, err := strconv.Atoi(s[2:])
		if err != nil {
			return PathElement{}, err
		}
		return PathElement{
			Index: &i,
		}, nil
	default:
		return PathElement{}, ErrUnknownPathElementType
	}
}

// SerializePathElement serializes a path element
func SerializePathElement(pe PathElement) (string, error) {
	serializer := pool.Get().(*pathElementSerializer)
	defer func() {
		serializer.reset()
		pool.Put(serializer)
	}()
	if err := serializer.serialize(pe); err != nil {
		return "", err
	}
	return serializer.builder.String(), nil
}

type pathElementSerializer struct {
	builder   bytes.Buffer
	fastValue fastMarshalValue
}

type fastMarshalValue struct {
	Value *value.Value
}

var _ json.MarshalerTo = fastMarshalValue{}

func (mv fastMarshalValue) MarshalJSONTo(enc *jsontext.Encoder) error {
	return value.ValueMarshalJSONTo(enc, *mv.Value)
}

func (pes *pathElementSerializer) reset() {
	if pes.builder.Cap() > maxRetainedBuffer {
		pes.builder = bytes.Buffer{}
	} else {
		pes.builder.Reset()
	}
	pes.fastValue.Value = nil
}

func (pes *pathElementSerializer) serialize(pe PathElement) error {
	switch {
	case pe.FieldName != nil:
		if _, err := pes.builder.Write(peFieldSepBytes); err != nil {
			return err
		}
		if _, err := pes.builder.WriteString(*pe.FieldName); err != nil {
			return err
		}
	case pe.Key != nil:
		if _, err := pes.builder.Write(peKeySepBytes); err != nil {
			return err
		}
		// preserve json-iterator behavior of tolerating duplicates (duplicates output multiple times)
		if err := json.MarshalWrite(&pes.builder, pe.Key, json.Deterministic(true), allowDuplicates, allowInvalidUTF8); err != nil {
			return err
		}
	case pe.Value != nil:
		if _, err := pes.builder.Write(peValueSepBytes); err != nil {
			return err
		}
		pes.fastValue.Value = pe.Value
		if err := json.MarshalWrite(&pes.builder, &pes.fastValue, json.Deterministic(true), allowInvalidUTF8); err != nil {
			return err
		}
	case pe.Index != nil:
		if _, err := pes.builder.Write(peIndexSepBytes); err != nil {
			return err
		}
		if _, err := pes.builder.WriteString(strconv.Itoa(*pe.Index)); err != nil {
			return err
		}
	default:
		return errors.New("invalid PathElement")
	}
	return nil
}
