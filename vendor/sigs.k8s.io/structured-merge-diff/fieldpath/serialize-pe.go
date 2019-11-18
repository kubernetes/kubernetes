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
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	jsoniter "github.com/json-iterator/go"
	"sigs.k8s.io/structured-merge-diff/value"
)

var ErrUnknownPathElementType = errors.New("unknown path element type")

const (
	// Field indicates that the content of this path element is a field's name
	peField = "f"

	// Value indicates that the content of this path element is a field's value
	peValue = "v"

	// Index indicates that the content of this path element is an index in an array
	peIndex = "i"

	// Key indicates that the content of this path element is a key value map
	peKey = "k"

	// Separator separates the type of a path element from the contents
	peSeparator = ":"
)

var (
	peFieldSepBytes = []byte(peField + peSeparator)
	peValueSepBytes = []byte(peValue + peSeparator)
	peIndexSepBytes = []byte(peIndex + peSeparator)
	peKeySepBytes   = []byte(peKey + peSeparator)
	peSepBytes      = []byte(peSeparator)

	peFieldSepV2Bytes = []byte(`"` + peField + `_",`)
	peValueSepV2Bytes = []byte(`"` + peValue + `_",`)
	peIndexSepV2Bytes = []byte(`"` + peIndex + `_",`)
	peKeySepV2Bytes   = []byte(`"` + peKey + `_",`)
)

// DeserializePathElement parses a serialized path element
func DeserializePathElement(s string) (PathElement, error) {
	b := []byte(s)
	if len(b) < 2 {
		return PathElement{}, errors.New("key must be 2 characters long:")
	}
	typeSep, b := b[:2], b[2:]
	if typeSep[1] != peSepBytes[0] {
		return PathElement{}, fmt.Errorf("missing colon: %v", s)
	}
	switch typeSep[0] {
	case peFieldSepBytes[0]:
		// Slice s rather than convert b, to save on
		// allocations.
		str := s[2:]
		return PathElement{
			FieldName: &str,
		}, nil
	case peValueSepBytes[0]:
		iter := readPool.BorrowIterator(b)
		defer readPool.ReturnIterator(iter)
		v, err := value.ReadJSONIter(iter)
		if err != nil {
			return PathElement{}, err
		}
		return PathElement{Value: &v}, nil
	case peKeySepBytes[0]:
		iter := readPool.BorrowIterator(b)
		defer readPool.ReturnIterator(iter)
		fields := value.FieldList{}
		iter.ReadObjectCB(func(iter *jsoniter.Iterator, key string) bool {
			v, err := value.ReadJSONIter(iter)
			if err != nil {
				iter.Error = err
				return false
			}
			fields = append(fields, value.Field{Name: key, Value: v})
			return true
		})
		fields.Sort()
		return PathElement{Key: &fields}, iter.Error
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

var (
	readPool  = jsoniter.NewIterator(jsoniter.ConfigCompatibleWithStandardLibrary).Pool()
	writePool = jsoniter.NewStream(jsoniter.ConfigCompatibleWithStandardLibrary, nil, 1024).Pool()
)

// SerializePathElement serializes a path element
func SerializePathElement(pe PathElement) (string, error) {
	buf := strings.Builder{}
	err := serializePathElementToWriter(&buf, pe)
	return buf.String(), err
}

func serializePathElementToWriter(w io.Writer, pe PathElement) error {
	stream := writePool.BorrowStream(w)
	defer writePool.ReturnStream(stream)
	switch {
	case pe.FieldName != nil:
		if _, err := stream.Write(peFieldSepBytes); err != nil {
			return err
		}
		stream.WriteRaw(*pe.FieldName)
	case pe.Key != nil:
		if _, err := stream.Write(peKeySepBytes); err != nil {
			return err
		}
		stream.WriteObjectStart()
		for i, field := range *pe.Key {
			if i > 0 {
				stream.WriteRaw(",")
			}
			stream.WriteObjectField(field.Name)
			field.Value.WriteJSONStream(stream)
		}
		stream.WriteObjectEnd()
	case pe.Value != nil:
		if _, err := stream.Write(peValueSepBytes); err != nil {
			return err
		}
		pe.Value.WriteJSONStream(stream)
	case pe.Index != nil:
		if _, err := stream.Write(peIndexSepBytes); err != nil {
			return err
		}
		stream.WriteInt(*pe.Index)
	default:
		return errors.New("invalid PathElement")
	}
	b := stream.Buffer()
	err := stream.Flush()
	// Help jsoniter manage its buffers--without this, the next
	// use of the stream is likely to require an allocation. Look
	// at the jsoniter stream code to understand why. They were probably
	// optimizing for folks using the buffer directly.
	stream.SetBuffer(b[:0])
	return err
}

type v2EntryType byte
type v2ValueType byte

const (
	etSelf     v2EntryType = 's'
	etChildren v2EntryType = 'c'
	etBoth     v2EntryType = 'b'
	etInvalid  v2EntryType = 0

	vtField   v2ValueType = 'f'
	vtValue   v2ValueType = 'v'
	vtIndex   v2ValueType = 'i'
	vtKey     v2ValueType = 'k'
	vtInvalid v2ValueType = 0
)

func (et v2EntryType) asNumber() int {
	switch et {
	case etSelf:
		return 0
	case etChildren:
		return 1
	case etBoth:
		return 2
	}
	panic("unexpected entry type")
}

func etFromNumber(i int) v2EntryType {
	switch i {
	case 0:
		return etSelf
	case 1:
		return etChildren
	case 2:
		return etBoth
	}
	return etInvalid
}

func (vt v2ValueType) asNumber() int {
	switch vt {
	case vtField:
		return 0
	case vtValue:
		return 1
	case vtIndex:
		return 2
	case vtKey:
		return 3
	}
	panic("unexpected value type")
}

func vtFromNumber(i int) v2ValueType {
	switch i {
	case 0:
		return vtField
	case 1:
		return vtValue
	case 2:
		return vtIndex
	case 3:
		return vtKey
	}
	return vtInvalid
}

func v2CombineTypes(et v2EntryType, vt v2ValueType) int {
	if et == etInvalid || vt == vtInvalid {
		panic("logic error - can't combine invalid things")
	}
	return et.asNumber()*4 + vt.asNumber()
}

func v2SplitTypes(i int) (et v2EntryType, vt v2ValueType) {
	et = etFromNumber(i / 4)
	vt = vtFromNumber(i % 4)
	return
}

var (
	v2NumberToAscii = func() map[int]string {
		out := map[int]string{}
		for i := 0; i < 12; i++ {
			out[i] = strconv.Itoa(i)
		}
		return out
	}()
)

func emitV2Prefix(stream value.Stream, vt v2ValueType, et v2EntryType) error {
	n := v2CombineTypes(et, vt)
	str, ok := v2NumberToAscii[n]
	if !ok {
		return fmt.Errorf("unexpected entry number %v", n)
	}
	stream.WriteRaw(str)
	stream.WriteRaw(",")
	return nil
}

// you must write the trailing "," if you need it.
func serializePathElementToStreamV2(stream value.Stream, pe PathElement, et v2EntryType) error {
	switch {
	case pe.FieldName != nil:
		if err := emitV2Prefix(stream, vtField, et); err != nil {
			return err
		}
		stream.WriteString(*pe.FieldName)
	case pe.Key != nil:
		if err := emitV2Prefix(stream, vtKey, et); err != nil {
			return err
		}
		stream.WriteObjectStart()
		kvPairs := *pe.Key
		for i := range kvPairs {
			if i > 0 {
				stream.WriteRaw(",")
			}
			stream.WriteObjectField(kvPairs[i].Name)
			kvPairs[i].Value.WriteJSONStream(stream)
		}
		stream.WriteObjectEnd()
	case pe.Value != nil:
		if err := emitV2Prefix(stream, vtValue, et); err != nil {
			return err
		}
		pe.Value.WriteJSONStream(stream)
	case pe.Index != nil:
		if err := emitV2Prefix(stream, vtIndex, et); err != nil {
			return err
		}
		stream.WriteInt(*pe.Index)
	default:
		return errors.New("invalid PathElement")
	}
	return nil
}
