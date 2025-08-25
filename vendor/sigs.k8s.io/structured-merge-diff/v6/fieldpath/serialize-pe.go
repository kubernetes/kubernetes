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
	"sigs.k8s.io/structured-merge-diff/v6/value"
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
)

// readJSONIter reads a Value from a JSON iterator.
// DO NOT EXPORT
// TODO: eliminate this https://github.com/kubernetes-sigs/structured-merge-diff/issues/202
func readJSONIter(iter *jsoniter.Iterator) (value.Value, error) {
	v := iter.Read()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil, iter.Error
	}
	return value.NewValueInterface(v), nil
}

// writeJSONStream writes a value into a JSON stream.
// DO NOT EXPORT
// TODO: eliminate this https://github.com/kubernetes-sigs/structured-merge-diff/issues/202
func writeJSONStream(v value.Value, stream *jsoniter.Stream) {
	stream.WriteVal(v.Unstructured())
}

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
		v, err := readJSONIter(iter)
		if err != nil {
			return PathElement{}, err
		}
		return PathElement{Value: &v}, nil
	case peKeySepBytes[0]:
		iter := readPool.BorrowIterator(b)
		defer readPool.ReturnIterator(iter)
		fields := value.FieldList{}

		iter.ReadObjectCB(func(iter *jsoniter.Iterator, key string) bool {
			v, err := readJSONIter(iter)
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
				stream.WriteMore()
			}
			stream.WriteObjectField(field.Name)
			writeJSONStream(field.Value, stream)
		}
		stream.WriteObjectEnd()
	case pe.Value != nil:
		if _, err := stream.Write(peValueSepBytes); err != nil {
			return err
		}
		writeJSONStream(*pe.Value, stream)
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
