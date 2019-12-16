/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"io"
	"unsafe"

	jsoniter "github.com/json-iterator/go"
	"sigs.k8s.io/structured-merge-diff/fieldpath/strings"
	"sigs.k8s.io/structured-merge-diff/value"
)

func (s *Set) ToJSON() ([]byte, error) {
	buf := bytes.Buffer{}
	err := s.ToJSONStream(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (s *Set) ToJSONStream(w io.Writer) error {
	stream := writePool.BorrowStream(w)
	defer writePool.ReturnStream(stream)

	var r reusableBuilder

	stream.WriteObjectStart()
	err := s.emitContents_v1(false, stream, &r)
	if err != nil {
		return err
	}
	stream.WriteObjectEnd()
	return stream.Flush()
}

func (s *Set) ToJSON_V2Experimental() ([]byte, error) {
	buf := bytes.Buffer{}
	err := s.ToJSONStream_V2Experimental(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (s *Set) ToJSONStream_V2Experimental(w io.Writer) error {
	innerStream := writePool.BorrowStream(w)
	defer writePool.ReturnStream(innerStream)

	stream, err := strings.NewStreamWithStringTable(innerStream)
	if err != nil {
		return err
	}

	if err := manageMemory(stream); err != nil {
		return err
	}

	var r reusableBuilder

	stream.WriteArrayStart()
	stream.WriteInt(strings.DefaultVersion)
	err = s.emitContents_v2(false, stream, &r, false)
	if err != nil {
		return err
	}
	stream.WriteArrayEnd()
	return stream.Flush()
}

func manageMemory(stream value.Stream) error {
	// Help jsoniter manage its buffers--without this, it does a bunch of
	// alloctaions that are not necessary. They were probably optimizing
	// for folks using the buffer directly.
	b := stream.Buffer()
	if cap(b) < 4*1024 {
		b2 := make([]byte, len(b), 5*1024)
		copy(b2, b)
		stream.SetBuffer(b2)
		b = b2
	}
	if len(b) > 4*1024 || cap(b)-len(b) < 1024 {
		if err := stream.Flush(); err != nil {
			return err
		}
		stream.SetBuffer(b[:0])
	}
	return nil
}

type reusableBuilder struct {
	bytes.Buffer
}

func (r *reusableBuilder) unsafeString() string {
	b := r.Bytes()
	return *(*string)(unsafe.Pointer(&b))
}

func (r *reusableBuilder) reset() *bytes.Buffer {
	r.Reset()
	return &r.Buffer
}

func (s *Set) emitContents_v1(includeSelf bool, stream *jsoniter.Stream, r *reusableBuilder) error {
	mi, ci := 0, 0
	first := true
	preWrite := func() {
		if first {
			first = false
			return
		}
		stream.WriteRaw(",")
		// WriteMore flushes, which we don't want since we manage our own flushing.
		// stream.WriteMore()
	}

	for mi < len(s.Members.members) && ci < len(s.Children.members) {
		mpe := s.Members.members[mi]
		cpe := s.Children.members[ci].pathElement

		if mpe.Less(cpe) {
			preWrite()
			if err := serializePathElementToWriter(r.reset(), mpe); err != nil {
				return err
			}
			stream.WriteObjectField(r.unsafeString())
			stream.WriteEmptyObject()
			mi++
		} else if cpe.Less(mpe) {
			preWrite()
			if err := serializePathElementToWriter(r.reset(), cpe); err != nil {
				return err
			}
			stream.WriteObjectField(r.unsafeString())
			stream.WriteObjectStart()
			if err := s.Children.members[ci].set.emitContents_v1(false, stream, r); err != nil {
				return err
			}
			stream.WriteObjectEnd()
			ci++
		} else {
			preWrite()
			if err := serializePathElementToWriter(r.reset(), cpe); err != nil {
				return err
			}
			stream.WriteObjectField(r.unsafeString())
			stream.WriteObjectStart()
			if err := s.Children.members[ci].set.emitContents_v1(true, stream, r); err != nil {
				return err
			}
			stream.WriteObjectEnd()
			mi++
			ci++
		}
	}

	for mi < len(s.Members.members) {
		mpe := s.Members.members[mi]

		preWrite()
		if err := serializePathElementToWriter(r.reset(), mpe); err != nil {
			return err
		}
		stream.WriteObjectField(r.unsafeString())
		stream.WriteEmptyObject()
		mi++
	}

	for ci < len(s.Children.members) {
		cpe := s.Children.members[ci].pathElement

		preWrite()
		if err := serializePathElementToWriter(r.reset(), cpe); err != nil {
			return err
		}
		stream.WriteObjectField(r.unsafeString())
		stream.WriteObjectStart()
		if err := s.Children.members[ci].set.emitContents_v1(false, stream, r); err != nil {
			return err
		}
		stream.WriteObjectEnd()
		ci++
	}

	if includeSelf && !first {
		preWrite()
		stream.WriteObjectField(".")
		stream.WriteEmptyObject()
	}
	return manageMemory(stream)
}

func (s *Set) emitContents_v2(includeSelf bool, stream value.Stream, r *reusableBuilder, skipFirstComma bool) error {
	mi, ci := 0, 0
	preWrite := func() {
		if skipFirstComma {
			skipFirstComma = false
			return
		}
		stream.WriteRaw(",")
		// WriteMore flushes, which we don't want since we manage our own flushing.
		// stream.WriteMore()
	}

	for mi < len(s.Members.members) && ci < len(s.Children.members) {
		if err := manageMemory(stream); err != nil {
			return err
		}
		mpe := s.Members.members[mi]
		cpe := s.Children.members[ci].pathElement

		if mpe.Less(cpe) {
			preWrite()
			if err := serializePathElementToStreamV2(stream, mpe, etSelf); err != nil {
				return err
			}
			mi++
		} else if cpe.Less(mpe) {
			preWrite()
			if err := serializePathElementToStreamV2(stream, cpe, etChildren); err != nil {
				return err
			}
			stream.WriteRaw(",")
			stream.WriteArrayStart()
			if err := s.Children.members[ci].set.emitContents_v2(false, stream, r, true); err != nil {
				return err
			}
			stream.WriteArrayEnd()
			ci++
		} else {
			preWrite()
			if err := serializePathElementToStreamV2(stream, cpe, etBoth); err != nil {
				return err
			}
			stream.WriteRaw(",")
			stream.WriteArrayStart()
			if err := s.Children.members[ci].set.emitContents_v2(true, stream, r, true); err != nil {
				return err
			}
			stream.WriteArrayEnd()
			mi++
			ci++
		}
	}

	for mi < len(s.Members.members) {
		if err := manageMemory(stream); err != nil {
			return err
		}
		mpe := s.Members.members[mi]

		preWrite()
		if err := serializePathElementToStreamV2(stream, mpe, etSelf); err != nil {
			return err
		}
		mi++
	}

	for ci < len(s.Children.members) {
		if err := manageMemory(stream); err != nil {
			return err
		}
		cpe := s.Children.members[ci].pathElement

		preWrite()
		if err := serializePathElementToStreamV2(stream, cpe, etChildren); err != nil {
			return err
		}
		stream.WriteRaw(",")
		stream.WriteArrayStart()
		if err := s.Children.members[ci].set.emitContents_v2(false, stream, r, true); err != nil {
			return err
		}
		stream.WriteArrayEnd()
		ci++
	}
	/*
		if includeSelf && !first {
			preWrite()
			stream.WriteString(".")
		}
	*/
	return manageMemory(stream)
}

// FromJSON clears s and reads a JSON formatted set structure.
func (s *Set) FromJSON(r io.Reader) error {
	firstByte := make([]byte, 1)
	n, err := r.Read(firstByte)
	if n != 1 || err != nil {
		return err
	}

	switch firstByte[0] {
	case byte('{'):
		r = io.MultiReader(bytes.NewReader(firstByte), r)
		// The iterator pool is completely useless for memory management, grrr.
		iter := jsoniter.Parse(jsoniter.ConfigCompatibleWithStandardLibrary, r, 4096)
		found, _ := readIter_v1(iter)
		if found == nil {
			*s = Set{}
		} else {
			*s = *found
		}
		return iter.Error
	case byte('['):
		r, err = strings.NewReaderWithStringTable(r)
		if err != nil {
			return err
		}
		r = io.MultiReader(bytes.NewReader(firstByte), r)
		// The iterator pool is completely useless for memory management, grrr.
		iter := jsoniter.Parse(jsoniter.ConfigCompatibleWithStandardLibrary, r, 4096)
		found, _ := readIter_v2(iter)
		if found == nil {
			*s = Set{}
		} else {
			*s = *found
		}
		return iter.Error
	}

	return fmt.Errorf("expected object or list, got %v", firstByte[0])
}

// returns true if this subtree is also (or only) a member of parent; s is nil
// if there are no further children.
func readIter_v1(iter *jsoniter.Iterator) (children *Set, isMember bool) {
	iter.ReadMapCB(func(iter *jsoniter.Iterator, key string) bool {
		if key == "." {
			isMember = true
			iter.Skip()
			return true
		}
		pe, err := DeserializePathElement(key)
		if err == ErrUnknownPathElementType {
			// Ignore these-- a future version maybe knows what
			// they are. We drop these completely rather than try
			// to preserve things we don't understand.
			iter.Skip()
			return true
		} else if err != nil {
			iter.ReportError("parsing key as path element", err.Error())
			iter.Skip()
			return true
		}
		grandchildren, childIsMember := readIter_v1(iter)
		if childIsMember {
			if children == nil {
				children = &Set{}
			}
			m := &children.Members.members
			// Since we expect that most of the time these will have been
			// serialized in the right order, we just verify that and append.
			appendOK := len(*m) == 0 || (*m)[len(*m)-1].Less(pe)
			if appendOK {
				*m = append(*m, pe)
			} else {
				children.Members.Insert(pe)
			}
		}
		if grandchildren != nil {
			if children == nil {
				children = &Set{}
			}
			// Since we expect that most of the time these will have been
			// serialized in the right order, we just verify that and append.
			m := &children.Children.members
			appendOK := len(*m) == 0 || (*m)[len(*m)-1].pathElement.Less(pe)
			if appendOK {
				*m = append(*m, setNode{pe, grandchildren})
			} else {
				*children.Children.Descend(pe) = *grandchildren
			}
		}
		return true
	})
	if children == nil {
		isMember = true
	}

	return children, isMember
}

// returns true if this subtree is also (or only) a member of parent; s is nil
// if there are no further children.
func readIter_v2(iter *jsoniter.Iterator) (children *Set, isMember bool) {
	const (
		KT int = iota
		KEY
		BODY
	)
	step := KT
	var vt v2ValueType
	var et v2EntryType
	var pe PathElement

	doMember := func() {
		if children == nil {
			children = &Set{}
		}
		m := &children.Members.members
		// Since we expect that most of the time these will have been
		// serialized in the right order, we just verify that and append.
		appendOK := len(*m) == 0 || (*m)[len(*m)-1].Less(pe)
		if appendOK {
			*m = append(*m, pe)
		} else {
			children.Members.Insert(pe)
		}
	}

	iter.ReadArrayCB(func(iter *jsoniter.Iterator) bool {
		if step == KT {
			// first is the key type
			number := iter.ReadInt()
			if number < 0 || number >= 12 {
				iter.Error = fmt.Errorf("expected type of next entry, got %q", number)
				return false
			}
			et, vt = v2SplitTypes(number)
			step = KEY
			pe = PathElement{}
			return true
		}

		if step == KEY {
			// second is the key contents
			switch vt {
			case vtField:
				str := iter.ReadString()
				pe.FieldName = &str
			case vtValue:
				v, err := value.ReadJSONIter(iter)
				if err != nil {
					iter.Error = err
					return false
				}
				pe.Value = &v
			case vtKey:
				kvPairs := value.FieldList{}
				if next := iter.WhatIsNext(); next != jsoniter.ObjectValue {
					iter.Error = fmt.Errorf("expecting array got: %v", next)
					return false
				}
				iter.ReadObjectCB(func(iter *jsoniter.Iterator, key string) bool {
					v, err := value.ReadJSONIter(iter)
					if err != nil {
						iter.Error = err
						return false
					}
					kvPairs = append(kvPairs, value.Field{Name: key, Value: v})
					return true
				})
				pe.Key = &kvPairs
			case vtIndex:
				i := iter.ReadInt()
				pe.Index = &i
			}
			if et == etSelf {
				doMember()
				step = KT
			} else if et == etBoth {
				doMember()
				step = BODY
			} else if et == etChildren {
				step = BODY
			}
			return true
		}

		if step != BODY {
			panic("unexpected step type")
		}

		next := iter.WhatIsNext()
		if next != jsoniter.ArrayValue {
			iter.Error = fmt.Errorf("expected a list, got %v", next)
			return false
		}

		grandchildren, childIsMember := readIter_v2(iter)
		if childIsMember {
			doMember()
		}
		if grandchildren != nil {
			if children == nil {
				children = &Set{}
			}
			// Since we expect that most of the time these will have been
			// serialized in the right order, we just verify that and append.
			m := &children.Children.members
			appendOK := len(*m) == 0 || (*m)[len(*m)-1].pathElement.Less(pe)
			if appendOK {
				*m = append(*m, setNode{pe, grandchildren})
			} else {
				*children.Children.Descend(pe) = *grandchildren
			}
		}
		step = KT
		return true
	})
	if children == nil {
		//isMember = true
	}

	return children, isMember
}
