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
	"io"
	"unsafe"

	jsoniter "github.com/json-iterator/go"
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

func manageMemory(stream *jsoniter.Stream) error {
	// Help jsoniter manage its buffers--without this, it does a bunch of
	// alloctaions that are not necessary. They were probably optimizing
	// for folks using the buffer directly.
	b := stream.Buffer()
	if len(b) > 4096 || cap(b)-len(b) < 2048 {
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
		stream.WriteMore()
	}

	if includeSelf && !(len(s.Members.members) == 0 && len(s.Children.members) == 0) {
		preWrite()
		stream.WriteObjectField(".")
		stream.WriteEmptyObject()
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

	return manageMemory(stream)
}

// FromJSON clears s and reads a JSON formatted set structure.
func (s *Set) FromJSON(r io.Reader) error {
	// The iterator pool is completely useless for memory management, grrr.
	iter := jsoniter.Parse(jsoniter.ConfigCompatibleWithStandardLibrary, r, 4096)

	found, _ := readIter_v1(iter)
	if found == nil {
		*s = Set{}
	} else {
		*s = *found
	}
	return iter.Error
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
