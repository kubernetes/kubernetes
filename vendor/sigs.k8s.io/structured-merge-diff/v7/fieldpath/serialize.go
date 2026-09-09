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
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"io"
	"sync"
)

func (s *Set) ToJSON() ([]byte, error) {
	return json.Marshal((*setContentsV1)(s), allowInvalidUTF8)
}

func (s *Set) ToJSONStream(w io.Writer) error {
	return json.MarshalWrite(w, (*setContentsV1)(s), allowInvalidUTF8)
}

const maxRetainedBuffer = 1024

var pool = sync.Pool{
	New: func() any {
		return &pathElementSerializer{}
	},
}

func writePathKey(enc *jsontext.Encoder, pe PathElement) error {
	serializer := pool.Get().(*pathElementSerializer)
	defer func() {
		serializer.reset()
		pool.Put(serializer)
	}()

	if err := serializer.serialize(pe); err != nil {
		return err
	}

	if err := enc.WriteToken(jsontext.String(serializer.builder.String())); err != nil {
		return err
	}
	return nil
}

type setContentsV1 Set

var _ json.MarshalerTo = (*setContentsV1)(nil)
var _ json.UnmarshalerFrom = (*setContentsV1)(nil)

func (s *setContentsV1) MarshalJSONTo(enc *jsontext.Encoder) error {
	return s.emitContentsV1(false, enc)
}

func (s *setContentsV1) emitContentsV1(includeSelf bool, enc *jsontext.Encoder) error {
	if err := enc.WriteToken(jsontext.BeginObject); err != nil {
		return err
	}

	if includeSelf && !(len(s.Members.members) == 0 && len(s.Children.members) == 0) {
		if err := enc.WriteToken(jsontext.String(".")); err != nil {
			return err
		}
		if err := writeEmptyObject(enc); err != nil {
			return err
		}
	}

	mi, ci := 0, 0
	for mi < len(s.Members.members) && ci < len(s.Children.members) {
		mpe := s.Members.members[mi]
		cpe := s.Children.members[ci].pathElement

		if c := mpe.Compare(cpe); c < 0 {
			if err := writePathKey(enc, mpe); err != nil {
				return err
			}
			if err := writeEmptyObject(enc); err != nil {
				return err
			}
			mi++
		} else if c > 0 {
			if err := writePathKey(enc, cpe); err != nil {
				return err
			}
			if err := (*setContentsV1)(s.Children.members[ci].set).emitContentsV1(false, enc); err != nil {
				return err
			}
			ci++
		} else {
			if err := writePathKey(enc, cpe); err != nil {
				return err
			}
			if err := (*setContentsV1)(s.Children.members[ci].set).emitContentsV1(true, enc); err != nil {
				return err
			}
			mi++
			ci++
		}
	}

	for mi < len(s.Members.members) {
		mpe := s.Members.members[mi]

		if err := writePathKey(enc, mpe); err != nil {
			return err
		}
		if err := writeEmptyObject(enc); err != nil {
			return err
		}

		mi++
	}

	for ci < len(s.Children.members) {
		cpe := s.Children.members[ci].pathElement

		if err := writePathKey(enc, cpe); err != nil {
			return err
		}
		if err := (*setContentsV1)(s.Children.members[ci].set).emitContentsV1(false, enc); err != nil {
			return err
		}

		ci++
	}

	if err := enc.WriteToken(jsontext.EndObject); err != nil {
		return err
	}

	return nil
}

func (s *setContentsV1) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	found, _, err := readIterV1(dec)
	if err != nil {
		return err
	} else if found == nil {
		*(*Set)(s) = Set{}
	} else {
		*(*Set)(s) = *found
	}
	return nil
}

// FromJSON clears s and reads a JSON formatted set structure.
func (s *Set) FromJSON(r io.Reader) error {
	return json.UnmarshalRead(r, (*setContentsV1)(s), allowInvalidUTF8, allowDuplicates)
}

// returns true if this subtree is also (or only) a member of parent; s is nil
// if there are no further children.
func readIterV1(parser *jsontext.Decoder) (children *Set, isMember bool, err error) {
	objStart, err := parser.ReadToken()
	if err != nil {
		return nil, false, fmt.Errorf("parsing JSON: %v", err)
	}
	switch objStart.Kind() {
	case jsontext.BeginObject.Kind():
		// Continue below.
	case jsontext.Null.Kind():
		// A null is equivalent to an empty object: it contributes no
		// children, and is a member of its parent (if any).
		return nil, true, nil
	default:
		return nil, false, fmt.Errorf("expected object")
	}

	for {
		rawKey, err := parser.ReadToken()
		if err == io.EOF {
			return nil, false, fmt.Errorf("unexpected EOF")
		} else if err != nil {
			return nil, false, fmt.Errorf("parsing JSON: %v", err)
		}

		if rawKey.Kind() == jsontext.EndObject.Kind() {
			break
		}

		k := rawKey.String()
		if k == "." {
			isMember = true
			if err := parser.SkipValue(); err != nil {
				return nil, false, fmt.Errorf("parsing JSON: %v", err)
			}
			continue
		}
		pe, err := DeserializePathElement(k)
		if err == ErrUnknownPathElementType {
			// Ignore these-- a future version maybe knows what
			// they are. We drop these completely rather than try
			// to preserve things we don't understand.
			if err := parser.SkipValue(); err != nil {
				return nil, false, fmt.Errorf("parsing JSON: %v", err)
			}
			continue
		} else if err != nil {
			return nil, false, fmt.Errorf("parsing key as path element: %v", err)
		}

		grandchildren, childIsMember, err := readIterV1(parser)
		if err != nil {
			return nil, false, fmt.Errorf("parsing value as set: %v", err)
		}

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
	}

	if children == nil {
		isMember = true
	}

	return children, isMember, nil
}

func writeEmptyObject(enc *jsontext.Encoder) error {
	if err := enc.WriteToken(jsontext.BeginObject); err != nil {
		return err
	}
	return enc.WriteToken(jsontext.EndObject)
}
