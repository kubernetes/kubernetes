/*
Copyright 2014 Google Inc. All rights reserved.

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

package conversion

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func (s *Scheme) EncodeOrDie(obj interface{}) string {
	bytes, err := s.Encode(obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// Encode turns the given api object into an appropriate JSON string.
// Obj may be a pointer to a struct, or a struct. If a struct, a copy
// will be made, therefore it's recommended to pass a pointer to a
// struct. The type must have been registered.
//
// Memory/wire format differences:
//  * Having to keep track of the Kind and APIVersion fields makes tests
//    very annoying, so the rule is that they are set only in wire format
//    (json), not when in native (memory) format. This is possible because
//    both pieces of information are implicit in the go typed object.
//     * An exception: note that, if there are embedded API objects of known
//       type, for example, PodList{... Items []Pod ...}, these embedded
//       objects must be of the same version of the object they are embedded
//       within, and their APIVersion and Kind must both be empty.
//     * Note that the exception does not apply to the APIObject type, which
//       recursively does Encode()/Decode(), and is capable of expressing any
//       API object.
//  * Only versioned objects should be encoded. This means that, if you pass
//    a native object, Encode will convert it to a versioned object. For
//    example, an api.Pod will get converted to a v1beta1.Pod. However, if
//    you pass in an object that's already versioned (v1beta1.Pod), Encode
//    will not modify it.
//
// The purpose of the above complex conversion behavior is to allow us to
// change the memory format yet not break compatibility with any stored
// objects, whether they be in our storage layer (e.g., etcd), or in user's
// config files.
//
func (s *Scheme) Encode(obj interface{}) (data []byte, err error) {
	return s.EncodeToVersion(obj, s.ExternalVersion)
}

// EncodeToVersion is like Encode, but you may choose the version.
func (s *Scheme) EncodeToVersion(obj interface{}, destVersion string) (data []byte, err error) {
	obj = maybeCopy(obj)
	v, _ := enforcePtr(obj) // maybeCopy guarantees a pointer
	if _, registered := s.typeToVersion[v.Type()]; !registered {
		return nil, fmt.Errorf("type %v is not registered and it will be impossible to Decode it, therefore Encode will refuse to encode it.", v.Type())
	}

	objVersion, objKind, err := s.ObjectAPIVersionAndKind(obj)
	if err != nil {
		return nil, err
	}

	// Perform a conversion if necessary.
	if objVersion != destVersion {
		objOut, err := s.NewObject(destVersion, objKind)
		if err != nil {
			return nil, err
		}
		err = s.converter.Convert(obj, objOut, 0)
		if err != nil {
			return nil, err
		}
		obj = objOut
	}

	// Version and Kind should be set on the wire.
	setVersionAndKind := s.MetaInsertionFactory.Create(destVersion, objKind)
	err = s.converter.Convert(setVersionAndKind, obj, SourceToDest|IgnoreMissingFields|AllowDifferentFieldNames)
	if err != nil {
		return nil, err
	}

	// To add metadata, do some simple surgery on the JSON.
	data, err = json.Marshal(obj)
	if err != nil {
		return nil, err
	}

	// Version and Kind should be blank in memory.
	blankVersionAndKind := s.MetaInsertionFactory.Create("", "")
	err = s.converter.Convert(blankVersionAndKind, obj, SourceToDest|IgnoreMissingFields|AllowDifferentFieldNames)
	if err != nil {
		return nil, err
	}

	return data, nil

	meta, err := json.Marshal(s.MetaInsertionFactory.Create(destVersion, objKind))
	if err != nil {
		return nil, err
	}
	// Stick these together, omitting the last } from meta and the first { from
	// data. Add a comma to meta if necessary.
	metaN := len(meta)
	if len(data) > 2 {
		meta[metaN-1] = ',' // Add comma
	} else {
		meta = meta[:metaN-1] // Just remove }
	}
	together := append(meta, data[1:]...)
	if s.Indent {
		var out bytes.Buffer
		err := json.Indent(&out, together, "", "	")
		if err != nil {
			return nil, err
		}
		return out.Bytes(), nil
	}
	return together, nil
}
