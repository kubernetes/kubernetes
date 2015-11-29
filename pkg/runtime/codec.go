/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package runtime

import (
	"bytes"
)

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(codec Codec, obj Object) string {
	bytes, err := Encode(codec, obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// Encode is a convenience wrapper for encoding to a []byte from a Serializer
func Encode(s Serializer, obj Object) ([]byte, error) {
	// TODO: reuse buffer
	buf := &bytes.Buffer{}
	if err := s.EncodeToStream(obj, buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
