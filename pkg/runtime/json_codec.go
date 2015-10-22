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
	"encoding/json"
	"fmt"
	"io"
)

type JSONCodec struct{}

func (JSONCodec) Encode(obj Object) (data []byte, err error) {
	return nil, fmt.Errorf("JSONCodec does not support method Encode\n")
}

func (JSONCodec) EncodeToStream(obj Object, stream io.Writer) error {
	return fmt.Errorf("JSONCodec does not support method EncodeToStream\n")
}

func (JSONCodec) Decode(data []byte) (Object, error) {

	return nil, fmt.Errorf("JSONCodec does not support method Decode\n")
}
func (JSONCodec) DecodeToVersion(data []byte, version string) (Object, error) {

	return nil, fmt.Errorf("JSONCodec does not support method DecodeToVersion\n")
}
func (JSONCodec) DecodeInto(data []byte, obj Object) error {
	return json.Unmarshal(data, obj)
}
func (JSONCodec) DecodeIntoWithSpecifiedVersionKind(data []byte, obj Object, kind, version string) error {
	return fmt.Errorf("JSONCodec does not support method DecodeIntoWithSpecifiedVersionKind\n")
}
