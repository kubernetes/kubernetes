/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package codec

import (
	"encoding/json"
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/runtime"
)

type JSONCodec struct{}

// Encode implements runtime.Encoder
func (JSONCodec) Encode(obj runtime.Object) (data []byte, err error) {
	return json.Marshal(obj)
}

// EncodeToStream implements runtime.Encoder
func (JSONCodec) EncodeToStream(obj runtime.Object, stream io.Writer) error {
	encoder := json.NewEncoder(stream)
	return encoder.Encode(obj)
}

// Decode implements runtime.Decoder
func (JSONCodec) Decode(data []byte) (runtime.Object, error) {
	return nil, fmt.Errorf("JSONCodec does not support method Decode\n")
}

// DecodeToVersion implements runtime.Decoder
func (JSONCodec) DecodeToVersion(data []byte, version string) (runtime.Object, error) {
	return nil, fmt.Errorf("JSONCodec does not support method DecodeToVersion\n")
}

// DecodeInto implements runtime.Decoder
func (JSONCodec) DecodeInto(data []byte, obj runtime.Object) error {
	return json.Unmarshal(data, obj)
}

// DecodeIntoWithSpecifiedVersionKind implements runtime.Decoder
func (JSONCodec) DecodeIntoWithSpecifiedVersionKind(data []byte, obj runtime.Object, kind, version string) error {
	return fmt.Errorf("JSONCodec does not support method DecodeIntoWithSpecifiedVersionKind\n")
}
