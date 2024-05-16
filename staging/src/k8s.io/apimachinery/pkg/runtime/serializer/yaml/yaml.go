/*
Copyright 2014 The Kubernetes Authors.

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

package yaml

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
)

// yamlSerializer converts YAML passed to the Decoder methods to JSON.
type yamlSerializer struct {
	// the nested serializer
	runtime.Serializer
}

// yamlSerializer implements Serializer
var _ runtime.Serializer = yamlSerializer{}

// NewDecodingSerializer adds YAML decoding support to a serializer that supports JSON.
func NewDecodingSerializer(jsonSerializer runtime.Serializer) runtime.Serializer {
	return &yamlSerializer{jsonSerializer}
}

func (c yamlSerializer) Decode(data []byte, gvk *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	out, err := yaml.ToJSON(data)
	if err != nil {
		return nil, nil, err
	}
	data = out
	return c.Serializer.Decode(data, gvk, into)
}
