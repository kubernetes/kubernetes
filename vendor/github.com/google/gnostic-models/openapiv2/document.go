// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openapi_v2

import (
	yaml "go.yaml.in/yaml/v3"

	"github.com/google/gnostic-models/compiler"
)

// ParseDocument reads an OpenAPI v2 description from a YAML/JSON representation.
func ParseDocument(b []byte) (*Document, error) {
	info, err := compiler.ReadInfoFromBytes("", b)
	if err != nil {
		return nil, err
	}
	root := info.Content[0]
	return NewDocument(root, compiler.NewContextWithExtensions("$root", root, nil, nil))
}

// YAMLValue produces a serialized YAML representation of the document.
func (d *Document) YAMLValue(comment string) ([]byte, error) {
	rawInfo := d.ToRawInfo()
	rawInfo = &yaml.Node{
		Kind:        yaml.DocumentNode,
		Content:     []*yaml.Node{rawInfo},
		HeadComment: comment,
	}
	return yaml.Marshal(rawInfo)
}
