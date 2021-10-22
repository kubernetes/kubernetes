// Copyright 2019 Google LLC. All Rights Reserved.
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

package discovery_v1

import (
	"github.com/googleapis/gnostic/compiler"
)

// FetchDocumentBytes downloads the bytes of a discovery document from a URL.
func FetchDocumentBytes(documentURL string) ([]byte, error) {
	return compiler.FetchFile(documentURL)
}

// ParseDocument reads a Discovery description from a YAML/JSON representation.
func ParseDocument(b []byte) (*Document, error) {
	info, err := compiler.ReadInfoFromBytes("", b)
	if err != nil {
		return nil, err
	}
	root := info.Content[0]
	return NewDocument(root, compiler.NewContext("$root", root, nil))
}
