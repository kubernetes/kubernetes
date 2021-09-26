// Copyright 2019 Google Inc. All Rights Reserved.
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
	"errors"
	"log"

	"github.com/googleapis/gnostic/compiler"
)

func FetchDocumentBytes(documentURL string) ([]byte, error) {
	return compiler.FetchFile(documentURL)
}

func ParseDocument(bytes []byte) (*Document, error) {
	// Unpack the discovery document.
	info, err := compiler.ReadInfoFromBytes("", bytes)
	if err != nil {
		return nil, err
	}
	m, ok := compiler.UnpackMap(info)
	if !ok {
		log.Printf("%s", string(bytes))
		return nil, errors.New("Invalid input")
	}
	return NewDocument(m, compiler.NewContext("$root", nil))
}
