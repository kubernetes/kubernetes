// Copyright 2017 Google LLC. All Rights Reserved.
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

package main

import (
	openapi "github.com/googleapis/gnostic/openapiv3"
        plugins "github.com/googleapis/gnostic/plugins"
)

// DocumentLinter contains information collected about an API description.
type DocumentLinterV3 struct {
}

func (d *DocumentLinterV3) Run() []*plugins.Message {
	return nil
}

// NewDocumentLinter builds a new DocumentLinter object.
func NewDocumentLinterV3(document *openapi.Document) *DocumentLinterV3 {
	return &DocumentLinterV3{}
}
