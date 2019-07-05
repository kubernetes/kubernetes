// Copyright 2015 go-swagger maintainers
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

package spec

// ExternalDocumentation allows referencing an external resource for
// extended documentation.
//
// For more information: http://goo.gl/8us55a#externalDocumentationObject
type ExternalDocumentation struct {
	Description string `json:"description,omitempty"`
	URL         string `json:"url,omitempty"`
}
