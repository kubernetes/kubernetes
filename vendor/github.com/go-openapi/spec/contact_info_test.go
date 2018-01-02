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

import (
	"testing"
)

var contactInfoJSON = `{"name":"wordnik api team","url":"http://developer.wordnik.com","email":"some@mailayada.dkdkd"}`
var contactInfoYAML = `name: wordnik api team
url: http://developer.wordnik.com
email: some@mailayada.dkdkd
`
var contactInfo = ContactInfo{
	Name:  "wordnik api team",
	URL:   "http://developer.wordnik.com",
	Email: "some@mailayada.dkdkd",
}

func TestIntegrationContactInfo(t *testing.T) {
	assertSerializeJSON(t, contactInfo, contactInfoJSON)
	assertSerializeYAML(t, contactInfo, contactInfoYAML)
	assertParsesJSON(t, contactInfoJSON, contactInfo)
	assertParsesYAML(t, contactInfoYAML, contactInfo)
}
