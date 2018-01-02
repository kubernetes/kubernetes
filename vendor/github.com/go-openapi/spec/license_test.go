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

import "testing"

func TestIntegrationLicense(t *testing.T) {
	license := License{"the name", "the url"}
	const licenseJSON = `{"name":"the name","url":"the url"}`
	const licenseYAML = "name: the name\nurl: the url\n"

	assertSerializeJSON(t, license, licenseJSON)
	assertSerializeYAML(t, license, licenseYAML)
	assertParsesJSON(t, licenseJSON, license)
	assertParsesYAML(t, licenseYAML, license)
}
