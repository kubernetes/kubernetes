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
	"io/ioutil"
	"testing"
)

func TestParseDocument(t *testing.T) {
	filename := "../examples/v2.0/yaml/petstore.yaml"
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Logf("unable to read file %s", filename)
		t.FailNow()
	}
	d, err := ParseDocument(b)
	if err != nil {
		t.Logf("%s", err.Error())
		t.FailNow()
	}
	title := "Swagger Petstore"
	if d.Info.Title != title {
		t.Errorf("unexpected value for Title: %s (expected %s)", d.Info.Title, title)
	}
}
