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

package discovery_v1

import (
	"io/ioutil"
	"testing"
)

func TestParseDocument(t *testing.T) {
	filename := "../examples/discovery/discovery-v1.json"
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
	// expected values
	name := "discovery"
	version := "v1"
	title := "API Discovery Service"
	// check actual values
	if d.Name != name {
		t.Errorf("unexpected value for Name: %s (expected %s)", d.Name, name)
	}
	if d.Version != version {
		t.Errorf("unexpected value for Version: %s (expected %s)", d.Version, version)
	}
	if d.Title != title {
		t.Errorf("unexpected value for Title: %s (expected %s)", d.Title, title)
	}
}
