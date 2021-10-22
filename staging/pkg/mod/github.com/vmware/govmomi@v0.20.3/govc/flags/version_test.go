/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package flags

import "testing"

func TestParseVersion(t *testing.T) {
	var v version
	var err error

	v, err = ParseVersion("5.5.5.5")
	if err != nil {
		t.Error(err)
	}

	if len(v) != 4 {
		t.Errorf("Expected %d elements, got %d", 4, len(v))
	}

	for i := 0; i < len(v); i++ {
		if v[i] != 5 {
			t.Errorf("Expected %d, got %d", 5, v[i])
		}
	}
}

func TestLte(t *testing.T) {
	v1, err := ParseVersion("5.5")
	if err != nil {
		panic(err)
	}

	v2, err := ParseVersion("5.6")
	if err != nil {
		panic(err)
	}

	if !v1.Lte(v1) {
		t.Errorf("Expected 5.5 <= 5.5")
	}

	if !v1.Lte(v2) {
		t.Errorf("Expected 5.5 <= 5.6")
	}

	if v2.Lte(v1) {
		t.Errorf("Expected not 5.6 <= 5.5")
	}
}

func TestDevelopmentVersion(t *testing.T) {
	if !isDevelopmentVersion("6.5.x") {
		t.Error("expected true")
	}

	if !isDevelopmentVersion("r4A70F") {
		t.Error("expected true")
	}

	if isDevelopmentVersion("6.5") {
		t.Error("expected false")
	}
}
