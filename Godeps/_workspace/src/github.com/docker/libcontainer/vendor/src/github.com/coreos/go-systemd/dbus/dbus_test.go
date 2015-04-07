/*
Copyright 2013 CoreOS Inc.

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

package dbus

import (
	"testing"
)

// TestObjectPath ensures path encoding of the systemd rules works.
func TestObjectPath(t *testing.T) {
	input := "/silly-path/to@a/unit..service"
	output := ObjectPath(input)
	expected := "/silly_2dpath/to_40a/unit_2e_2eservice"

	if string(output) != expected {
		t.Fatalf("Output '%s' did not match expected '%s'", output, expected)
	}
}

// TestNew ensures that New() works without errors.
func TestNew(t *testing.T) {
	_, err := New()

	if err != nil {
		t.Fatal(err)
	}
}
