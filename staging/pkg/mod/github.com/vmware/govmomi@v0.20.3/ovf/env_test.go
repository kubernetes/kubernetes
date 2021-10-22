/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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
package ovf

import "testing"

func testEnv() Env {
	return Env{
		EsxID: "vm moref",
		Platform: &PlatformSection{
			Kind:    "VMware vCenter Server",
			Version: "5.5.0",
			Vendor:  "VMware, Inc.",
			Locale:  "US",
		},
		Property: &PropertySection{
			Properties: []EnvProperty{
				{"foo", "bar"},
				{"ham", "eggs"}}},
	}
}

func TestMarshalEnv(t *testing.T) {
	env := testEnv()

	xenv, err := env.Marshal()
	if err != nil {
		t.Fatalf("error marshalling environment %s", err)
	}
	if len(xenv) < 1 {
		t.Fatal("marshalled document is empty")
	}
}

func TestMarshalManualEnv(t *testing.T) {
	env := testEnv()

	xenv := env.MarshalManual()
	if len(xenv) < 1 {
		t.Fatal("marshal document is empty")
	}
}
