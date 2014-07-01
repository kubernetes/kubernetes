/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"strings"
	"testing"
)

func TestValidateManifest(t *testing.T) {
	successCases := []ContainerManifest{
		{Version: "v1beta1", ID: "abc"},
		{Version: "v1beta1", ID: "123"},
		{Version: "v1beta1", ID: "abc.123.do-re-mi"},
	}
	for _, manifest := range successCases {
		err := ValidateManifest(&manifest)
		if err != nil {
			t.Errorf("expected success: %v", err)
		}
	}

	errorCases := map[string]ContainerManifest{
		"empty version":          {Version: "", ID: "abc"},
		"invalid version":        {Version: "bogus", ID: "abc"},
		"zero-length id":         {Version: "v1beta1", ID: ""},
		"id > 255 characters":    {Version: "v1beta1", ID: strings.Repeat("a", 256)},
		"id not a DNS subdomain": {Version: "v1beta1", ID: "a.b.c."},
	}
	for k, v := range errorCases {
		err := ValidateManifest(&v)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}
