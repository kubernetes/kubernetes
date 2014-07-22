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

package kubelet_test

import (
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	. "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
)

func TestValidatePodNoName(t *testing.T) {
	errorCases := map[string]Pod{
		// manifest is tested in api/validation_test.go, ensure it is invoked
		"empty version": {Name: "test", Manifest: api.ContainerManifest{Version: "", ID: "abc"}},

		// Name
		"zero-length name":         {Name: "", Manifest: api.ContainerManifest{Version: "v1beta1"}},
		"name > 255 characters":    {Name: strings.Repeat("a", 256), Manifest: api.ContainerManifest{Version: "v1beta1"}},
		"name not a DNS subdomain": {Name: "a.b.c.", Manifest: api.ContainerManifest{Version: "v1beta1"}},
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}
