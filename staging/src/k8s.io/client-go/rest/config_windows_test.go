/*
Copyright 2016 The Kubernetes Authors.

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

package rest

import (
	"os"
	"testing"
)

func Test_getPath(t *testing.T) {
	tests := []struct {
		name          string
		wantToken     string
		wantCaCert    string
		mountpointEnv string
	}{
		{
			name:          "when env set it prepends",
			wantToken:     "c:/random/var/run/secrets/kubernetes.io/serviceaccount/token",
			wantCaCert:    "c:/random/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
			mountpointEnv: "c:/random",
		},
		{
			name:          "when env set it returns same value",
			wantToken:     "/var/run/secrets/kubernetes.io/serviceaccount/token",
			wantCaCert:    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
			mountpointEnv: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mountpointEnv != "" {
				os.Setenv("CONTAINER_SANDBOX_MOUNT_POINT", tt.mountpointEnv)
				defer os.Unsetenv("CONTAINER_SANDBOX_MOUNT_POINT")
			}

			if gotToken, gotCaCert := getServiceAccountFilePaths(); gotToken != tt.wantToken || gotCaCert != tt.wantCaCert {
				t.Errorf("getServiceAccountFilePaths() = %v, %v, want %v, %v", gotToken, gotCaCert, tt.wantToken, tt.wantCaCert)
			}
		})
	}
}
