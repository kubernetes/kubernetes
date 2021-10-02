/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSetDefaults_Config(t *testing.T) {
	tests := []struct {
		name        string
		in, wantOut *ExecConfig
	}{
		{
			name: "alpha exec API with empty interactive mode",
			in:   &ExecConfig{APIVersion: "client.authentication.k8s.io/v1alpha1"},
			wantOut: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1alpha1",
				InteractiveMode: IfAvailableExecInteractiveMode,
			},
		},
		{
			name: "beta exec API with empty interactive mode",
			in:   &ExecConfig{APIVersion: "client.authentication.k8s.io/v1beta1"},
			wantOut: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: IfAvailableExecInteractiveMode,
			},
		},
		{
			name: "alpha exec API with set interactive mode",
			in: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1alpha1",
				InteractiveMode: NeverExecInteractiveMode,
			},
			wantOut: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1alpha1",
				InteractiveMode: NeverExecInteractiveMode,
			},
		},
		{
			name: "beta exec API with set interactive mode",
			in: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: NeverExecInteractiveMode,
			},
			wantOut: &ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: NeverExecInteractiveMode,
			},
		},
		{
			name:    "v1 exec API with empty interactive mode",
			in:      &ExecConfig{APIVersion: "client.authentication.k8s.io/v1"},
			wantOut: &ExecConfig{APIVersion: "client.authentication.k8s.io/v1"},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			gotOut := test.in.DeepCopy()
			SetDefaults_ExecConfig(gotOut)
			if diff := cmp.Diff(test.wantOut, gotOut); diff != "" {
				t.Errorf("unexpected defaulting; -want, +got:\n %s", diff)
			}
		})
	}
}
