/*
Copyright 2026 The Kubernetes Authors.

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

package templates

import "testing"

func TestExampleCommands(t *testing.T) {
	tests := []struct {
		name    string
		example string
		want    string
	}{
		{
			name:    "empty",
			example: "",
			want:    "",
		},
		{
			name: "indentation is removed, comments and blank lines are kept",
			example: Examples(`
		# Create a pod using the data in pod.json
		kubectl create -f ./pod.json

		# Create a pod based on the JSON passed into stdin
		cat pod.json | kubectl create -f -`),
			want: "# Create a pod using the data in pod.json\nkubectl create -f ./pod.json\n\n# Create a pod based on the JSON passed into stdin\ncat pod.json | kubectl create -f -",
		},
		{
			name:    "commands only",
			example: "  kubectl get pods\n  kubectl get pods -o wide\n",
			want:    "kubectl get pods\nkubectl get pods -o wide",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := exampleCommands(tt.example); got != tt.want {
				t.Errorf("exampleCommands() = %q, want %q", got, tt.want)
			}
		})
	}
}
