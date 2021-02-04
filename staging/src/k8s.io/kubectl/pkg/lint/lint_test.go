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

package lint

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIsRunningAsRoot(t *testing.T) {
	tests := []struct {
		runAsUser   interface{}
		unit        string
		name        string
		want        string
		description string
	}{
		{0, "Pod", "fakepod", "Pod fakepod is running as root", "pod_runnning_as_root"},
		{0, "Pod/fakepod/container", "fakecontainer", "Pod/fakepod/container fakecontainer is running as root", "container_running_as_root"},
		{1000, "", "", "", "no_resources_running_as_root"},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			got := isRunningAsRoot(test.runAsUser, test.unit, test.name)
			require.Equal(t, test.want, got)
		})
	}
}
