// +build linux

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

package cadvisor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	cadvisorfs "github.com/google/cadvisor/fs"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

func TestImageFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelDockerImages should be returned",
		runtime:         types.DockerContainerRuntime,
		runtimeEndpoint: "",
		expectedLabel:   cadvisorfs.LabelDockerImages,
		expectedError:   nil,
	}, {
		description:     "LabelCrioImages should be returned",
		runtime:         types.RemoteContainerRuntime,
		runtimeEndpoint: CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioImages,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtime:         "invalid-runtime",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no imagefs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtime, tc.runtimeEndpoint)
			label, err := infoProvider.ImageFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}
