//go:build linux
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

	"github.com/google/cadvisor/container/crio"
	cadvisorfs "github.com/google/cadvisor/fs"
)

func TestImageFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioImages,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no imagefs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ImageFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}

func TestContainerFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioWriteableImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   LabelCrioContainers,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no containerfs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ContainerFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}
