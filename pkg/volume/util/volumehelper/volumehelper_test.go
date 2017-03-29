/*
Copyright 2017 The Kubernetes Authors.

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

// Package volumehelper contains consts and helper methods used by various
// volume components (attach/detach controller, kubelet, etc.).
package volumehelper

import (
	"testing"
)

func TestDeletingVolumeDetection(t *testing.T) {
	cases := []struct {
		volumeName       string
		expectedDeleting bool
	}{
		{
			volumeName:       ".deleting~",
			expectedDeleting: true,
		},
		{
			volumeName:       ".deleting~1234",
			expectedDeleting: true,
		},
		{
			volumeName:       "myvol.deleting~",
			expectedDeleting: true,
		},
		{
			volumeName:       "myvol.deleting~1234",
			expectedDeleting: true,
		},
		{
			volumeName:       "deleting~",
			expectedDeleting: false,
		},
		{
			volumeName:       ".deleting",
			expectedDeleting: false,
		},
		{
			volumeName:       "deleting",
			expectedDeleting: false,
		},
		{
			volumeName:       ".Deleting~",
			expectedDeleting: false,
		},
		{
			volumeName:       "myvol.deleting",
			expectedDeleting: false,
		},
		{
			volumeName:       "myvol",
			expectedDeleting: false,
		},
	}

	for _, tc := range cases {
		isDeleting := IsVolumeDeleting(tc.volumeName)
		if isDeleting != tc.expectedDeleting {
			t.Errorf("Result for volumeName %v: %v does not match expected %v", tc.volumeName, isDeleting, tc.expectedDeleting)
		}
	}
}
