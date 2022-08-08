//go:build !providerless && windows
// +build !providerless,windows

/*
Copyright 2022 The Kubernetes Authors.

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

package vsphere_volume

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormatIfNotFormatted(t *testing.T) {
	// If this volume has already been mounted then
	// its devicePath will have already been converted to a disk number,
	// meaning that the original path is returned.
	devPath, err := verifyDevicePath("foo")
	require.NoError(t, err)
	assert.Equal(t, "foo", devPath)

	// Won't match any serial number, meaning that an error will be returned.
	devPath, err = verifyDevicePath(diskByIDPath + diskSCSIPrefix + "fake-serial")
	expectedErrMsg := `unable to find vSphere disk with serial fake-serial`
	if err == nil || err.Error() != expectedErrMsg {
		t.Errorf("expected error message `%s` but got `%v`", expectedErrMsg, err)
	}
}
