/*
Copyright 2015 The Kubernetes Authors.

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

package flocker

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/stretchr/testify/assert"
)

func TestFlockerUtil_CreateVolume(t *testing.T) {
	assert := assert.New(t)

	// test CreateVolume happy path
	pvc := volumetest.CreateTestPVC("3Gi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce})
	options := volume.VolumeOptions{
		PVC: pvc,
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}

	fakeFlockerClient := newFakeFlockerClient()
	dir, p := newTestableProvisioner(assert, options)
	provisioner := p.(*flockerVolumeProvisioner)
	defer os.RemoveAll(dir)
	provisioner.flockerClient = fakeFlockerClient

	flockerUtil := &FlockerUtil{}

	datasetID, size, _, err := flockerUtil.CreateVolume(provisioner)
	assert.NoError(err)
	assert.Equal(datasetOneID, datasetID)
	assert.Equal(3, size)

	// test error during CreateVolume
	fakeFlockerClient.Error = fmt.Errorf("Do not feel like provisioning")
	_, _, _, err = flockerUtil.CreateVolume(provisioner)
	assert.Equal(fakeFlockerClient.Error.Error(), err.Error())
}
