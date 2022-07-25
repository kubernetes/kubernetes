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

package flocker

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/rand"

	volumehelpers "k8s.io/cloud-provider/volume/helpers"

	flockerapi "github.com/clusterhq/flocker-go"
	"k8s.io/klog/v2"
)

type flockerUtil struct{}

func (util *flockerUtil) DeleteVolume(d *flockerVolumeDeleter) error {
	var err error

	if d.flockerClient == nil {
		d.flockerClient, err = d.plugin.newFlockerClient("")
		if err != nil {
			return err
		}
	}

	datasetUUID, err := d.GetDatasetUUID()
	if err != nil {
		return err
	}

	return d.flockerClient.DeleteDataset(datasetUUID)
}

func (util *flockerUtil) CreateVolume(c *flockerVolumeProvisioner) (datasetUUID string, volumeSizeGiB int, labels map[string]string, err error) {

	if c.flockerClient == nil {
		c.flockerClient, err = c.plugin.newFlockerClient("")
		if err != nil {
			return
		}
	}

	nodes, err := c.flockerClient.ListNodes()
	if err != nil {
		return
	}
	if len(nodes) < 1 {
		err = fmt.Errorf("no nodes found inside the flocker cluster to provision a dataset")
		return
	}

	// select random node
	node := nodes[rand.Intn(len(nodes))]
	klog.V(2).Infof("selected flocker node with UUID '%s' to provision dataset", node.UUID)

	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestBytes := capacity.Value()
	volumeSizeGiB, err = volumehelpers.RoundUpToGiBInt(capacity)
	if err != nil {
		return
	}

	createOptions := &flockerapi.CreateDatasetOptions{
		MaximumSize: requestBytes,
		Metadata: map[string]string{
			"type": "k8s-dynamic-prov",
			"pvc":  c.options.PVC.Name,
		},
		Primary: node.UUID,
	}

	datasetState, err := c.flockerClient.CreateDataset(createOptions)
	if err != nil {
		return
	}
	datasetUUID = datasetState.DatasetID

	klog.V(2).Infof("successfully created Flocker dataset with UUID '%s'", datasetUUID)

	return
}
