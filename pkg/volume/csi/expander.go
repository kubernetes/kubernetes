/*
Copyright 2019 The Kubernetes Authors.

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

package csi

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume"
)

func (c *csiPlugin) RequiresFSResize() bool {
	// We could check plugin's node capability but we instead are going to rely on
	// NodeExpand to do the right thing and return early if plugin does not have
	// node expansion capability.
	return true
}

func (c *csiPlugin) NodeExpand(spec *volume.Spec, devicePath, deviceMountPath string, newSize, oldSize resource.Quantity) (bool, error) {
	klog.V(4).Infof(log("Expander.NodeExpand(%s)", deviceMountPath))
	pvSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		return false, err
	}
	k8s := c.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("failed to get a kubernetes client"))
		return false, errors.New("failed to get a Kubernetes client")
	}

	csiClient, err := newCsiDriverClient(csiDriverName(pvSource.Driver))
	if err != nil {
		return false, err
	}

	csiSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		klog.Error(log("Expander.NodeExpand failed to get CSI persistent source: %v", err))
		return false, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	nodeExpandSet, err := csiClient.NodeSupportsNodeExpand(ctx)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to check if node supports expansion : %v", err)
	}
	return false, nil
}
