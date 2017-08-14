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

package digitalocean

import (
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

func getVolumeSource(spec *volume.Spec) (*v1.DOVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.DOVolume != nil {
		return spec.Volume.DOVolume, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.DOVolume != nil {
		return spec.PersistentVolume.Spec.DOVolume, nil
	}

	return nil, fmt.Errorf("Spec does not reference a Digital Ocean volume type")
}

// nodeFromName takes a node name and returns the Node object
func nodeFromName(host volume.VolumeHost, nodeName types.NodeName) (*v1.Node, error) {

	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("Cannot get kube client")
	}

	node, err := kubeClient.Core().Nodes().Get(string(nodeName), metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return node, nil
}

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(doVolumePluginName), volName)
}
