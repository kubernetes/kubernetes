/*
Copyright 2018 The Kubernetes Authors.

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

package patchnode

import (
	"fmt"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// AnnotateCRISocket annotates the node with the given crisocket
func AnnotateCRISocket(client clientset.Interface, nodeName string, criSocket string) error {

	fmt.Printf("[patchnode] Uploading the CRI Socket information %q to the Node API object %q as an annotation\n", criSocket, nodeName)

	return apiclient.PatchNode(client, nodeName, func(n *v1.Node) {
		annotateNodeWithCRISocket(n, criSocket)
	})
}

func annotateNodeWithCRISocket(n *v1.Node, criSocket string) {
	if n.ObjectMeta.Annotations == nil {
		n.ObjectMeta.Annotations = make(map[string]string)
	}
	n.ObjectMeta.Annotations[constants.AnnotationKubeadmCRISocket] = criSocket
}
