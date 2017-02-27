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

package apiconfig

import (
	"encoding/json"
	"fmt"
	"time"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const apiCallRetryInterval = 500 * time.Millisecond

// TODO: Can we think of any unit tests here? Or should this code just be covered through integration/e2e tests?

// It's safe to do this for alpha, as we don't have HA and there is no way we can get
// more then one node here (TODO(phase1+) use os.Hostname)
func findMyself(client *clientset.Clientset) (*v1.Node, error) {
	nodeList, err := client.Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("unable to list nodes [%v]", err)
	}
	if len(nodeList.Items) < 1 {
		return nil, fmt.Errorf("no nodes found")
	}
	node := &nodeList.Items[0]
	return node, nil
}

func attemptToUpdateMasterRoleLabelsAndTaints(client *clientset.Clientset) error {
	n, err := findMyself(client)
	if err != nil {
		return err
	}

	oldData, err := json.Marshal(n)
	if err != nil {
		return err
	}

	// The master node is tainted and labelled accordingly
	n.ObjectMeta.Labels[kubeadmconstants.LabelNodeRoleMaster] = ""
	n.Spec.Taints = []v1.Taint{{Key: kubeadmconstants.LabelNodeRoleMaster, Value: "", Effect: "NoSchedule"}}

	newData, err := json.Marshal(n)
	if err != nil {
		return err
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
	if err != nil {
		return err
	}

	if _, err := client.Nodes().Patch(n.Name, types.StrategicMergePatchType, patchBytes); err != nil {
		if apierrs.IsConflict(err) {
			fmt.Println("[apiclient] Temporarily unable to update master node metadata due to conflict (will retry)")
			time.Sleep(apiCallRetryInterval)
			attemptToUpdateMasterRoleLabelsAndTaints(client)
		} else {
			return err
		}
	}

	return nil
}

// UpdateMasterRoleLabelsAndTaints taints the master and sets the master label
func UpdateMasterRoleLabelsAndTaints(client *clientset.Clientset) error {
	// TODO: Use iterate instead of recursion
	err := attemptToUpdateMasterRoleLabelsAndTaints(client)
	if err != nil {
		return fmt.Errorf("failed to update master node - [%v]", err)
	}
	return nil
}
