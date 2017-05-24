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

package nodeconfig

import (
	"encoding/json"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

const configOKType = "ConfigOK"

// fatalSyncConfigOK attempts to sync a ConfigOK status describing a fatal error.
// It is typical to call fatalf after fatalSyncConfigOK.
func (cc *NodeConfigController) fatalSyncConfigOK(cause string) {
	cc.setConfigOK("fatal-class error occurred while resolving config", cause, apiv1.ConditionFalse)
	cc.syncConfigOK()
}

// setConfigOK constructs a new ConfigOK NodeCondition from and sets it on the NodeConfigController
func (cc *NodeConfigController) setConfigOK(effect, cause string, status apiv1.ConditionStatus) {
	cc.configOK = &apiv1.NodeCondition{
		Message: effect,
		Reason:  cause,
		Status:  status,
		Type:    configOKType,
	}
}

// syncConfigOK attempts to sync `cc.configOK` with the Node object for this Kubelet.
// If syncing fails, an error is logged.
func (cc *NodeConfigController) syncConfigOK() {
	if cc.client == nil {
		infof("client was nil, skipping ConfigOK sync")
		return
	} else if cc.configOK == nil {
		infof("ConfigOK condition was nil, skipping ConfigOK sync")
		return
	}

	// get the Node so we can check the current condition
	node, err := cc.client.CoreV1().Nodes().Get(cc.nodeName, metav1.GetOptions{})
	if err != nil {
		errorf("could not get Node %q, will not sync ConfigOK condition, error: %v", cc.nodeName, err)
		return
	}

	// set timestamps
	syncTime := metav1.NewTime(time.Now())
	cc.configOK.LastHeartbeatTime = syncTime
	if c := getConfigOK(node.Status.Conditions); c != nil {
		if !configOKEq(c, cc.configOK) {
			cc.configOK.LastTransitionTime = syncTime
		}
	}

	// generate the patch
	data, err := json.Marshal(&[]apiv1.NodeCondition{*cc.configOK})
	if err != nil {
		errorf("could not serialize ConfigOK condition to JSON, condition: %+v, error: %v", cc.configOK, err)
		return
	}
	patch := []byte(fmt.Sprintf(`{"status":{"conditions":%s}}`, data))

	// update the conditions list on the Node object
	_, err = cc.client.CoreV1().Nodes().PatchStatus(cc.nodeName, patch)
	if err != nil {
		errorf("could not update ConfigOK condition, error: %v", err)
	}
}

// configOKEq returns true if the conditions' messages, reasons, and statuses match, false otherwise.
func configOKEq(a, b *apiv1.NodeCondition) bool {
	return a.Message == b.Message && a.Reason == b.Reason && a.Status == b.Status
}

// getConfigOK returns the first NodeCondition in `cs` with Type == configOKType.
// If no such condition exists, returns nil.
func getConfigOK(cs []apiv1.NodeCondition) *apiv1.NodeCondition {
	for i, _ := range cs {
		if cs[i].Type == configOKType {
			return &cs[i]
		}
	}
	return nil
}
