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

package node

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	cmoptions "k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	kubeletoptions "k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

func Test_nodeWithUpdatedStatus(t *testing.T) {
	now := time.Now()
	testNode := func(d time.Duration, s api.ConditionStatus, r string) *api.Node {
		return &api.Node{
			Status: api.NodeStatus{
				Conditions: []api.NodeCondition{{
					Type: api.NodeOutOfDisk,
				}, {
					Type:               api.NodeReady,
					Status:             s,
					Reason:             r,
					Message:            "some message we don't care about here",
					LastTransitionTime: unversioned.Time{Time: now.Add(-time.Minute)},
					LastHeartbeatTime:  unversioned.Time{Time: now.Add(d)},
				}},
			},
		}
	}

	cm := cmoptions.NewCMServer()
	kubecfg := kubeletoptions.NewKubeletServer()
	assert.True(t, kubecfg.NodeStatusUpdateFrequency.Duration*3 < cm.NodeMonitorGracePeriod.Duration) // sanity check for defaults

	n := testNode(0, api.ConditionTrue, "KubeletReady")
	su := NewStatusUpdater(nil, cm.NodeMonitorPeriod.Duration, func() time.Time { return now })
	_, updated, err := su.nodeWithUpdatedStatus(n)
	assert.NoError(t, err)
	assert.False(t, updated, "no update expected b/c kubelet updated heartbeat just now")

	n = testNode(-cm.NodeMonitorGracePeriod.Duration, api.ConditionTrue, "KubeletReady")
	n2, updated, err := su.nodeWithUpdatedStatus(n)
	assert.NoError(t, err)
	assert.True(t, updated, "update expected b/c kubelet's update is older than DefaultNodeMonitorGracePeriod")
	assert.Equal(t, getCondition(&n2.Status, api.NodeReady).Reason, slaveReadyReason)
	assert.Equal(t, getCondition(&n2.Status, api.NodeReady).Message, slaveReadyMessage)

	n = testNode(-kubecfg.NodeStatusUpdateFrequency.Duration, api.ConditionTrue, "KubeletReady")
	n2, updated, err = su.nodeWithUpdatedStatus(n)
	assert.NoError(t, err)
	assert.False(t, updated, "no update expected b/c kubelet's update was missed only once")

	n = testNode(-kubecfg.NodeStatusUpdateFrequency.Duration*3, api.ConditionTrue, "KubeletReady")
	n2, updated, err = su.nodeWithUpdatedStatus(n)
	assert.NoError(t, err)
	assert.True(t, updated, "update expected b/c kubelet's update is older than 3*DefaultNodeStatusUpdateFrequency")
	assert.Equal(t, getCondition(&n2.Status, api.NodeReady).Reason, slaveReadyReason)
	assert.Equal(t, getCondition(&n2.Status, api.NodeReady).Message, slaveReadyMessage)
}
