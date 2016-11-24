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

package cloud

import (
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/node"
	"k8s.io/kubernetes/pkg/util/wait"
)

// This test checks that the node is deleted when kubelet stops reporting
// and cloud provider says node is gone
func TestNodeDeleted(t *testing.T) {
	pod0 := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      "pod0",
		},
		Spec: api.PodSpec{
			NodeName: "node0",
		},
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{
					Type:   api.PodReady,
					Status: api.ConditionTrue,
				},
			},
		},
	}

	pod1 := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      "pod1",
		},
		Spec: api.PodSpec{
			NodeName: "node0",
		},
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{
					Type:   api.PodReady,
					Status: api.ConditionTrue,
				},
			},
		},
	}

	fnh := &node.FakeNodeHandler{
		Existing: []*api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: api.NodeStatus{
					Conditions: []api.NodeCondition{
						{
							Type:               api.NodeReady,
							Status:             api.ConditionUnknown,
							LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset:      fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*pod0, *pod1}}),
		DeleteWaitChan: make(chan struct{}),
	}

	factory := informers.NewSharedInformerFactory(fnh, controller.NoResyncPeriodFunc())

	eventBroadcaster := record.NewBroadcaster()
	cloudNodeController := &CloudNodeController{
		kubeClient:        fnh,
		nodeInformer:      factory.Nodes(),
		cloud:             &fakecloud.FakeCloud{Err: cloudprovider.InstanceNotFound},
		nodeMonitorPeriod: 5 * time.Second,
		recorder:          eventBroadcaster.NewRecorder(api.EventSource{Component: "controllermanager"}),
	}
	eventBroadcaster.StartLogging(glog.Infof)

	cloudNodeController.Run()

	select {
	case <-fnh.DeleteWaitChan:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Timed out waiting %v for node to be deleted", wait.ForeverTestTimeout)
	}
	if len(fnh.DeletedNodes) != 1 || fnh.DeletedNodes[0].Name != "node0" {
		t.Errorf("Node was not deleted")
	}
}
