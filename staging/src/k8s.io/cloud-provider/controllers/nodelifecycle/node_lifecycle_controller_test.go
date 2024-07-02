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
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	fakecloud "k8s.io/cloud-provider/fake"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

func Test_NodesDeleted(t *testing.T) {
	createNodeTime := metav1.Now()
	createNodeTimeOut := metav1.NewTime(createNodeTime.Add(-2 * uninitializedNodeDelay))

	testcases := []struct {
		name            string
		fakeCloud       *fakecloud.Cloud
		existingNode    *v1.Node
		expectedNode    *v1.Node
		expectedDeleted bool
	}{
		{
			name: "node is not ready and does not exist",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
		},
		{
			name: "node is not ready and provider returns err",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
				ErrByProviderID:    errors.New("err!"),
			},
		},
		{
			name: "node is not ready but still exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: true,
			},
		},
		{
			name: "node ready condition is unknown, node doesn't exist",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
		},
		{
			name: "node ready condition is unknown, node exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:       false,
				ExistsByProviderID: true,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "foo://12345",
				},
			},
		},
		{
			name: "node is ready, but provider said it is deleted (maybe a bug in provider)",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
		},
		{
			name: "[instance2] node is not ready and does not exist",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: false,
			},
		},
		{
			name: "[instancev2] node is not ready and provider returns err",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: false,
				ErrByProviderID:    errors.New("err!"),
			},
		},
		{
			name: "[instancev2] node is not ready but still exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: true,
			},
		},
		{
			name: "[instancev2] node is not ready and uninitialized yet",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node-uninitialized",
					CreationTimestamp: createNodeTime,
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node-uninitialized",
					CreationTimestamp: createNodeTime,
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
			},
		},
		{
			name: "[instancev2] node is not ready condition is unknown and uninitialized timeout passed",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node-uninitialized",
					CreationTimestamp: createNodeTimeOut,
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: false,
			},
		},
		{
			name: "[instancev2] node ready condition is unknown, node doesn't exist",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: false,
			},
		},
		{
			name: "[instancev2] node ready condition is unknown, node exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				NodeShutdown:       false,
				ExistsByProviderID: true,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "foo://12345",
				},
			},
		},
		{
			name: "[instancev2] node is ready, but provider said it is deleted (maybe a bug in provider)",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2:  true,
				ExistsByProviderID: false,
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			clientset := fake.NewSimpleClientset(testcase.existingNode)
			informer := informers.NewSharedInformerFactory(clientset, time.Second)
			nodeInformer := informer.Core().V1().Nodes()

			if err := syncNodeStore(ctx, nodeInformer, clientset); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
			cloudNodeLifecycleController := &CloudNodeLifecycleController{
				nodeLister:        nodeInformer.Lister(),
				kubeClient:        clientset,
				cloud:             testcase.fakeCloud,
				recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-lifecycle-controller"}),
				nodeMonitorPeriod: 1 * time.Second,
			}

			w := eventBroadcaster.StartLogging(klog.Infof)
			defer w.Stop()
			cloudNodeLifecycleController.MonitorNodes(ctx)

			updatedNode, err := clientset.CoreV1().Nodes().Get(ctx, testcase.existingNode.Name, metav1.GetOptions{})
			if testcase.expectedDeleted != apierrors.IsNotFound(err) {
				t.Fatalf("unexpected error happens when getting the node: %v", err)
			}
			if !reflect.DeepEqual(updatedNode, testcase.expectedNode) {
				t.Logf("actual nodes: %v", updatedNode)
				t.Logf("expected nodes: %v", testcase.expectedNode)
				t.Error("unexpected updated nodes")
			}
		})
	}
}

func Test_NodesShutdown(t *testing.T) {
	testcases := []struct {
		name            string
		fakeCloud       *fakecloud.Cloud
		existingNode    *v1.Node
		expectedNode    *v1.Node
		expectedDeleted bool
	}{
		{
			name: "node is not ready and was shutdown, but exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
					Taints: []v1.Taint{
						*ShutdownTaint,
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ExistsByProviderID:      true,
				ErrShutdownByProviderID: nil,
			},
		},
		{
			name: "node with empty spec providerID is not ready and was shutdown, but exists",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
					Taints: []v1.Taint{
						*ShutdownTaint,
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ExistsByProviderID:      true,
				ErrShutdownByProviderID: nil,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "foo://12345",
				},
			},
		},
		{
			name: "node with non-existing providerID (missing in cloud provider) gets deleted",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				ErrShutdownByProviderID: nil,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "",
				},
			},
		},
		{
			name: "node with error when getting providerID does not have shutdown taint",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				ErrShutdownByProviderID: nil,
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): errors.New("err!"),
				},
			},
		},
		{
			name: "node with InstanceID returning InstanceNotFound gets deleted",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Spec: v1.NodeSpec{
					ProviderID: "",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				ErrShutdownByProviderID: nil,
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.InstanceNotFound,
				},
			},
		},
		{
			name: "node is not ready, but there is error checking if node is shutdown",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            false,
				ErrShutdownByProviderID: errors.New("err!"),
			},
		},
		{
			name: "node is not ready and is not shutdown",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            false,
				ErrShutdownByProviderID: nil,
			},
		},
		{
			name: "node is ready but provider says it's shutdown (maybe a bug by provider)",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedDeleted: false,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ErrShutdownByProviderID: nil,
			},
		},
		{
			name: "node is shutdown but provider says it does not exist",
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.Local),
						},
					},
				},
			},
			expectedNode:    &v1.Node{},
			expectedDeleted: true,
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ExistsByProviderID:      false,
				ErrShutdownByProviderID: nil,
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			clientset := fake.NewSimpleClientset(testcase.existingNode)
			informer := informers.NewSharedInformerFactory(clientset, time.Second)
			nodeInformer := informer.Core().V1().Nodes()

			if err := syncNodeStore(ctx, nodeInformer, clientset); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
			cloudNodeLifecycleController := &CloudNodeLifecycleController{
				nodeLister:        nodeInformer.Lister(),
				kubeClient:        clientset,
				cloud:             testcase.fakeCloud,
				recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-lifecycle-controller"}),
				nodeMonitorPeriod: 1 * time.Second,
			}

			w := eventBroadcaster.StartStructuredLogging(0)
			defer w.Stop()
			cloudNodeLifecycleController.MonitorNodes(ctx)

			updatedNode, err := clientset.CoreV1().Nodes().Get(ctx, testcase.existingNode.Name, metav1.GetOptions{})
			if testcase.expectedDeleted != apierrors.IsNotFound(err) {
				t.Fatalf("unexpected error happens when getting the node: %v", err)
			}
			if !reflect.DeepEqual(updatedNode, testcase.expectedNode) {
				t.Logf("actual nodes: %v", updatedNode)
				t.Logf("expected nodes: %v", testcase.expectedNode)
				t.Error("unexpected updated nodes")
			}
		})
	}
}

func Test_GetProviderID(t *testing.T) {
	testcases := []struct {
		name               string
		fakeCloud          *fakecloud.Cloud
		existingNode       *v1.Node
		expectedProviderID string
		expectedErr        error
	}{
		{
			name: "node initialized with provider ID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				Err:               nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
				},
			},
			expectedProviderID: "fake://12345",
			expectedErr:        nil,
		},
		{
			name: "node initialized with provider ID with InstancesV2",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				Err:               nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
				},
			},
			expectedProviderID: "fake://12345",
			expectedErr:        nil,
		},
		{
			name: "cloud implemented with Instances",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
			expectedProviderID: "fake://12345",
			expectedErr:        nil,
		},
		{
			name: "cloud implemented with InstancesV2",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				ProviderID: map[types.NodeName]string{
					types.NodeName("node0"): "fake://12345",
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
			expectedProviderID: "fake://12345",
			expectedErr:        nil,
		},
		{
			name: "cloud implemented with InstancesV2 (without providerID)",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				Err:               nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
			expectedProviderID: "",
			expectedErr:        nil,
		},
		{
			name: "cloud implemented with Instances with instance missing",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.InstanceNotFound,
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
			expectedProviderID: "",
			expectedErr:        cloudprovider.InstanceNotFound,
		},
		{
			name: "cloud implemented with Instances with unknown error",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): errors.New("unknown error"),
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
			expectedProviderID: "",
			expectedErr:        errors.New("failed to get instance ID from cloud provider: unknown error"),
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			cloudNodeLifecycleController := &CloudNodeLifecycleController{
				cloud: testcase.fakeCloud,
			}

			providerID, err := cloudNodeLifecycleController.getProviderID(ctx, testcase.existingNode)

			if err != nil && testcase.expectedErr == nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && testcase.expectedErr != nil {
				t.Fatalf("did not get expected error %q", testcase.expectedErr)
			}
			if err != nil && err.Error() != testcase.expectedErr.Error() {
				t.Fatalf("expected error %q, got %q", testcase.expectedErr.Error(), err.Error())
			}

			if !cmp.Equal(providerID, testcase.expectedProviderID) {
				t.Errorf("unexpected providerID %s", cmp.Diff(providerID, testcase.expectedProviderID))
			}
		})
	}
}

func syncNodeStore(ctx context.Context, nodeinformer coreinformers.NodeInformer, f *fake.Clientset) error {
	nodes, err := f.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	newElems := make([]interface{}, 0, len(nodes.Items))
	for i := range nodes.Items {
		newElems = append(newElems, &nodes.Items[i])
	}
	return nodeinformer.Informer().GetStore().Replace(newElems, "newRV")
}
