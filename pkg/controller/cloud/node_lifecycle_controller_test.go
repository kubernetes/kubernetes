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
	"errors"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	fakecloud "k8s.io/cloud-provider/fake"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

func Test_NodesDeleted(t *testing.T) {
	testcases := []struct {
		name        string
		fnh         *testutil.FakeNodeHandler
		fakeCloud   *fakecloud.Cloud
		deleteNodes []*v1.Node
	}{
		{
			name: "node is not ready and does not exist",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
			deleteNodes: []*v1.Node{
				testutil.NewNode("node0"),
			},
		},
		{
			name: "node is not ready and provider returns err",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
				ErrByProviderID:    errors.New("err!"),
			},
			deleteNodes: []*v1.Node{},
		},
		{
			name: "node is not ready but still exists",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: true,
			},
			deleteNodes: []*v1.Node{},
		},
		{
			name: "node ready condition is unknown, node doesn't exist",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
			deleteNodes: []*v1.Node{
				testutil.NewNode("node0"),
			},
		},
		{
			name: "node ready condition is unknown, node exists",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:       false,
				ExistsByProviderID: true,
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "foo://12345",
				},
			},
			deleteNodes: []*v1.Node{},
		},
		{
			name: "node is ready, but provider said it is deleted (maybe a bug in provider)",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				DeletedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				ExistsByProviderID: false,
			},
			deleteNodes: []*v1.Node{},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			informer := informers.NewSharedInformerFactory(testcase.fnh.Clientset, time.Second)
			nodeInformer := informer.Core().V1().Nodes()

			if err := syncNodeStore(nodeInformer, testcase.fnh); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			eventBroadcaster := record.NewBroadcaster()
			cloudNodeLifecycleController := &CloudNodeLifecycleController{
				nodeLister:        nodeInformer.Lister(),
				kubeClient:        testcase.fnh,
				cloud:             testcase.fakeCloud,
				recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-lifecycle-controller"}),
				nodeMonitorPeriod: 1 * time.Second,
			}

			eventBroadcaster.StartLogging(klog.Infof)
			cloudNodeLifecycleController.MonitorNodes()

			if !reflect.DeepEqual(testcase.fnh.DeletedNodes, testcase.deleteNodes) {
				t.Logf("actual nodes: %v", testcase.fnh.DeletedNodes)
				t.Logf("expected nodes: %v", testcase.deleteNodes)
				t.Error("unexpected deleted nodes")
			}
		})
	}
}

func Test_NodesShutdown(t *testing.T) {
	testcases := []struct {
		name         string
		fnh          *testutil.FakeNodeHandler
		fakeCloud    *fakecloud.Cloud
		updatedNodes []*v1.Node
	}{
		{
			name: "node is not ready and was shutdown",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				UpdatedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ErrShutdownByProviderID: nil,
			},
			updatedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.Local),
					},
					Spec: v1.NodeSpec{
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
			},
		},
		{
			name: "node is not ready, but there is error checking if node is shutdown",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				UpdatedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            false,
				ErrShutdownByProviderID: errors.New("err!"),
			},
			updatedNodes: []*v1.Node{},
		},
		{
			name: "node is not ready and is not shutdown",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				UpdatedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            false,
				ErrShutdownByProviderID: nil,
			},
			updatedNodes: []*v1.Node{},
		},
		{
			name: "node is ready but provider says it's shutdown (maybe a bug by provider)",
			fnh: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
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
				},
				UpdatedNodes: []*v1.Node{},
				Clientset:    fake.NewSimpleClientset(),
			},
			fakeCloud: &fakecloud.Cloud{
				NodeShutdown:            true,
				ErrShutdownByProviderID: nil,
			},
			updatedNodes: []*v1.Node{},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			informer := informers.NewSharedInformerFactory(testcase.fnh.Clientset, time.Second)
			nodeInformer := informer.Core().V1().Nodes()

			if err := syncNodeStore(nodeInformer, testcase.fnh); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			eventBroadcaster := record.NewBroadcaster()
			cloudNodeLifecycleController := &CloudNodeLifecycleController{
				nodeLister:        nodeInformer.Lister(),
				kubeClient:        testcase.fnh,
				cloud:             testcase.fakeCloud,
				recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-lifecycle-controller"}),
				nodeMonitorPeriod: 1 * time.Second,
			}

			eventBroadcaster.StartLogging(klog.Infof)
			cloudNodeLifecycleController.MonitorNodes()

			if !reflect.DeepEqual(testcase.fnh.UpdatedNodes, testcase.updatedNodes) {
				t.Logf("actual nodes: %v", testcase.fnh.UpdatedNodes)
				t.Logf("expected nodes: %v", testcase.updatedNodes)
				t.Error("unexpected updated nodes")
			}
		})
	}
}

func syncNodeStore(nodeinformer coreinformers.NodeInformer, f *testutil.FakeNodeHandler) error {
	nodes, err := f.List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	newElems := make([]interface{}, 0, len(nodes.Items))
	for i := range nodes.Items {
		newElems = append(newElems, &nodes.Items[i])
	}
	return nodeinformer.Informer().GetStore().Replace(newElems, "newRV")
}
