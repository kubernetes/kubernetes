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

package node

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/clock"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeNodeHandler is a fake implementation of NodesInterface and NodeInterface. It
// allows test cases to have fine-grained control over mock behaviors. We also need
// PodsInterface and PodInterface to test list & delet pods, which is implemented in
// the embedded client.Fake field.
type FakeNodeHandler struct {
	*fake.Clientset

	// Input: Hooks determine if request is valid or not
	CreateHook func(*FakeNodeHandler, *v1.Node) bool
	Existing   []*v1.Node

	// Output
	CreatedNodes        []*v1.Node
	DeletedNodes        []*v1.Node
	UpdatedNodes        []*v1.Node
	UpdatedNodeStatuses []*v1.Node
	RequestCount        int

	// Synchronization
	lock           sync.Mutex
	deleteWaitChan chan struct{}
}

type FakeLegacyHandler struct {
	v1core.CoreV1Interface
	n *FakeNodeHandler
}

func (c *FakeNodeHandler) getUpdatedNodesCopy() []*v1.Node {
	c.lock.Lock()
	defer c.lock.Unlock()
	updatedNodesCopy := make([]*v1.Node, len(c.UpdatedNodes), len(c.UpdatedNodes))
	for i, ptr := range c.UpdatedNodes {
		updatedNodesCopy[i] = ptr
	}
	return updatedNodesCopy
}

func (c *FakeNodeHandler) Core() v1core.CoreV1Interface {
	return &FakeLegacyHandler{c.Clientset.Core(), c}
}

func (m *FakeLegacyHandler) Nodes() v1core.NodeInterface {
	return m.n
}

func (m *FakeNodeHandler) Create(node *v1.Node) (*v1.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	for _, n := range m.Existing {
		if n.Name == node.Name {
			return nil, apierrors.NewAlreadyExists(api.Resource("nodes"), node.Name)
		}
	}
	if m.CreateHook == nil || m.CreateHook(m, node) {
		nodeCopy := *node
		m.CreatedNodes = append(m.CreatedNodes, &nodeCopy)
		return node, nil
	} else {
		return nil, errors.New("Create error.")
	}
}

func (m *FakeNodeHandler) Get(name string) (*v1.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	for i := range m.UpdatedNodes {
		if m.UpdatedNodes[i].Name == name {
			nodeCopy := *m.UpdatedNodes[i]
			return &nodeCopy, nil
		}
	}
	for i := range m.Existing {
		if m.Existing[i].Name == name {
			nodeCopy := *m.Existing[i]
			return &nodeCopy, nil
		}
	}
	return nil, nil
}

func (m *FakeNodeHandler) List(opts v1.ListOptions) (*v1.NodeList, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	var nodes []*v1.Node
	for i := 0; i < len(m.UpdatedNodes); i++ {
		if !contains(m.UpdatedNodes[i], m.DeletedNodes) {
			nodes = append(nodes, m.UpdatedNodes[i])
		}
	}
	for i := 0; i < len(m.Existing); i++ {
		if !contains(m.Existing[i], m.DeletedNodes) && !contains(m.Existing[i], nodes) {
			nodes = append(nodes, m.Existing[i])
		}
	}
	for i := 0; i < len(m.CreatedNodes); i++ {
		if !contains(m.CreatedNodes[i], m.DeletedNodes) && !contains(m.CreatedNodes[i], nodes) {
			nodes = append(nodes, m.CreatedNodes[i])
		}
	}
	nodeList := &v1.NodeList{}
	for _, node := range nodes {
		nodeList.Items = append(nodeList.Items, *node)
	}
	return nodeList, nil
}

func (m *FakeNodeHandler) Delete(id string, opt *v1.DeleteOptions) error {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		if m.deleteWaitChan != nil {
			m.deleteWaitChan <- struct{}{}
		}
		m.lock.Unlock()
	}()
	m.DeletedNodes = append(m.DeletedNodes, newNode(id))
	return nil
}

func (m *FakeNodeHandler) DeleteCollection(opt *v1.DeleteOptions, listOpts v1.ListOptions) error {
	return nil
}

func (m *FakeNodeHandler) Update(node *v1.Node) (*v1.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	nodeCopy := *node
	for i, updateNode := range m.UpdatedNodes {
		if updateNode.Name == nodeCopy.Name {
			m.UpdatedNodes[i] = &nodeCopy
			return node, nil
		}
	}
	m.UpdatedNodes = append(m.UpdatedNodes, &nodeCopy)
	return node, nil
}

func (m *FakeNodeHandler) UpdateStatus(node *v1.Node) (*v1.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	nodeCopy := *node
	m.UpdatedNodeStatuses = append(m.UpdatedNodeStatuses, &nodeCopy)
	return node, nil
}

func (m *FakeNodeHandler) PatchStatus(nodeName string, data []byte) (*v1.Node, error) {
	m.RequestCount++
	return &v1.Node{}, nil
}

func (m *FakeNodeHandler) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return watch.NewFake(), nil
}

func (m *FakeNodeHandler) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (*v1.Node, error) {
	return nil, nil
}

// FakeRecorder is used as a fake during testing.
type FakeRecorder struct {
	source v1.EventSource
	events []*v1.Event
	clock  clock.Clock
}

func (f *FakeRecorder) Event(obj runtime.Object, eventtype, reason, message string) {
	f.generateEvent(obj, metav1.Now(), eventtype, reason, message)
}

func (f *FakeRecorder) Eventf(obj runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	f.Event(obj, eventtype, reason, fmt.Sprintf(messageFmt, args...))
}

func (f *FakeRecorder) PastEventf(obj runtime.Object, timestamp metav1.Time, eventtype, reason, messageFmt string, args ...interface{}) {
}

func (f *FakeRecorder) generateEvent(obj runtime.Object, timestamp metav1.Time, eventtype, reason, message string) {
	ref, err := v1.GetReference(obj)
	if err != nil {
		return
	}
	event := f.makeEvent(ref, eventtype, reason, message)
	event.Source = f.source
	if f.events != nil {
		fmt.Println("write event")
		f.events = append(f.events, event)
	}
}

func (f *FakeRecorder) makeEvent(ref *v1.ObjectReference, eventtype, reason, message string) *v1.Event {
	fmt.Println("make event")
	t := metav1.Time{Time: f.clock.Now()}
	namespace := ref.Namespace
	if namespace == "" {
		namespace = v1.NamespaceDefault
	}
	return &v1.Event{
		ObjectMeta: v1.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: *ref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
		Type:           eventtype,
	}
}

func NewFakeRecorder() *FakeRecorder {
	return &FakeRecorder{
		source: v1.EventSource{Component: "nodeControllerTest"},
		events: []*v1.Event{},
		clock:  clock.NewFakeClock(time.Now()),
	}
}

func newNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: v1.ObjectMeta{Name: name},
		Spec: v1.NodeSpec{
			ExternalID: name,
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}
}

func newPod(name, host string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "default",
			Name:      name,
		},
		Spec: v1.PodSpec{
			NodeName: host,
		},
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}

	return pod
}

func contains(node *v1.Node, nodes []*v1.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}

// Returns list of zones for all Nodes stored in FakeNodeHandler
func getZones(nodeHandler *FakeNodeHandler) []string {
	nodes, _ := nodeHandler.List(v1.ListOptions{})
	zones := sets.NewString()
	for _, node := range nodes.Items {
		zones.Insert(utilnode.GetZoneKey(&node))
	}
	return zones.List()
}

func createZoneID(region, zone string) string {
	return region + ":\x00:" + zone
}
