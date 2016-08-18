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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
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
	CreateHook func(*FakeNodeHandler, *api.Node) bool
	Existing   []*api.Node

	// Output
	CreatedNodes        []*api.Node
	DeletedNodes        []*api.Node
	UpdatedNodes        []*api.Node
	UpdatedNodeStatuses []*api.Node
	RequestCount        int

	// Synchronization
	lock           sync.Mutex
	deleteWaitChan chan struct{}
}

type FakeLegacyHandler struct {
	unversionedcore.CoreInterface
	n *FakeNodeHandler
}

func (c *FakeNodeHandler) getUpdatedNodesCopy() []*api.Node {
	c.lock.Lock()
	defer c.lock.Unlock()
	updatedNodesCopy := make([]*api.Node, len(c.UpdatedNodes), len(c.UpdatedNodes))
	for i, ptr := range c.UpdatedNodes {
		updatedNodesCopy[i] = ptr
	}
	return updatedNodesCopy
}

func (c *FakeNodeHandler) Core() unversionedcore.CoreInterface {
	return &FakeLegacyHandler{c.Clientset.Core(), c}
}

func (m *FakeLegacyHandler) Nodes() unversionedcore.NodeInterface {
	return m.n
}

func (m *FakeNodeHandler) Create(node *api.Node) (*api.Node, error) {
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

func (m *FakeNodeHandler) Get(name string) (*api.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	for i := range m.Existing {
		if m.Existing[i].Name == name {
			nodeCopy := *m.Existing[i]
			return &nodeCopy, nil
		}
	}
	return nil, nil
}

func (m *FakeNodeHandler) List(opts api.ListOptions) (*api.NodeList, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	var nodes []*api.Node
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
	nodeList := &api.NodeList{}
	for _, node := range nodes {
		nodeList.Items = append(nodeList.Items, *node)
	}
	return nodeList, nil
}

func (m *FakeNodeHandler) Delete(id string, opt *api.DeleteOptions) error {
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

func (m *FakeNodeHandler) DeleteCollection(opt *api.DeleteOptions, listOpts api.ListOptions) error {
	return nil
}

func (m *FakeNodeHandler) Update(node *api.Node) (*api.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	nodeCopy := *node
	m.UpdatedNodes = append(m.UpdatedNodes, &nodeCopy)
	return node, nil
}

func (m *FakeNodeHandler) UpdateStatus(node *api.Node) (*api.Node, error) {
	m.lock.Lock()
	defer func() {
		m.RequestCount++
		m.lock.Unlock()
	}()
	nodeCopy := *node
	m.UpdatedNodeStatuses = append(m.UpdatedNodeStatuses, &nodeCopy)
	return node, nil
}

func (m *FakeNodeHandler) PatchStatus(nodeName string, data []byte) (*api.Node, error) {
	m.RequestCount++
	return &api.Node{}, nil
}

func (m *FakeNodeHandler) Watch(opts api.ListOptions) (watch.Interface, error) {
	return nil, nil
}

func (m *FakeNodeHandler) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (*api.Node, error) {
	return nil, nil
}

// FakeRecorder is used as a fake during testing.
type FakeRecorder struct {
	source api.EventSource
	events []*api.Event
	clock  clock.Clock
}

func (f *FakeRecorder) Event(obj runtime.Object, eventtype, reason, message string) {
	f.generateEvent(obj, unversioned.Now(), eventtype, reason, message)
}

func (f *FakeRecorder) Eventf(obj runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	f.Event(obj, eventtype, reason, fmt.Sprintf(messageFmt, args...))
}

func (f *FakeRecorder) PastEventf(obj runtime.Object, timestamp unversioned.Time, eventtype, reason, messageFmt string, args ...interface{}) {
}

func (f *FakeRecorder) generateEvent(obj runtime.Object, timestamp unversioned.Time, eventtype, reason, message string) {
	ref, err := api.GetReference(obj)
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

func (f *FakeRecorder) makeEvent(ref *api.ObjectReference, eventtype, reason, message string) *api.Event {
	fmt.Println("make event")
	t := unversioned.Time{Time: f.clock.Now()}
	namespace := ref.Namespace
	if namespace == "" {
		namespace = api.NamespaceDefault
	}
	return &api.Event{
		ObjectMeta: api.ObjectMeta{
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
		source: api.EventSource{Component: "nodeControllerTest"},
		events: []*api.Event{},
		clock:  clock.NewFakeClock(time.Now()),
	}
}

func newNode(name string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec: api.NodeSpec{
			ExternalID: name,
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}
}

func newPod(name, host string) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      name,
		},
		Spec: api.PodSpec{
			NodeName: host,
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

	return pod
}

func contains(node *api.Node, nodes []*api.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}

// Returns list of zones for all Nodes stored in FakeNodeHandler
func getZones(nodeHandler *FakeNodeHandler) []string {
	nodes, _ := nodeHandler.List(api.ListOptions{})
	zones := sets.NewString()
	for _, node := range nodes.Items {
		zones.Insert(utilnode.GetZoneKey(&node))
	}
	return zones.List()
}

func createZoneID(region, zone string) string {
	return region + ":\x00:" + zone
}
