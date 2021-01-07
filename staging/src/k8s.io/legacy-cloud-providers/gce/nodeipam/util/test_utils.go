/*
Copyright 2021 The Kubernetes Authors.

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

package util

import (
	"context"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	ref "k8s.io/client-go/tools/reference"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
)

// FakeNodeHandler is a fake implementation of NodesInterface and NodeInterface. It
// allows test cases to have fine-grained control over mock behaviors. We also need
// PodsInterface and PodInterface to test list & delete pods, which is implemented in
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
	DeleteWaitChan chan struct{}
	PatchWaitChan  chan struct{}
}

// GetUpdatedNodesCopy returns a slice of Nodes with updates applied.
func (m *FakeNodeHandler) GetUpdatedNodesCopy() []*v1.Node {
	m.lock.Lock()
	defer m.lock.Unlock()
	updatedNodesCopy := make([]*v1.Node, len(m.UpdatedNodes), len(m.UpdatedNodes))
	for i, ptr := range m.UpdatedNodes {
		updatedNodesCopy[i] = ptr
	}
	return updatedNodesCopy
}

// List returns a list of Nodes from the fake store.
func (m *FakeNodeHandler) List(_ context.Context, opts metav1.ListOptions) (*v1.NodeList, error) {
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

func contains(node *v1.Node, nodes []*v1.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}

// FakeRecorder is used as a fake during testing.
type FakeRecorder struct {
	sync.Mutex
	source v1.EventSource
	Events []*v1.Event
	clock  clock.Clock
}

// Event emits a fake event to the fake recorder
func (f *FakeRecorder) Event(obj runtime.Object, eventtype, reason, message string) {
	f.generateEvent(obj, metav1.Now(), eventtype, reason, message)
}

// Eventf emits a fake formatted event to the fake recorder
func (f *FakeRecorder) Eventf(obj runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	f.Event(obj, eventtype, reason, fmt.Sprintf(messageFmt, args...))
}

// AnnotatedEventf emits a fake formatted event to the fake recorder
func (f *FakeRecorder) AnnotatedEventf(obj runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
	f.Eventf(obj, eventtype, reason, messageFmt, args...)
}

func (f *FakeRecorder) generateEvent(obj runtime.Object, timestamp metav1.Time, eventtype, reason, message string) {
	f.Lock()
	defer f.Unlock()
	ref, err := ref.GetReference(runtime.NewScheme(), obj)
	if err != nil {
		klog.Errorf("Encountered error while getting reference: %v", err)
		return
	}
	event := f.makeEvent(ref, eventtype, reason, message)
	event.Source = f.source
	if f.Events != nil {
		f.Events = append(f.Events, event)
	}
}

func (f *FakeRecorder) makeEvent(ref *v1.ObjectReference, eventtype, reason, message string) *v1.Event {
	t := metav1.Time{Time: f.clock.Now()}
	namespace := ref.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}

	clientref := v1.ObjectReference{
		Kind:            ref.Kind,
		Namespace:       ref.Namespace,
		Name:            ref.Name,
		UID:             ref.UID,
		APIVersion:      ref.APIVersion,
		ResourceVersion: ref.ResourceVersion,
		FieldPath:       ref.FieldPath,
	}

	return &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: clientref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
		Type:           eventtype,
	}
}

// NewFakeRecorder returns a pointer to a newly constructed FakeRecorder.
func NewFakeRecorder() *FakeRecorder {
	return &FakeRecorder{
		source: v1.EventSource{Component: "nodeControllerTest"},
		Events: []*v1.Event{},
		clock:  clock.NewFakeClock(time.Now()),
	}
}
