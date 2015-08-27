/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package daemon

import (
	"fmt"
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	simpleDaemonLabel  = map[string]string{"name": "simple-daemon", "type": "production"}
	simpleDaemonLabel2 = map[string]string{"name": "simple-daemon", "type": "test"}
	simpleNodeLabel    = map[string]string{"color": "blue", "speed": "fast"}
	simpleNodeLabel2   = map[string]string{"color": "red", "speed": "fast"}
)

type FakePodControl struct {
	daemonSpec    []expapi.DaemonSet
	deletePodName []string
	lock          sync.Mutex
	err           error
}

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
}

func (f *FakePodControl) CreateReplica(namespace string, spec *api.ReplicationController) error {
	return nil
}

func (f *FakePodControl) CreateReplicaOnNode(namespace string, daemon *expapi.DaemonSet, nodeName string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.err != nil {
		return f.err
	}
	f.daemonSpec = append(f.daemonSpec, *daemon)
	return nil
}

func (f *FakePodControl) DeletePod(namespace string, podName string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.err != nil {
		return f.err
	}
	f.deletePodName = append(f.deletePodName, podName)
	return nil
}
func (f *FakePodControl) clear() {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.deletePodName = []string{}
	f.daemonSpec = []expapi.DaemonSet{}
}

func newDaemon(name string) *expapi.DaemonSet {
	return &expapi.DaemonSet{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: expapi.DaemonSetSpec{
			Selector: simpleDaemonLabel,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: simpleDaemonLabel,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: api.TerminationMessagePathDefault,
							ImagePullPolicy:        api.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					DNSPolicy: api.DNSDefault,
				},
			},
		},
	}
}

func newNode(name string, label map[string]string) *api.Node {
	return &api.Node{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: api.NamespaceDefault,
		},
	}
}

func addNodes(nodeStore cache.Store, startIndex, numNodes int, label map[string]string) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		nodeStore.Add(newNode(fmt.Sprintf("node-%d", i), label))
	}
}

func newPod(podName string, nodeName string, label map[string]string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		ObjectMeta: api.ObjectMeta{
			GenerateName: podName,
			Labels:       label,
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			NodeName: nodeName,
			Containers: []api.Container{
				{
					Image: "foo/bar",
					TerminationMessagePath: api.TerminationMessagePathDefault,
					ImagePullPolicy:        api.PullIfNotPresent,
					SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
			DNSPolicy: api.DNSDefault,
		},
	}
	api.GenerateName(api.SimpleNameGenerator, &pod.ObjectMeta)
	return pod
}

func addPods(podStore cache.Store, nodeName string, label map[string]string, number int) {
	for i := 0; i < number; i++ {
		podStore.Add(newPod(fmt.Sprintf("%s-", nodeName), nodeName, label))
	}
}

func makeTestManager() (*DaemonManager, *FakePodControl) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	manager := NewDaemonManager(client)
	podControl := &FakePodControl{}
	manager.podControl = podControl
	return manager, podControl
}

func validateSyncDaemons(t *testing.T, fakePodControl *FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.daemonSpec) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.daemonSpec))
	}
	if len(fakePodControl.deletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.deletePodName))
	}
}

func syncAndValidateDaemons(t *testing.T, manager *DaemonManager, daemon *expapi.DaemonSet, podControl *FakePodControl, expectedCreates, expectedDeletes int) {
	key, err := controller.KeyFunc(daemon)
	if err != nil {
		t.Errorf("Could not get key for daemon.")
	}
	manager.syncHandler(key)
	validateSyncDaemons(t, podControl, expectedCreates, expectedDeletes)
}

// Daemon without node selectors should launch pods on every node.
func TestSimpleDaemonLaunchesPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	daemon := newDaemon("foo")
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 5, 0)
}

// Daemon without node selectors should launch pods on every node.
func TestNoNodesDoesNothing(t *testing.T) {
	manager, podControl := makeTestManager()
	daemon := newDaemon("foo")
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 0, 0)
}

// Daemon without node selectors should launch pods on every node.
func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	manager, podControl := makeTestManager()
	manager.nodeStore.Add(newNode("only-node", nil))
	daemon := newDaemon("foo")
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 1, 0)
}

// Manager should not create pods on nodes which have daemon pods, and should remove excess pods from nodes that have extra pods.
func TestDealsWithExistingPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addPods(manager.podStore.Store, "node-1", simpleDaemonLabel, 1)
	addPods(manager.podStore.Store, "node-2", simpleDaemonLabel, 2)
	addPods(manager.podStore.Store, "node-3", simpleDaemonLabel, 5)
	addPods(manager.podStore.Store, "node-4", simpleDaemonLabel2, 2)
	daemon := newDaemon("foo")
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 2, 5)
}

// Daemon with node selector should launch pods on nodes matching selector.
func TestSelectorDaemonLaunchesPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 3, 0)
}

// Daemon with node selector should delete pods from nodes that do not satisfy selector.
func TestSelectorDaemonDeletesUnselectedPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Store, "node-0", simpleDaemonLabel2, 2)
	addPods(manager.podStore.Store, "node-1", simpleDaemonLabel, 3)
	addPods(manager.podStore.Store, "node-1", simpleDaemonLabel2, 1)
	addPods(manager.podStore.Store, "node-4", simpleDaemonLabel, 1)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 5, 4)
}

// Daemon with node selector should launch pods on nodes matching selector, but also deal with existing pods on nodes.
func TestSelectorDaemonDealsWithExistingPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Store, "node-0", simpleDaemonLabel, 1)
	addPods(manager.podStore.Store, "node-1", simpleDaemonLabel, 3)
	addPods(manager.podStore.Store, "node-1", simpleDaemonLabel2, 2)
	addPods(manager.podStore.Store, "node-2", simpleDaemonLabel, 4)
	addPods(manager.podStore.Store, "node-6", simpleDaemonLabel, 13)
	addPods(manager.podStore.Store, "node-7", simpleDaemonLabel2, 4)
	addPods(manager.podStore.Store, "node-9", simpleDaemonLabel, 1)
	addPods(manager.podStore.Store, "node-9", simpleDaemonLabel2, 1)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 3, 20)
}

// Daemon with node selector which does not match any node labels should not launch pods.
func TestBadSelectorDaemonDoesNothing(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel2
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 0, 0)
}

// Daemon with node name should launch pod on node with corresponding name.
func TestNameDaemonLaunchesPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeName = "node-0"
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 1, 0)
}

// Daemon with node name that does not exist should not launch pods.
func TestBadNameDaemonDoesNothing(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeName = "node-10"
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 0, 0)
}

// Daemon with node selector, and node name, matching a node, should launch a pod on the node.
func TestNameAndSelectorDaemonLaunchesPods(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	daemon.Spec.Template.Spec.NodeName = "node-6"
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 1, 0)
}

// Daemon with node selector that matches some nodes, and node name that matches a different node, should do nothing.
func TestInconsistentNameSelectorDaemonDoesNothing(t *testing.T) {
	manager, podControl := makeTestManager()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemon("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	daemon.Spec.Template.Spec.NodeName = "node-0"
	manager.dcStore.Add(daemon)
	syncAndValidateDaemons(t, manager, daemon, podControl, 0, 0)
}
