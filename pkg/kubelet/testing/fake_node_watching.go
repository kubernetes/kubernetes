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

package testing

import (
	"time"

	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

type FakeNodeWatchReactor struct {
	sync.Mutex
	node     v1.Node
	nodeChan chan v1.Node
	isClosed bool
}

func AddFakeNodeWatchReactor(
	kubeClient *fake.Clientset,
	node *v1.Node) *FakeNodeWatchReactor {

	fnw := &FakeNodeWatchReactor{
		node:     *node,
		nodeChan: make(chan v1.Node, 1),
	}

	kubeClient.AddWatchReactor("nodes",
		func(action core.Action) (bool, watch.Interface, error) {
			fakeWatch := watch.NewRaceFreeFake()
			go func() {
				for {
					node, more := <-fnw.nodeChan
					if !more {
						return
					}
					fakeWatch.Add(&node)
				}
			}()
			return true, fakeWatch, nil
		})

	return fnw
}

func (fnw *FakeNodeWatchReactor) GetNode() *v1.Node {
	fnw.Lock()
	defer fnw.Unlock()
	return &fnw.node
}

func (fnw *FakeNodeWatchReactor) send() {
	if fnw.isClosed {
		return
	}
	fnw.nodeChan <- fnw.node
}

func (fnw *FakeNodeWatchReactor) UpdateNode(node *v1.Node) {
	fnw.Lock()
	defer fnw.Unlock()
	fnw.node = *node
	fnw.send()
}

func (fnw *FakeNodeWatchReactor) UpdateNodeStatus(status *v1.NodeStatus) {
	fnw.Lock()
	defer fnw.Unlock()
	fnw.node.Status = *status
	fnw.send()
}

func (fnw *FakeNodeWatchReactor) Close() {
	fnw.Lock()
	defer fnw.Unlock()
	if !fnw.isClosed {
		close(fnw.nodeChan)
	}
	fnw.isClosed = true
}

func SimulateVolumeInUseUpdate(
	volumeName v1.UniqueVolumeName,
	stopCh <-chan struct{},
	fnw *FakeNodeWatchReactor) {

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			status := fnw.GetNode().Status
			status.VolumesInUse = []v1.UniqueVolumeName{volumeName}
			fnw.UpdateNodeStatus(&status)
		case <-stopCh:
			return
		}
	}
}
