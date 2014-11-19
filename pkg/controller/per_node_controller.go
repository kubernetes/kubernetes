/*
Copyright 2014 Google Inc. All rights reserved.

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

package controller

import (
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// PerNodeManager is responsible for synchronizing PerNodeController objects stored
// in the system with actual running pods.
type PerNodeManager struct {
	kubeClient client.Interface
	podControl PodController
	syncTime   <-chan time.Time

	// To allow injection of syncPerNodeController for testing.
	syncHandler func(controller api.PerNodeController) error
}

// NewPerNodeManager creates a new PerNodeManager.
func NewPerNodeManager(kubeClient client.Interface) *PerNodeManager {
	pnm := &PerNodeManager{
		kubeClient: kubeClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
		},
	}
	pnm.syncHandler = pnm.syncPerNodeController
	return pnm
}

// Run begins watching and syncing.
func (pnm *PerNodeManager) Run(period time.Duration) {
	pnm.syncTime = time.Tick(period)
	resourceVersion := ""
	go util.Forever(func() { pnm.watchControllers(&resourceVersion) }, period)
}

// resourceVersion is a pointer to the resource version to use/update.
func (pnm *PerNodeManager) watchControllers(resourceVersion *string) {
	watching, err := pnm.kubeClient.PerNodeControllers(api.NamespaceAll).Watch(
		labels.Everything(),
		labels.Everything(),
		*resourceVersion,
	)
	if err != nil {
		glog.Errorf("Unexpected failure to watch: %v", err)
		time.Sleep(5 * time.Second)
		return
	}
	// TODO: We could watch minions too, and react when a new one joins.

	for {
		select {
		case <-pnm.syncTime:
			pnm.synchronize()
		case event, open := <-watching.ResultChan():
			if !open {
				// watchChannel has been closed, or something else went
				// wrong with our etcd watch call. Let the util.Forever()
				// that called us call us again.
				return
			}
			glog.V(4).Infof("Got watch: %#v", event)
			rc, ok := event.Object.(*api.PerNodeController)
			if !ok {
				glog.Errorf("unexpected object: %#v", event.Object)
				continue
			}
			// If we get disconnected, start where we left off.
			*resourceVersion = rc.ResourceVersion
			// Sync even if this is a deletion event, to ensure that we leave
			// it in the desired state.
			glog.V(4).Infof("About to sync from watch: %v", rc.Name)
			if err := pnm.syncHandler(*rc); err != nil {
				glog.Errorf("unexpected sync. error: %v", err)
			}
		}
	}
}

func (pnm *PerNodeManager) syncPerNodeController(controller api.PerNodeController) error {
	// Get the list of matching pods.
	s := labels.Set(controller.Spec.Selector).AsSelector()
	allPods, err := pnm.kubeClient.Pods(controller.Namespace).List(s)
	if err != nil {
		return err
	}
	pods := pnm.filterActivePods(allPods.Items)

	// Get the list of minions.
	nodes, err := pnm.kubeClient.Minions().List()
	if err != nil {
		return err
	}

	nodeMap := map[string][]*api.Pod{}
	for i := range pods {
		pod := &pods[i]
		if nodeMap[pod.CurrentState.Host] == nil {
			nodeMap[pod.CurrentState.Host] = []*api.Pod{}
		}
		nodeMap[pod.CurrentState.Host] = append(nodeMap[pod.CurrentState.Host], pod)
	}
	if len(nodes.Items) == len(pods) && len(nodeMap) == len(nodes.Items) {
		// We have one pod per node - return early.
		return nil
	}

	wait := sync.WaitGroup{}
	for i := range nodes.Items {
		node := &nodes.Items[i]
		if nodeMap[node.Name] == nil {
			// No pod on this node - start it.
			glog.V(2).Infof("Node %s not running per-node pod %s.%s, creating it\n", node.Name, controller.Namespace, controller.Name)
			wait.Add(1)
			go func() {
				defer wait.Done()
				pnm.podControl.createPod(controller.Namespace, controller.Spec.Template)
			}()
		} else if len(nodeMap[node.Name]) != 1 {
			// Too many pods on this node - kill some.
			n := len(nodeMap[node.Name])
			glog.V(2).Infof("Node %s has %d instances of per-node pod %s.%s, deleting %d\n", node.Name, n, controller.Namespace, controller.Name, n-1)
			for i := range nodeMap[node.Name][1:] {
				go func() {
					defer wait.Done()
					pnm.podControl.deletePod(controller.Namespace, nodeMap[node.Name][i].Name)
				}()
			}
		}
	}
	wait.Wait()
	return nil
}

func (pnm *PerNodeManager) filterActivePods(pods []api.Pod) []api.Pod {
	var result []api.Pod
	for _, value := range pods {
		status := value.CurrentState.Status
		if status != api.PodSucceeded && status != api.PodFailed {
			result = append(result, value)
		}
	}
	return result
}

func (pnm *PerNodeManager) synchronize() {
	// TODO: remove this method completely and rely on the watch.
	// Add resource version tracking to watch to make this work.
	var controllers []api.PerNodeController
	list, err := pnm.kubeClient.PerNodeControllers(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	controllers = list.Items
	wg := sync.WaitGroup{}
	wg.Add(len(controllers))
	for ix := range controllers {
		go func(ix int) {
			defer wg.Done()
			glog.V(4).Infof("periodic sync of %v", controllers[ix].Name)
			err := pnm.syncHandler(controllers[ix])
			if err != nil {
				glog.Errorf("Error synchronizing: %#v", err)
			}
		}(ix)
	}
	wg.Wait()
}
