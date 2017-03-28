/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	intstrutil "k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/controller/daemon/util"
)

// rollingUpdate deletes old daemon set pods making sure that no more than
// ds.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable pods are unavailable
func (dsc *DaemonSetsController) rollingUpdate(ds *extensions.DaemonSet) error {
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		return fmt.Errorf("couldn't get node to daemon pod mapping for daemon set %q: %v", ds.Name, err)
	}

	_, oldPods, err := dsc.getAllDaemonSetPods(ds, nodeToDaemonPods)
	maxUnavailable, numUnavailable, err := dsc.getUnavailableNumbers(ds, nodeToDaemonPods)
	if err != nil {
		return fmt.Errorf("Couldn't get unavailable numbers: %v", err)
	}
	oldAvailablePods, oldUnavailablePods := util.SplitByAvailablePods(ds.Spec.MinReadySeconds, oldPods)

	// for oldPods delete all not running pods
	var podsToDelete []string
	glog.V(4).Infof("Marking all unavailable old pods for deletion")
	for _, pod := range oldUnavailablePods {
		// Skip terminating pods. We won't delete them again
		if pod.DeletionTimestamp != nil {
			continue
		}
		glog.V(4).Infof("Marking pod %s/%s for deletion", ds.Name, pod.Name)
		podsToDelete = append(podsToDelete, pod.Name)
	}

	glog.V(4).Infof("Marking old pods for deletion")
	for _, pod := range oldAvailablePods {
		if numUnavailable >= maxUnavailable {
			glog.V(4).Infof("Number of unavailable DaemonSet pods: %d, is equal to or exceeds allowed maximum: %d", numUnavailable, maxUnavailable)
			break
		}
		glog.V(4).Infof("Marking pod %s/%s for deletion", ds.Name, pod.Name)
		podsToDelete = append(podsToDelete, pod.Name)
		numUnavailable++
	}
	errors := dsc.syncNodes(ds, podsToDelete, []string{})
	return utilerrors.NewAggregate(errors)
}

func (dsc *DaemonSetsController) getAllDaemonSetPods(ds *extensions.DaemonSet, nodeToDaemonPods map[string][]*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	var newPods []*v1.Pod
	var oldPods []*v1.Pod

	for _, pods := range nodeToDaemonPods {
		for _, pod := range pods {
			if util.IsPodUpdated(ds.Spec.TemplateGeneration, pod) {
				newPods = append(newPods, pod)
			} else {
				oldPods = append(oldPods, pod)
			}
		}
	}
	return newPods, oldPods, nil
}

func (dsc *DaemonSetsController) getUnavailableNumbers(ds *extensions.DaemonSet, nodeToDaemonPods map[string][]*v1.Pod) (int, int, error) {
	glog.V(4).Infof("Getting unavailable numbers")
	// TODO: get nodeList once in syncDaemonSet and pass it to other functions
	nodeList, err := dsc.nodeLister.List(labels.Everything())
	if err != nil {
		return -1, -1, fmt.Errorf("couldn't get list of nodes during rolling update of daemon set %#v: %v", ds, err)
	}

	var numUnavailable, desiredNumberScheduled int
	for i := range nodeList {
		node := nodeList[i]
		wantToRun, _, _, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			return -1, -1, err
		}
		if !wantToRun {
			continue
		}
		desiredNumberScheduled++
		daemonPods, exists := nodeToDaemonPods[node.Name]
		if !exists {
			numUnavailable++
			continue
		}
		available := false
		for _, pod := range daemonPods {
			if v1.IsPodAvailable(pod, ds.Spec.MinReadySeconds, metav1.Now()) {
				available = true
				break
			}
		}
		if !available {
			numUnavailable++
		}
	}
	maxUnavailable, err := intstrutil.GetValueFromIntOrPercent(ds.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable, desiredNumberScheduled, true)
	if err != nil {
		return -1, -1, fmt.Errorf("Invalid value for MaxUnavailable: %v", err)
	}
	return maxUnavailable, numUnavailable, nil
}
