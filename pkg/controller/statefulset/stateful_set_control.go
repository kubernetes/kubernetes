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

package statefulset

import (
	"fmt"
	"sort"

	"k8s.io/client-go/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"

	"github.com/golang/glog"
)

// StatefulSetControl implements the control logic for updating StatefulSets and their children Pods. It is implemented
// as an interface to allow for extensions that provide different semantics. Currently, there is only one implementation.
type StatefulSetControlInterface interface {
	// UpdateStatefulSet implements the control logic for Pod creation, update, and deletion, and
	// persistent volume creation, update, and deletion.
	// If an implementation returns a non-nil error, the invocation will be retried using a rate-limited strategy.
	// Implementors should sink any errors that they do not wish to trigger a retry, and they may feel free to
	// exit exceptionally at any point provided they wish the update to be re-run at a later point in time.
	UpdateStatefulSet(set *apps.StatefulSet, pods []*v1.Pod) error
}

// NewDefaultStatefulSetControl returns a new instance of the default implementation StatefulSetControlInterface that
// implements the documented semantics for StatefulSets. podControl is the PodControlInterface used to create, update,
// and delete Pods and to create PersistentVolumeClaims. You should use an instance returned from
// NewRealStatefulPodControl() for any scenario other than testing.
func NewDefaultStatefulSetControl(podControl StatefulPodControlInterface) StatefulSetControlInterface {
	return &defaultStatefulSetControl{podControl}
}

type defaultStatefulSetControl struct {
	podControl StatefulPodControlInterface
}

func (ssc *defaultStatefulSetControl) UpdateStatefulSet(set *apps.StatefulSet, pods []*v1.Pod) error {
	replicaCount := int(*set.Spec.Replicas)
	// slice that will contain all Pods such that 0 <= getOrdinal(pod) < set.Spec.Replicas
	replicas := make([]*v1.Pod, replicaCount)
	// slice that will contain all Pods such that set.Spec.Replicas <= getOrdinal(pod)
	condemned := make([]*v1.Pod, 0, len(pods))
	ready := 0
	unhealthy := 0

	// First we partition pods into two lists valid replicas and condemned Pods
	for i := range pods {
		//count the number of running and ready replicas
		if isRunningAndReady(pods[i]) {
			ready++
		}
		if ord := getOrdinal(pods[i]); 0 <= ord && ord < replicaCount {
			// if the ordinal of the pod is within the range of the current number of replicas,
			// insert it at the indirection of its ordinal
			replicas[ord] = pods[i]

		} else if ord >= replicaCount {
			// if the ordinal is greater than the number of replicas add it to the condemned list
			condemned = append(condemned, pods[i])
		}
		// If the ordinal could not be parsed (ord < 0), ignore the Pod.
	}

	// for any empty indices in the sequence [0,set.Spec.Replicas) create a new Pod
	for ord := 0; ord < replicaCount; ord++ {
		if replicas[ord] == nil {
			replicas[ord] = newStatefulSetPod(set, ord)
		}
	}

	// count the number of unhealthy pods
	for i := range replicas {
		if !isHealthy(replicas[i]) {
			unhealthy++
		}
	}
	for i := range condemned {
		if !isHealthy(condemned[i]) {
			unhealthy++
		}
	}

	// sort the condemned Pods by their ordinals
	sort.Sort(ascendingOrdinal(condemned))

	// if the current number of replicas has changed update the statefulSets replicas
	if set.Status.Replicas != int32(ready) || set.Status.ObservedGeneration == nil || set.Generation > *set.Status.ObservedGeneration {
		obj, err := api.Scheme.Copy(set)
		if err != nil {
			return fmt.Errorf("unable to copy set: %v", err)
		}
		set = obj.(*apps.StatefulSet)

		if err := ssc.podControl.UpdateStatefulSetStatus(set, int32(ready), set.Generation); err != nil {
			return err
		}
	}

	// If the StatefulSet is being deleted, don't do anything other than updating
	// status.
	if set.DeletionTimestamp != nil {
		return nil
	}

	// Examine each replica with respect to its ordinal
	for i := range replicas {
		// delete and recreate failed pods
		if isFailed(replicas[i]) {
			glog.V(2).Infof("StatefulSet %s is recreating failed Pod %s", set.Name, replicas[i].Name)
			if err := ssc.podControl.DeleteStatefulPod(set, replicas[i]); err != nil {
				return err
			}
			replicas[i] = newStatefulSetPod(set, i)
		}
		// If we find a Pod that has not been created we create the Pod immediately and return
		if !isCreated(replicas[i]) {
			return ssc.podControl.CreateStatefulPod(set, replicas[i])
		}
		// If we have a Pod that has been created but is not running and ready we can not make progress.
		// We must ensure that all for each Pod, when we create it, all of its predecessors, with respect to its
		// ordinal, are Running and Ready.
		if !isRunningAndReady(replicas[i]) {
			glog.V(2).Infof("StatefulSet %s is waiting for Pod %s to be Running and Ready",
				set.Name, replicas[i].Name)
			return nil
		}
		// Enforce the StatefulSet invariants
		if identityMatches(set, replicas[i]) && storageMatches(set, replicas[i]) {
			continue
		}
		// Make a deep copy so we don't mutate the shared cache
		copy, err := api.Scheme.DeepCopy(replicas[i])
		if err != nil {
			return err
		}
		replica := copy.(*v1.Pod)
		if err := ssc.podControl.UpdateStatefulPod(set, replica); err != nil {
			return err
		}
	}

	// At this point, all of the current Replicas are Running and Ready, we can consider termination.
	// We will wait for all predecessors to be Running and Ready prior to attempting a deletion.
	// We will terminate Pods in a monotonically decreasing order over [len(pods),set.Spec.Replicas).
	// Note that we do not resurrect Pods in this interval.
	if unhealthy > 0 {
		glog.V(2).Infof("StatefulSet %s is waiting on %d Pods", set.Name, unhealthy)
		return nil
	}
	if target := len(condemned) - 1; target >= 0 {
		glog.V(2).Infof("StatefulSet %s terminating Pod %s", set.Name, condemned[target])
		return ssc.podControl.DeleteStatefulPod(set, condemned[target])
	}
	return nil
}

var _ StatefulSetControlInterface = &defaultStatefulSetControl{}
