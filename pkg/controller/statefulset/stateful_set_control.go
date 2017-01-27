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
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
)

// StatefulSetControl implements the control logic for updating StatefulSets and their children Pods. It is implemented
// as an interface to allow for extensions that provide different semantics. Currently, there is only one implementation.
// If you are looking to fork the StatefulSet codebase in order to modify its semantics, this is the code you are
// looking for.
type StatefulSetControlInterface interface {
	// UpdateStatefulSet implements the control logic for Pod creation, update, and deletion, and
	// persistent volume creation, update, and deletion.
	// If an implementation returns a non-nil error, the invocation will be retried using a rate-limited strategy.
	// Implementors should sink any errors that they do not wish to trigger a retry, and they may feel free to
	// exit exceptionally at any point provided they wish the update to be re-run at a later point in time.
	UpdateStatefulSet(set *apps.StatefulSet, pods []*v1.Pod) error
}

// Returns a new instance of the default implementation StatefulSetControlInterface that implements the documented
// semantics for StatefulSets. podControl is the PodControlInterface used to create, update, and delete Pods and to
// create PersistentVolumeClaims. You should use an instance returned from NewRealStatefulPodControl() for any scenario
// other than testing.
func NewDefaultStatefulSetControl(podControl StatefulPodControlInterface) StatefulSetControlInterface {
	return &defaultStatefulSetControl{podControl}
}

type defaultStatefulSetControl struct {
	podControl StatefulPodControlInterface
}

func (ssc *defaultStatefulSetControl) UpdateStatefulSet(set *apps.StatefulSet, pods []*v1.Pod) error {
	if set == nil {
		return nilParameterError
	}
	replicaCount := int(*set.Spec.Replicas)
	replicas := make([]*v1.Pod, replicaCount)
	condemned := make([]*v1.Pod, 0, len(pods))
	readyReplicas := 0
	terminatingCondemned := 0

	// First we partition pods into two lists, replicas will contain all Pods such that
	// 0 <= getOrdinal(pod) < set.Spec.Replicas and condemned will contain all Pods such that
	// set.Spec.Replicas <= getOrdinal(pod)
	for i := range pods {
		//count the number of running and ready replicas
		if isRunningAndReady(pods[i]) {
			readyReplicas++
		}
		if ord := getOrdinal(pods[i]); 0 <= ord && ord < replicaCount {
			// if the ordinal of the pod is withing the range of the current number of replicas,
			// insert it at the indirection of its ordinal
			replicas[ord] = pods[i]

		} else if ord >= replicaCount {
			// if the ordinal is greater than the number of replicas add it to the condemned list
			condemned = append(condemned, pods[i])
			// count the number of terminating condemned
			if isTerminated(pods[i]) {
				terminatingCondemned++
			}
		}
	}

	// for any empty indices in the sequence [0,set.Spec.Replicas) create a new Pod
	for ord := 0; ord < replicaCount; ord++ {
		if replicas[ord] == nil {
			replicas[ord] = newStatefulSetPod(set, ord)
		}
	}
	// sort the condemned Pods by their ordinals
	sort.Sort(ascendingOrdinal(condemned))

	// attempt to update set with the current number of replicas
	set.Status = apps.StatefulSetStatus{Replicas: int32(readyReplicas)}
	if err := ssc.podControl.UpdateStatefulSetStatus(set); err != nil {
		return err
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
		if isCreated(replicas[i]) {
			// if we have a Pod that has been created but is not running and ready we can not make progress
			if !isRunningAndReady(replicas[i]) {
				glog.V(2).Infof("StatefulSet %s is waiting for Pod %s to be Running and Ready",
					set.Name, replicas[i].Name)
				return nil
			} else {
				// enforce the StatefulSet invariants
				if err := ssc.podControl.UpdateStatefulPod(set, replicas[i]); err != nil {
					return err
				}
			}
		} else {
			// If we find a Pod that has not been created we create the Pod immediately and return
			return ssc.podControl.CreateStatefulPod(set, replicas[i])
		}
	}

	// At this point, all of the current Replicas are Running and Ready, if no condemned Pod is terminating,
	// terminate the the Pod with the highest ordinal.
	if terminatingCondemned > 0 {
		glog.V(2).Infof("StatefulSet %s is waiting for %d Pods to terminate", set.Name, terminatingCondemned)
	} else if target := len(condemned) - 1; target >= 0 {
		glog.V(2).Infof("StatefulSet %s terminating Pod %s", set.Name, condemned[target])
		return ssc.podControl.DeleteStatefulPod(set, condemned[target])
	}
	return nil
}

var _ StatefulSetControlInterface = &defaultStatefulSetControl{}
