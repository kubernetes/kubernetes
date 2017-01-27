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
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errorUtils "k8s.io/apimachinery/pkg/util/errors"

	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
)

// StatefulPodControlInterface defines the interface that StatefulSetController uses to create, update, and delete Pods,
// and to update the Status of a StatefulSet. It follows the design paradigms used for PodControl, but its
// implementation provides for PVC creation, ordered Pod creation, ordered Pod termination, and Pod identity enforcement.
// Like controller.PodControlInterface, it is implemented as an interface to provide for testing fakes.
type StatefulPodControlInterface interface {
	// Creates a Pod in a StatefulSet. Any PVCs necessary for the Pod are created prior to creating the Pod. If
	// the returned error is nil the Pod and its PVCs have been created.
	CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// Updates a Pod in a StatefulSet. If the Pod already has the correct identity and stable storage this method
	// is a no-op. If the Pod must be mutated to conform to the Set, it is mutated and updated. pod is an in-out
	// parameter, and any updates made to the pod are reflected as mutations to this parameter. If the method is
	// successful the returned error is nil.
	UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// Deletes a Pod in a StatefulSet. The pods PVCs are not deleted. If the returned error is nil the method is
	// successful.
	DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// Updates the Status of a StatefulSet. set is an in-out parameter, and any updates made to the set are
	// made visible as mutations to the parameter. If the method is successful the returned error is nil.
	UpdateStatefulSetStatus(set *apps.StatefulSet) error
}

func NewRealStatefulPodControl(client clientset.Interface, recorder record.EventRecorder) StatefulPodControlInterface {
	return &realStatefulPodControl{client, recorder}
}

// realStatefulPodControl implements StatefulPodControlInterface using a clientset.Interface to communicate with the
// API server. The struct is package provide because we don't want to expose internal details importing packages.
type realStatefulPodControl struct {
	client   clientset.Interface
	recorder record.EventRecorder
}

func (spc *realStatefulPodControl) CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	if pod == nil || set == nil {
		return nilParameterError
	}
	// Create the Pod's PVCs prior to creating the Pod
	err := spc.createPersistentVolumeClaims(set, pod)
	// If we created the PVCs attempt to create the Pod
	if err == nil {
		// sink already exists errors
		_, err = spc.client.Core().Pods(set.Namespace).Create(pod)
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
	}
	spc.recordPodEvent("create", set, pod, err)
	return err
}

func (spc *realStatefulPodControl) UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	if pod == nil || set == nil {
		return nilParameterError
	}
	// we make a copy of the Pod on the stack and mutate the copy. Successful mutations are made visible my copying
	// the address of the copy to back to memory location indicated by pod
	clone := *pod
	for attempt := 0; attempt < maxUpdateRetries; attempt++ {
		// assume the Pod is consistent
		consistent := true
		// if the Pod does not conform to it's identity, update the identity and dirty the Pod
		if !identityMatches(set, &clone) {
			updateIdentity(set, &clone)
			consistent = false
		}
		// if the Pod does not conform to the StatefulSet's storage requirements, update the Pod's PVC's,
		// dirty the Pod, and create any missing PVCs
		if !storageMatches(set, &clone) {
			updateStorage(set, &clone)
			consistent = false
			if err := spc.createPersistentVolumeClaims(set, &clone); err != nil {
				spc.recordPodEvent("update", set, pod, err)
				return err
			}
		}
		// if the Pod is not dirty do nothing
		if consistent {
			*pod = clone
			return nil
		}
		// commit the update, retrying on conflicts
		// TODO do we need the extra GET request or can we just use the return parameter from the client on conflict?
		if _, err := spc.client.Core().Pods(set.Namespace).Update(&clone); apierrors.IsConflict(err) {
			if conflicting, err := spc.client.Core().Pods(set.Namespace).Get(clone.Name, metav1.GetOptions{}); err != nil {
				spc.recordPodEvent("update", set, &clone, err)
				return err
			} else {
				clone = *conflicting
			}
		} else {
			if err == nil {
				*pod = clone
			}
			spc.recordPodEvent("update", set, pod, err)
			return err
		}
	}
	spc.recordPodEvent("update", set, pod, updateConflictError)
	return updateConflictError
}

func (spc *realStatefulPodControl) DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	if pod == nil || set == nil {
		return nilParameterError
	}
	err := spc.client.Core().Pods(set.Namespace).Delete(pod.Name, nil)
	spc.recordPodEvent("delete", set, pod, err)
	return err
}

func (spc *realStatefulPodControl) UpdateStatefulSetStatus(set *apps.StatefulSet) error {
	if set == nil {
		return nilParameterError
	}
	setCopy := *set
	for attempt := 0; attempt < maxUpdateRetries; attempt++ {
		if _, err := spc.client.Apps().StatefulSets(setCopy.Namespace).UpdateStatus(&setCopy); apierrors.IsConflict(err) {
			if conflicting, err := spc.client.Apps().StatefulSets(setCopy.Namespace).Get(setCopy.Name, metav1.GetOptions{}); err != nil {
				return err
			} else {
				conflicting.Status = setCopy.Status
				setCopy = *conflicting
			}
		} else {
			return err
		}
	}
	*set = setCopy
	return updateConflictError
}

// Records an event for method applied to a Pod in a StatefulSet. If err is nil the generated event will have a reason
// of v1.EventTypeNormal. If err is not nil the generated event will have a reason of v1.EventTypeWarning.
func (spc *realStatefulPodControl) recordPodEvent(method string, set *apps.StatefulSet, pod *v1.Pod, err error) {
	if err == nil {
		reason := fmt.Sprintf("Successful%s", strings.Title(method))
		message := fmt.Sprintf("%s Pod %s in StatefulSet %s successful",
			strings.ToLower(method), pod.Name, set.Name)
		spc.recorder.Event(set, v1.EventTypeNormal, reason, message)
	} else {
		reason := fmt.Sprintf("Failed%s", strings.Title(method))
		message := fmt.Sprintf("%s Pod %s in StatefulSet %s failed error: %s",
			strings.ToLower(method), pod.Name, set.Name, err)
		spc.recorder.Event(set, v1.EventTypeWarning, reason, message)
	}
}

// Records an event for method applied to the PersistentVolumeClaim of a Pod in a StatefulSet. If err is nil the
// generated event will have a reason of v1.EventTypeNormal. If err is not nil the generated event will have a reason of
// v1.EventTypeWarning.
func (spc *realStatefulPodControl) recordClaimEvent(method string, set *apps.StatefulSet, pod *v1.Pod, claim *v1.PersistentVolumeClaim, err error) {
	if err == nil {
		reason := fmt.Sprintf("Successful%s", strings.Title(method))
		message := fmt.Sprintf("%s Claim %s Pod %s in StatefulSet %s success",
			strings.ToLower(method), claim.Name, pod.Name, set.Name)
		spc.recorder.Event(set, v1.EventTypeNormal, reason, message)
	} else {
		reason := fmt.Sprintf("Failed%s", strings.Title(method))
		message := fmt.Sprintf("%s Claim %s for Pod %s in StatefulSet %s failed error: %s",
			strings.ToLower(method), claim.Name, pod.Name, set.Name, err)
		spc.recorder.Event(set, v1.EventTypeWarning, reason, message)
	}
}

func (spc *realStatefulPodControl) createPersistentVolumeClaims(set *apps.StatefulSet, pod *v1.Pod) error {
	errs := make([]error, 0)
	for _, claim := range getPersistentVolumeClaims(set, pod) {
		_, err := spc.client.Core().PersistentVolumeClaims(set.Namespace).Get(claim.Namespace, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				_, err := spc.client.Core().PersistentVolumeClaims(set.Namespace).Create(&claim)
				if err != nil {
					errs = append(errs, fmt.Errorf("Failed to create PVC %s: %s", claim.Name, err))
				}
				spc.recordClaimEvent("create", set, pod, &claim, err)
			} else {
				errs = append(errs, fmt.Errorf("Failed to retrieve PVC %s: %s", claim.Name, err))
				spc.recordClaimEvent("create", set, pod, &claim, err)
			}
		}
		// TODO: Check resource requirements and accessmodes, update if necessary
	}
	if len(errs) > 0 {
		return errorUtils.NewAggregate(errs)
	}
	return nil
}

var _ StatefulPodControlInterface = &realStatefulPodControl{}
