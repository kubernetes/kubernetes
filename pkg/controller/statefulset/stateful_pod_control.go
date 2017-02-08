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
	errorutils "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// StatefulPodControlInterface defines the interface that StatefulSetController uses to create, update, and delete Pods,
// and to update the Status of a StatefulSet. It follows the design paradigms used for PodControl, but its
// implementation provides for PVC creation, ordered Pod creation, ordered Pod termination, and Pod identity enforcement.
// Like controller.PodControlInterface, it is implemented as an interface to provide for testing fakes.
type StatefulPodControlInterface interface {
	// CreateStatefulPod create a Pod in a StatefulSet. Any PVCs necessary for the Pod are created prior to creating
	// the Pod. If the returned error is nil the Pod and its PVCs have been created.
	CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// UpdateStatefulPod Updates a Pod in a StatefulSet. If the Pod already has the correct identity and stable
	// storage this method is a no-op. If the Pod must be mutated to conform to the Set, it is mutated and updated.
	// pod is an in-out parameter, and any updates made to the pod are reflected as mutations to this parameter. If
	// the create is successful, the returned error is nil.
	UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// DeleteStatefulPod deletes a Pod in a StatefulSet. The pods PVCs are not deleted. If the delete is successful,
	// the returned error is nil.
	DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error
	// UpdateStatefulSetStatus Updates the Status.Replicas of a StatefulSet. set is an in-out parameter, and any
	// updates made to the set are made visible as mutations to the parameter. If the method is successful, the
	// returned error is nil, and set has its Status.Replicas field set to replicas.
	UpdateStatefulSetReplicas(set *apps.StatefulSet, replicas int32) error
}

func NewRealStatefulPodControl(client clientset.Interface, recorder record.EventRecorder) StatefulPodControlInterface {
	return &realStatefulPodControl{client, recorder}
}

// realStatefulPodControl implements StatefulPodControlInterface using a clientset.Interface to communicate with the
// API server. The struct is package private as the internal details are irrelevant to importing packages.
type realStatefulPodControl struct {
	client   clientset.Interface
	recorder record.EventRecorder
}

func (spc *realStatefulPodControl) CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	// Create the Pod's PVCs prior to creating the Pod
	if err := spc.createPersistentVolumeClaims(set, pod); err != nil {
		spc.recordPodEvent("create", set, pod, err)
		return err
	}
	// If we created the PVCs attempt to create the Pod
	_, err := spc.client.Core().Pods(set.Namespace).Create(pod)
	// sink already exists errors
	if apierrors.IsAlreadyExists(err) {
		return err
	}
	spc.recordPodEvent("create", set, pod, err)
	return err
}

func (spc *realStatefulPodControl) UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	// we make a copy of the Pod on the stack and mutate the copy
	// we copy back to pod to notify the caller of successful mutation
	obj, err := api.Scheme.Copy(pod)
	if err != nil {
		return fmt.Errorf("unable to copy pod: %v", err)
	}
	podCopy := obj.(*v1.Pod)
	for attempt := 0; attempt < maxUpdateRetries; attempt++ {
		// assume the Pod is consistent
		consistent := true
		// if the Pod does not conform to it's identity, update the identity and dirty the Pod
		if !identityMatches(set, podCopy) {
			updateIdentity(set, podCopy)
			consistent = false
		}
		// if the Pod does not conform to the StatefulSet's storage requirements, update the Pod's PVC's,
		// dirty the Pod, and create any missing PVCs
		if !storageMatches(set, podCopy) {
			updateStorage(set, podCopy)
			consistent = false
			if err := spc.createPersistentVolumeClaims(set, podCopy); err != nil {
				spc.recordPodEvent("update", set, pod, err)
				return err
			}
		}
		// if the Pod is not dirty do nothing
		if consistent {
			*pod = *podCopy
			return nil
		}
		// commit the update, retrying on conflicts
		_, err = spc.client.Core().Pods(set.Namespace).Update(podCopy)
		if !apierrors.IsConflict(err) {
			if err == nil {
				*pod = *podCopy
			}
			spc.recordPodEvent("update", set, pod, err)
			return err
		}
		conflicting, err := spc.client.Core().Pods(set.Namespace).Get(podCopy.Name, metav1.GetOptions{})
		if err != nil {
			spc.recordPodEvent("update", set, podCopy, err)
			return err
		}
		*podCopy = *conflicting
	}
	spc.recordPodEvent("update", set, pod, updateConflictError)
	return updateConflictError
}

func (spc *realStatefulPodControl) DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	err := spc.client.Core().Pods(set.Namespace).Delete(pod.Name, nil)
	spc.recordPodEvent("delete", set, pod, err)
	return err
}

func (spc *realStatefulPodControl) UpdateStatefulSetReplicas(set *apps.StatefulSet, replicas int32) error {
	if set.Status.Replicas == replicas {
		return nil
	}
	obj, err := api.Scheme.Copy(set)
	if err != nil {
		return fmt.Errorf("unable to copy set: %v", err)
	}
	setCopy := obj.(*apps.StatefulSet)
	setCopy.Status.Replicas = replicas
	for attempt := 0; attempt < maxUpdateRetries; attempt++ {
		_, err := spc.client.Apps().StatefulSets(setCopy.Namespace).UpdateStatus(setCopy)
		if !apierrors.IsConflict(err) {
			if err == nil {
				*set = *setCopy
			}
			return err
		}
		conflicting, err := spc.client.Apps().StatefulSets(setCopy.Namespace).Get(setCopy.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		conflicting.Status.Replicas = setCopy.Status.Replicas
		*setCopy = *conflicting
	}
	return updateConflictError
}

// recordPodEvent records an event for verb applied to a Pod in a StatefulSet. If err is nil the generated event will
// have a reason of v1.EventTypeNormal. If err is not nil the generated event will have a reason of v1.EventTypeWarning.
func (spc *realStatefulPodControl) recordPodEvent(verb string, set *apps.StatefulSet, pod *v1.Pod, err error) {
	if err == nil {
		reason := fmt.Sprintf("Successful%s", strings.Title(verb))
		message := fmt.Sprintf("%s Pod %s in StatefulSet %s successful",
			strings.ToLower(verb), pod.Name, set.Name)
		spc.recorder.Event(set, v1.EventTypeNormal, reason, message)
	} else {
		reason := fmt.Sprintf("Failed%s", strings.Title(verb))
		message := fmt.Sprintf("%s Pod %s in StatefulSet %s failed error: %s",
			strings.ToLower(verb), pod.Name, set.Name, err)
		spc.recorder.Event(set, v1.EventTypeWarning, reason, message)
	}
}

// recordClaimEvent records an event for verb applied to the PersistentVolumeClaim of a Pod in a StatefulSet. If err is
// nil the generated event will have a reason of v1.EventTypeNormal. If err is not nil the generated event will have a
// reason of v1.EventTypeWarning.
func (spc *realStatefulPodControl) recordClaimEvent(verb string, set *apps.StatefulSet, pod *v1.Pod, claim *v1.PersistentVolumeClaim, err error) {
	if err == nil {
		reason := fmt.Sprintf("Successful%s", strings.Title(verb))
		message := fmt.Sprintf("%s Claim %s Pod %s in StatefulSet %s success",
			strings.ToLower(verb), claim.Name, pod.Name, set.Name)
		spc.recorder.Event(set, v1.EventTypeNormal, reason, message)
	} else {
		reason := fmt.Sprintf("Failed%s", strings.Title(verb))
		message := fmt.Sprintf("%s Claim %s for Pod %s in StatefulSet %s failed error: %s",
			strings.ToLower(verb), claim.Name, pod.Name, set.Name, err)
		spc.recorder.Event(set, v1.EventTypeWarning, reason, message)
	}
}

// createPersistentVolumeClaims creates all of the required PersistentVolumeClaims for pod, which mush be a member of
// set. If all of the claims for Pod are successfully created, the returned error is nil. If creation fails, this method
// may be called again until no error is returned, indicating the PersistentVolumeClaims for pod are consistent with
// set's Spec.
func (spc *realStatefulPodControl) createPersistentVolumeClaims(set *apps.StatefulSet, pod *v1.Pod) error {
	var errs []error
	for _, claim := range getPersistentVolumeClaims(set, pod) {
		_, err := spc.client.Core().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				_, err := spc.client.Core().PersistentVolumeClaims(claim.Namespace).Create(&claim)
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
	return errorutils.NewAggregate(errs)
}

var _ StatefulPodControlInterface = &realStatefulPodControl{}
