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
	"errors"
	"fmt"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"sort"
	"testing"
	"time"
)

type requestTracker struct {
	requests int
	err      error
	after    int
}

func (rt *requestTracker) errorReady() bool {
	return rt.err != nil && rt.requests >= rt.after
}

func (rt *requestTracker) inc() {
	rt.requests++
}

func (rt *requestTracker) reset() {
	rt.err = nil
	rt.after = 0
}

type fakePods struct {
	podsMap map[string]v1.Pod
}

func (pods *fakePods) add(pod *v1.Pod) {
	if pod != nil {
		pods.podsMap[pod.Name] = *pod
	}
}

func (pods *fakePods) get(name string) *v1.Pod {
	if pod, contains := pods.podsMap[name]; contains {
		clone := pod
		return &clone
	} else {
		return nil
	}
}

func (pods *fakePods) remove(name string) {
	delete(pods.podsMap, name)
}

func (pods *fakePods) count() int {
	return len(pods.podsMap)
}

func (pods *fakePods) contains(name string) bool {
	_, contains := pods.podsMap[name]
	return contains
}

func (pods *fakePods) list() []*v1.Pod {
	list := make([]*v1.Pod, 0, len(pods.podsMap))
	for _, pod := range pods.podsMap {
		clone := pod
		list = append(list, &clone)
	}
	return list
}

type fakeClaims struct {
	claimsMap map[string]v1.PersistentVolumeClaim
}

func (claims *fakeClaims) add(claim *v1.PersistentVolumeClaim) {
	if claim != nil {
		claims.claimsMap[claim.Name] = *claim
	}
}

func (claims *fakeClaims) get(name string) *v1.PersistentVolumeClaim {
	if claim, contains := claims.claimsMap[name]; contains {
		clone := claim
		return &clone
	} else {
		return nil
	}
}

func (claims *fakeClaims) remove(name string) {
	delete(claims.claimsMap, name)
}

func (claims *fakeClaims) count() int {
	return len(claims.claimsMap)
}

func (claims *fakeClaims) contains(name string) bool {
	_, contains := claims.claimsMap[name]
	return contains
}

func (claims *fakeClaims) list() []*v1.PersistentVolumeClaim {
	list := make([]*v1.PersistentVolumeClaim, 0, len(claims.claimsMap))
	for _, claim := range claims.claimsMap {
		clone := claim
		list = append(list, &clone)
	}
	return list
}

type fakeSets struct {
	setsMap map[string]apps.StatefulSet
}

func (sets *fakeSets) add(set *apps.StatefulSet) {
	if sets != nil {
		sets.setsMap[set.Name] = *set
	}
}

func (sets *fakeSets) get(name string) *apps.StatefulSet {
	if set, contains := sets.setsMap[name]; contains {
		clone := set
		return &clone
	} else {
		return nil
	}
}

func (sets *fakeSets) remove(name string) {
	delete(sets.setsMap, name)
}

func (sets *fakeSets) count() int {
	return len(sets.setsMap)
}

func (sets *fakeSets) contains(name string) bool {
	_, contains := sets.setsMap[name]
	return contains
}

func (sets *fakeSets) list() []*apps.StatefulSet {
	list := make([]*apps.StatefulSet, 0, len(sets.setsMap))
	for _, set := range sets.setsMap {
		clone := set
		list = append(list, &clone)
	}
	return list
}

type fakeStatefulPodControl struct {
	fakePods            map[string]*fakePods
	fakeClaims          map[string]*fakeClaims
	fakeSets            map[string]*fakeSets
	createPodTracker    requestTracker
	updatePodTracker    requestTracker
	deletePodTracker    requestTracker
	updateStatusTracker requestTracker
}

func newFakeStatefulPodControl() *fakeStatefulPodControl {
	return &fakeStatefulPodControl{
		make(map[string]*fakePods),
		make(map[string]*fakeClaims),
		make(map[string]*fakeSets),
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0}}
}

func (spc *fakeStatefulPodControl) pods(namespace string) *fakePods {
	if pods, exists := spc.fakePods[namespace]; exists {
		return pods
	} else {
		pods = &fakePods{make(map[string]v1.Pod)}
		spc.fakePods[namespace] = pods
		return pods
	}
}

func (spc *fakeStatefulPodControl) persistentVolumeClaims(namespace string) *fakeClaims {
	if claims, exists := spc.fakeClaims[namespace]; exists {
		return claims
	} else {
		claims = &fakeClaims{make(map[string]v1.PersistentVolumeClaim)}
		spc.fakeClaims[namespace] = claims
		return claims
	}
}

func (spc *fakeStatefulPodControl) statefulSets(namespace string) *fakeSets {
	if statuses, exists := spc.fakeSets[namespace]; exists {
		return statuses
	} else {
		statuses = &fakeSets{make(map[string]apps.StatefulSet)}
		spc.fakeSets[namespace] = statuses
		return statuses
	}
}

func (spc *fakeStatefulPodControl) SetCreateStatefulPodError(err error, after int) {
	spc.createPodTracker.err = err
	spc.createPodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetUpdateStatefulPodError(err error, after int) {
	spc.updatePodTracker.err = err
	spc.updatePodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetDeleteStatefulPodError(err error, after int) {
	spc.deletePodTracker.err = err
	spc.deletePodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetUpdateStatefulSetStatusError(err error, after int) {
	spc.updateStatusTracker.err = err
	spc.updateStatusTracker.after = after
}

func (spc *fakeStatefulPodControl) CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.createPodTracker.inc()
	if pod == nil || set == nil {
		return nilParameterError
	} else if spc.createPodTracker.errorReady() {
		defer spc.createPodTracker.reset()
		return spc.createPodTracker.err
	}
	spc.pods(set.Namespace).add(pod)
	for _, claim := range getPersistentVolumeClaims(set, pod) {
		spc.persistentVolumeClaims(claim.Namespace).add(&claim)
	}
	return nil
}

func (spc *fakeStatefulPodControl) UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.updatePodTracker.inc()
	if pod == nil || set == nil {
		return nilParameterError
	} else if spc.updatePodTracker.errorReady() {
		defer spc.updatePodTracker.reset()
		return spc.updatePodTracker.err
	}
	if !identityMatches(set, pod) {
		updateIdentity(set, pod)
	}
	if !storageMatches(set, pod) {
		updateStorage(set, pod)
		for _, claim := range getPersistentVolumeClaims(set, pod) {
			spc.persistentVolumeClaims(claim.Namespace).add(&claim)
		}
	}
	spc.pods(set.Namespace).add(pod)
	return nil
}

func (spc *fakeStatefulPodControl) DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.deletePodTracker.inc()
	if pod == nil || set == nil {
		return nilParameterError
	} else if spc.deletePodTracker.errorReady() {
		defer spc.deletePodTracker.reset()
		return spc.deletePodTracker.err
	}
	spc.pods(pod.Namespace).remove(pod.Name)
	return nil
}

func (spc *fakeStatefulPodControl) UpdateStatefulSetStatus(set *apps.StatefulSet) error {
	defer spc.updateStatusTracker.inc()
	if set == nil {
		return nilParameterError
	} else if spc.updateStatusTracker.errorReady() {
		defer spc.updateStatusTracker.reset()
		return spc.updateStatusTracker.err
	}
	spc.statefulSets(set.Namespace).add(set)
	return nil
}

var _ StatefulPodControlInterface = &fakeStatefulPodControl{}

func assertInvariants(set *apps.StatefulSet, control *fakeStatefulPodControl) error {
	pods := control.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	readyCount := 0
	for ord := 0; ord < len(pods); ord++ {
		if isRunningAndReady(pods[ord]) {
			readyCount++
		}
		if ord > 0 && isRunningAndReady(pods[ord]) && !isRunningAndReady(pods[ord-1]) {
			return fmt.Errorf("Predecessor %s is Running and Ready while %s is not",
				pods[ord-1].Name,
				pods[ord].Name)
		}
		if getOrdinal(pods[ord]) != ord {
			return fmt.Errorf("Pods %s deployed in the wrong order",
				pods[ord].Name)
		}
		if !storageMatches(set, pods[ord]) {
			return fmt.Errorf("Pods %s does not match the storage specification of StatefulSet %s ",
				pods[ord].
					Name, set.Name)
		} else {
			for _, claim := range getPersistentVolumeClaims(set, pods[ord]) {
				if !control.persistentVolumeClaims(claim.Namespace).contains(claim.Name) {
					return fmt.Errorf("claim %s for Pod %s was not created",
						claim.Name,
						pods[ord].Name)
				}
			}
		}
		if !identityMatches(set, pods[ord]) {
			return fmt.Errorf("Pods %s does not match the identity specification of StatefulSet %s ",
				pods[ord].Name,
				set.Name)
		}
	}
	if int(set.Status.Replicas) != readyCount {
		return fmt.Errorf("Found %d Pods Running and Ready, StatefulSet %s replicas = %d",
			readyCount, set.Name,
			set.Status.Replicas)
	}
	return nil
}

func setPodRunning(set *apps.StatefulSet, spc *fakeStatefulPodControl, ordinal int) {
	if 0 > ordinal || ordinal >= spc.pods(set.Namespace).count() {
		return
	}
	pods := spc.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[ordinal]
	pod.Status.Phase = v1.PodRunning
	condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
	v1.UpdatePodCondition(&pod.Status, &condition)
	spc.pods(pod.Namespace).add(pod)
}

func addTerminatedPod(set *apps.StatefulSet, spc *fakeStatefulPodControl, ordinal int) {
	pods := spc.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	pod := newStatefulSetPod(set, ordinal)
	pod.Status.Phase = v1.PodRunning
	deleted := metav1.NewTime(time.Now())
	pod.DeletionTimestamp = &deleted
	spc.pods(pod.Namespace).add(pod)
}

func TestDefaultStatefulSetControlCreatesPods(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestStatefulSetControlScaleUp(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	*set.Spec.Replicas = 4
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if set.Status.Replicas != 4 {
		t.Error("Falied to scale statefulset to 4 replicas")
	}
}

func TestStatefulSetControlScaleDown(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	*set.Spec.Replicas = 0
	for set.Status.Replicas > *set.Spec.Replicas {
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		addTerminatedPod(set, spc, spc.pods(set.Namespace).count())
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		pods := spc.pods(set.Namespace).list()
		if target := len(pods) - 1; target >= 0 {
			sort.Sort(ascendingOrdinal(pods))
			spc.pods(set.Namespace).remove(pods[target].Name)
		}
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestStatefulSetControlReplacesPods(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(5)
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods := spc.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	spc.pods(set.Namespace).remove(pods[2].Name)
	spc.pods(set.Namespace).remove(pods[4].Name)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	setPodRunning(set, spc, 2)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	setPodRunning(set, spc, 4)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestDefaultStatefulSetControlInitAnnotation(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	pod := spc.pods(set.Namespace).list()[0]
	pod.Annotations[StatefulSetInitAnnotation] = "false"
	spc.pods(set.Namespace).add(pod)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	replicas := int(set.Status.Replicas)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if replicas != int(set.Status.Replicas) {
		t.Errorf("StatefulSetControl does not block on %d=false", StatefulSetInitAnnotation)
	}
	pod = spc.pods(set.Namespace).list()[0]
	pod.Annotations[StatefulSetInitAnnotation] = "true"
	spc.pods(set.Namespace).add(pod)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if replicas+1 != int(set.Status.Replicas) {
		t.Errorf("StatefulSetControl continues to block after %d=true", StatefulSetInitAnnotation)
	}
}

func TestDefaultStatefulSetControlCreatePodFailure(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	spc.SetCreateStatefulPodError(
		apierrors.NewInternalError(errors.New("API server failed")),
		2)
	errorReturned := false
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); apierrors.IsInternalError(err) {
			errorReturned = true
		} else if err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if !errorReturned {
		t.Error("StatefulSetControl failed to return error thrown from PodControl")
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestDefaultStatefulSetControlUpdatePodFailure(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	spc.SetUpdateStatefulPodError(
		apierrors.NewInternalError(errors.New("API server failed")),
		2)
	errorReturned := false
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); apierrors.IsInternalError(err) {
			errorReturned = true
		} else if err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if !errorReturned {
		t.Error("StatefulSetControl failed to return error thrown from PodControl")
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestDefaultStatefulSetControlStatusUpdateFailure(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	spc.SetUpdateStatefulSetStatusError(
		apierrors.NewInternalError(errors.New("API server failed")),
		2)
	errorReturned := false
	for set.Status.Replicas < *set.Spec.Replicas {
		setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
		if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); apierrors.IsInternalError(err) {
			errorReturned = true
		} else if err != nil {
			t.Errorf("Error updating StatefulSet %s", err)
		}
		if err := assertInvariants(set, spc); err != nil {
			t.Error(err)
		}
	}
	if !errorReturned {
		t.Error("StatefulSetControl failed to return error thrown from PodControl")
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
}

func TestDefaultStatefulSetControlPodRecreatesFailedPod(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods := spc.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[len(pods)-1]
	pod.Status.Phase = v1.PodFailed
	spc.pods(set.Namespace).add(pod)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if replicas := int(set.Status.Replicas); replicas != 2 {
		t.Errorf("Expected 2 replicas after Pod recreation found %d", replicas)
	}
}

func TestDefaultStatefulSetControlPodRecreateDeleteError(t *testing.T) {
	spc := newFakeStatefulPodControl()
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	spc.SetDeleteStatefulPodError(
		apierrors.NewInternalError(errors.New("API server failed")),
		0)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods := spc.pods(set.Namespace).list()
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[len(pods)-1]
	pod.Status.Phase = v1.PodFailed
	spc.pods(set.Namespace).add(pod)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); !apierrors.IsInternalError(err) {
		t.Errorf("Expected InternalError for Pod deletion when Pod is failed found %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	setPodRunning(set, spc, spc.pods(set.Namespace).count()-1)
	if err := ssc.UpdateStatefulSet(set, spc.pods(set.Namespace).list()); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if replicas := int(set.Status.Replicas); replicas != 2 {
		t.Errorf("Expected 2 replicas after Pod recreation found %d", replicas)
	}
}
