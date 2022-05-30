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
	"math/rand"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/history"
)

// noopRecorder is an EventRecorder that does nothing. record.FakeRecorder has a fixed
// buffer size, which causes tests to hang if that buffer's exceeded.
type noopRecorder struct{}

func (r *noopRecorder) Event(object runtime.Object, eventtype, reason, message string) {}
func (r *noopRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
}
func (r *noopRecorder) AnnotatedEventf(object runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
}

// getClaimPodName gets the name of the Pod associated with the Claim, or an empty string if this doesn't look matching.
func getClaimPodName(set *apps.StatefulSet, claim *v1.PersistentVolumeClaim) string {
	podName := ""

	statefulClaimRegex := regexp.MustCompile(fmt.Sprintf(".*-(%s-[0-9]+)$", set.Name))
	matches := statefulClaimRegex.FindStringSubmatch(claim.Name)
	if len(matches) != 2 {
		return podName
	}
	return matches[1]
}

func TestGetParentNameAndOrdinal(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if parent, ordinal := getParentNameAndOrdinal(pod); parent != set.Name {
		t.Errorf("Extracted the wrong parent name expected %s found %s", set.Name, parent)
	} else if ordinal != 1 {
		t.Errorf("Extracted the wrong ordinal expected %d found %d", 1, ordinal)
	}
	pod.Name = "1-bar"
	if parent, ordinal := getParentNameAndOrdinal(pod); parent != "" {
		t.Error("Expected empty string for non-member Pod parent")
	} else if ordinal != -1 {
		t.Error("Expected -1 for non member Pod ordinal")
	}
}

func TestGetClaimPodName(t *testing.T) {
	set := apps.StatefulSet{}
	set.Name = "my-set"
	claim := v1.PersistentVolumeClaim{}
	claim.Name = "volume-my-set-2"
	if pod := getClaimPodName(&set, &claim); pod != "my-set-2" {
		t.Errorf("Expected my-set-2 found %s", pod)
	}
	claim.Name = "long-volume-my-set-20"
	if pod := getClaimPodName(&set, &claim); pod != "my-set-20" {
		t.Errorf("Expected my-set-20 found %s", pod)
	}
	claim.Name = "volume-2-my-set"
	if pod := getClaimPodName(&set, &claim); pod != "" {
		t.Errorf("Expected empty string found %s", pod)
	}
	claim.Name = "volume-pod-2"
	if pod := getClaimPodName(&set, &claim); pod != "" {
		t.Errorf("Expected empty string found %s", pod)
	}
}

func TestIsMemberOf(t *testing.T) {
	set := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod := newStatefulSetPod(set, 1)
	if !isMemberOf(set, pod) {
		t.Error("isMemberOf returned false negative")
	}
	if isMemberOf(set2, pod) {
		t.Error("isMemberOf returned false positive")
	}
}

func TestIdentityMatches(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !identityMatches(set, pod) {
		t.Error("Newly created Pod has a bad identity")
	}
	pod.Name = "foo"
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with the wrong name")
	}
	pod = newStatefulSetPod(set, 1)
	pod.Namespace = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with the wrong namespace")
	}
	pod = newStatefulSetPod(set, 1)
	delete(pod.Labels, apps.StatefulSetPodNameLabel)
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with the wrong statefulSetPodNameLabel")
	}
}

func TestStorageMatches(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !storageMatches(set, pod) {
		t.Error("Newly created Pod has a invalid storage")
	}
	pod.Spec.Volumes = nil
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes has valid storage")
	}
	pod = newStatefulSetPod(set, 1)
	for i := range pod.Spec.Volumes {
		pod.Spec.Volumes[i].PersistentVolumeClaim = nil
	}
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes claim valid storage")
	}
	pod = newStatefulSetPod(set, 1)
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].PersistentVolumeClaim != nil {
			pod.Spec.Volumes[i].PersistentVolumeClaim.ClaimName = "foo"
		}
	}
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes claim valid storage")
	}
	pod = newStatefulSetPod(set, 1)
	pod.Name = "bar"
	if storageMatches(set, pod) {
		t.Error("Pod with invalid ordinal has valid storage")
	}
}

func TestUpdateIdentity(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !identityMatches(set, pod) {
		t.Error("Newly created Pod has a bad identity")
	}
	pod.Namespace = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with the wrong namespace")
	}
	updateIdentity(set, pod)
	if !identityMatches(set, pod) {
		t.Error("updateIdentity failed to update the Pods namespace")
	}
	delete(pod.Labels, apps.StatefulSetPodNameLabel)
	updateIdentity(set, pod)
	if !identityMatches(set, pod) {
		t.Error("updateIdentity failed to restore the statefulSetPodName label")
	}
}

func TestUpdateStorage(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !storageMatches(set, pod) {
		t.Error("Newly created Pod has a invalid storage")
	}
	pod.Spec.Volumes = nil
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes has valid storage")
	}
	updateStorage(set, pod)
	if !storageMatches(set, pod) {
		t.Error("updateStorage failed to recreate volumes")
	}
	pod = newStatefulSetPod(set, 1)
	for i := range pod.Spec.Volumes {
		pod.Spec.Volumes[i].PersistentVolumeClaim = nil
	}
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes claim valid storage")
	}
	updateStorage(set, pod)
	if !storageMatches(set, pod) {
		t.Error("updateStorage failed to recreate volume claims")
	}
	pod = newStatefulSetPod(set, 1)
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].PersistentVolumeClaim != nil {
			pod.Spec.Volumes[i].PersistentVolumeClaim.ClaimName = "foo"
		}
	}
	if storageMatches(set, pod) {
		t.Error("Pod with invalid Volumes claim valid storage")
	}
	updateStorage(set, pod)
	if !storageMatches(set, pod) {
		t.Error("updateStorage failed to recreate volume claim names")
	}
}

func TestGetPersistentVolumeClaimRetentionPolicy(t *testing.T) {
	retainPolicy := apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
		WhenScaled:  apps.RetainPersistentVolumeClaimRetentionPolicyType,
		WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
	}
	scaledownPolicy := apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
		WhenScaled:  apps.DeletePersistentVolumeClaimRetentionPolicyType,
		WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
	}

	set := apps.StatefulSet{}
	set.Spec.PersistentVolumeClaimRetentionPolicy = &retainPolicy
	got := getPersistentVolumeClaimRetentionPolicy(&set)
	if got.WhenScaled != apps.RetainPersistentVolumeClaimRetentionPolicyType || got.WhenDeleted != apps.RetainPersistentVolumeClaimRetentionPolicyType {
		t.Errorf("Expected retain policy")
	}
	set.Spec.PersistentVolumeClaimRetentionPolicy = &scaledownPolicy
	got = getPersistentVolumeClaimRetentionPolicy(&set)
	if got.WhenScaled != apps.DeletePersistentVolumeClaimRetentionPolicyType || got.WhenDeleted != apps.RetainPersistentVolumeClaimRetentionPolicyType {
		t.Errorf("Expected scaledown policy")
	}
}

func TestClaimOwnerMatchesSetAndPod(t *testing.T) {
	testCases := []struct {
		name            string
		scaleDownPolicy apps.PersistentVolumeClaimRetentionPolicyType
		setDeletePolicy apps.PersistentVolumeClaimRetentionPolicyType
		needsPodRef     bool
		needsSetRef     bool
		replicas        int32
		ordinal         int
	}{
		{
			name:            "retain",
			scaleDownPolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     false,
			needsSetRef:     false,
		},
		{
			name:            "on SS delete",
			scaleDownPolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     false,
			needsSetRef:     true,
		},
		{
			name:            "on scaledown only, condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     true,
			needsSetRef:     false,
			replicas:        2,
			ordinal:         2,
		},
		{
			name:            "on scaledown only, remains",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     false,
			needsSetRef:     false,
			replicas:        2,
			ordinal:         1,
		},
		{
			name:            "on both, condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     true,
			needsSetRef:     false,
			replicas:        2,
			ordinal:         2,
		},
		{
			name:            "on both, remains",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			needsPodRef:     false,
			needsSetRef:     true,
			replicas:        2,
			ordinal:         1,
		},
	}

	for _, tc := range testCases {
		for _, useOtherRefs := range []bool{false, true} {
			for _, setPodRef := range []bool{false, true} {
				for _, setSetRef := range []bool{false, true} {
					claim := v1.PersistentVolumeClaim{}
					claim.Name = "target-claim"
					pod := v1.Pod{}
					pod.Name = fmt.Sprintf("pod-%d", tc.ordinal)
					pod.GetObjectMeta().SetUID("pod-123")
					set := apps.StatefulSet{}
					set.Name = "stateful-set"
					set.GetObjectMeta().SetUID("ss-456")
					set.Spec.PersistentVolumeClaimRetentionPolicy = &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
						WhenScaled:  tc.scaleDownPolicy,
						WhenDeleted: tc.setDeletePolicy,
					}
					set.Spec.Replicas = &tc.replicas
					if setPodRef {
						setOwnerRef(&claim, &pod, &pod.TypeMeta)
					}
					if setSetRef {
						setOwnerRef(&claim, &set, &set.TypeMeta)
					}
					if useOtherRefs {
						randomObject1 := v1.Pod{}
						randomObject1.Name = "rand1"
						randomObject1.GetObjectMeta().SetUID("rand1-abc")
						randomObject2 := v1.Pod{}
						randomObject2.Name = "rand2"
						randomObject2.GetObjectMeta().SetUID("rand2-def")
						setOwnerRef(&claim, &randomObject1, &randomObject1.TypeMeta)
						setOwnerRef(&claim, &randomObject2, &randomObject2.TypeMeta)
					}
					shouldMatch := setPodRef == tc.needsPodRef && setSetRef == tc.needsSetRef
					if claimOwnerMatchesSetAndPod(&claim, &set, &pod) != shouldMatch {
						t.Errorf("Bad match for %s with pod=%v,set=%v,others=%v", tc.name, setPodRef, setSetRef, useOtherRefs)
					}
				}
			}
		}
	}
}

func TestUpdateClaimOwnerRefForSetAndPod(t *testing.T) {
	testCases := []struct {
		name            string
		scaleDownPolicy apps.PersistentVolumeClaimRetentionPolicyType
		setDeletePolicy apps.PersistentVolumeClaimRetentionPolicyType
		condemned       bool
		needsPodRef     bool
		needsSetRef     bool
	}{
		{
			name:            "retain",
			scaleDownPolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			condemned:       false,
			needsPodRef:     false,
			needsSetRef:     false,
		},
		{
			name:            "delete with set",
			scaleDownPolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			condemned:       false,
			needsPodRef:     false,
			needsSetRef:     true,
		},
		{
			name:            "delete with scaledown, not condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			condemned:       false,
			needsPodRef:     false,
			needsSetRef:     false,
		},
		{
			name:            "delete on scaledown, condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			condemned:       true,
			needsPodRef:     true,
			needsSetRef:     false,
		},
		{
			name:            "delete on both, not condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			condemned:       false,
			needsPodRef:     false,
			needsSetRef:     true,
		},
		{
			name:            "delete on both, condemned",
			scaleDownPolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			setDeletePolicy: apps.DeletePersistentVolumeClaimRetentionPolicyType,
			condemned:       true,
			needsPodRef:     true,
			needsSetRef:     false,
		},
	}
	for _, tc := range testCases {
		for _, hasPodRef := range []bool{true, false} {
			for _, hasSetRef := range []bool{true, false} {
				set := apps.StatefulSet{}
				set.Name = "ss"
				numReplicas := int32(5)
				set.Spec.Replicas = &numReplicas
				set.SetUID("ss-123")
				set.Spec.PersistentVolumeClaimRetentionPolicy = &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
					WhenScaled:  tc.scaleDownPolicy,
					WhenDeleted: tc.setDeletePolicy,
				}
				pod := v1.Pod{}
				if tc.condemned {
					pod.Name = "pod-8"
				} else {
					pod.Name = "pod-1"
				}
				pod.SetUID("pod-456")
				claim := v1.PersistentVolumeClaim{}
				if hasPodRef {
					setOwnerRef(&claim, &pod, &pod.TypeMeta)
				}
				if hasSetRef {
					setOwnerRef(&claim, &set, &set.TypeMeta)
				}
				needsUpdate := hasPodRef != tc.needsPodRef || hasSetRef != tc.needsSetRef
				shouldUpdate := updateClaimOwnerRefForSetAndPod(&claim, &set, &pod)
				if shouldUpdate != needsUpdate {
					t.Errorf("Bad update for %s hasPodRef=%v hasSetRef=%v", tc.name, hasPodRef, hasSetRef)
				}
				if hasOwnerRef(&claim, &pod) != tc.needsPodRef {
					t.Errorf("Bad pod ref for %s hasPodRef=%v hasSetRef=%v", tc.name, hasPodRef, hasSetRef)
				}
				if hasOwnerRef(&claim, &set) != tc.needsSetRef {
					t.Errorf("Bad set ref for %s hasPodRef=%v hasSetRef=%v", tc.name, hasPodRef, hasSetRef)
				}
			}
		}
	}
}

func TestHasOwnerRef(t *testing.T) {
	target := v1.Pod{}
	target.SetOwnerReferences([]metav1.OwnerReference{
		{UID: "123"}, {UID: "456"}})
	ownerA := v1.Pod{}
	ownerA.GetObjectMeta().SetUID("123")
	ownerB := v1.Pod{}
	ownerB.GetObjectMeta().SetUID("789")
	if !hasOwnerRef(&target, &ownerA) {
		t.Error("Missing owner")
	}
	if hasOwnerRef(&target, &ownerB) {
		t.Error("Unexpected owner")
	}
}

func TestHasStaleOwnerRef(t *testing.T) {
	target := v1.Pod{}
	target.SetOwnerReferences([]metav1.OwnerReference{
		{Name: "bob", UID: "123"}, {Name: "shirley", UID: "456"}})
	ownerA := v1.Pod{}
	ownerA.SetUID("123")
	ownerA.Name = "bob"
	ownerB := v1.Pod{}
	ownerB.Name = "shirley"
	ownerB.SetUID("789")
	ownerC := v1.Pod{}
	ownerC.Name = "yvonne"
	ownerC.SetUID("345")
	if hasStaleOwnerRef(&target, &ownerA) {
		t.Error("ownerA should not be stale")
	}
	if !hasStaleOwnerRef(&target, &ownerB) {
		t.Error("ownerB should be stale")
	}
	if hasStaleOwnerRef(&target, &ownerC) {
		t.Error("ownerC should not be stale")
	}
}

func TestSetOwnerRef(t *testing.T) {
	target := v1.Pod{}
	ownerA := v1.Pod{}
	ownerA.Name = "A"
	ownerA.GetObjectMeta().SetUID("ABC")
	if setOwnerRef(&target, &ownerA, &ownerA.TypeMeta) != true {
		t.Errorf("Unexpected lack of update")
	}
	ownerRefs := target.GetObjectMeta().GetOwnerReferences()
	if len(ownerRefs) != 1 {
		t.Errorf("Unexpected owner ref count: %d", len(ownerRefs))
	}
	if ownerRefs[0].UID != "ABC" {
		t.Errorf("Unexpected owner UID %v", ownerRefs[0].UID)
	}
	if setOwnerRef(&target, &ownerA, &ownerA.TypeMeta) != false {
		t.Errorf("Unexpected update")
	}
	if len(target.GetObjectMeta().GetOwnerReferences()) != 1 {
		t.Error("Unexpected duplicate reference")
	}
	ownerB := v1.Pod{}
	ownerB.Name = "B"
	ownerB.GetObjectMeta().SetUID("BCD")
	if setOwnerRef(&target, &ownerB, &ownerB.TypeMeta) != true {
		t.Error("Unexpected lack of second update")
	}
	ownerRefs = target.GetObjectMeta().GetOwnerReferences()
	if len(ownerRefs) != 2 {
		t.Errorf("Unexpected owner ref count: %d", len(ownerRefs))
	}
	if ownerRefs[0].UID != "ABC" || ownerRefs[1].UID != "BCD" {
		t.Errorf("Bad second ownerRefs: %v", ownerRefs)
	}
}

func TestRemoveOwnerRef(t *testing.T) {
	target := v1.Pod{}
	ownerA := v1.Pod{}
	ownerA.Name = "A"
	ownerA.GetObjectMeta().SetUID("ABC")
	if removeOwnerRef(&target, &ownerA) != false {
		t.Error("Unexpected update on empty remove")
	}
	setOwnerRef(&target, &ownerA, &ownerA.TypeMeta)
	if removeOwnerRef(&target, &ownerA) != true {
		t.Error("Unexpected lack of update")
	}
	if len(target.GetObjectMeta().GetOwnerReferences()) != 0 {
		t.Error("Unexpected owner reference remains")
	}

	ownerB := v1.Pod{}
	ownerB.Name = "B"
	ownerB.GetObjectMeta().SetUID("BCD")

	setOwnerRef(&target, &ownerA, &ownerA.TypeMeta)
	if removeOwnerRef(&target, &ownerB) != false {
		t.Error("Unexpected update for mismatched owner")
	}
	if len(target.GetObjectMeta().GetOwnerReferences()) != 1 {
		t.Error("Missing ref after no-op remove")
	}
	setOwnerRef(&target, &ownerB, &ownerB.TypeMeta)
	if removeOwnerRef(&target, &ownerA) != true {
		t.Error("Missing update for second remove")
	}
	ownerRefs := target.GetObjectMeta().GetOwnerReferences()
	if len(ownerRefs) != 1 {
		t.Error("Extra ref after second remove")
	}
	if ownerRefs[0].UID != "BCD" {
		t.Error("Bad UID after second remove")
	}
}

func TestIsRunningAndReady(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if isRunningAndReady(pod) {
		t.Error("isRunningAndReady does not respect Pod phase")
	}
	pod.Status.Phase = v1.PodRunning
	if isRunningAndReady(pod) {
		t.Error("isRunningAndReady does not respect Pod condition")
	}
	condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
	podutil.UpdatePodCondition(&pod.Status, &condition)
	if !isRunningAndReady(pod) {
		t.Error("Pod should be running and ready")
	}
}

func TestAscendingOrdinal(t *testing.T) {
	set := newStatefulSet(10)
	pods := make([]*v1.Pod, 10)
	perm := rand.Perm(10)
	for i, v := range perm {
		pods[i] = newStatefulSetPod(set, v)
	}
	sort.Sort(ascendingOrdinal(pods))
	if !sort.IsSorted(ascendingOrdinal(pods)) {
		t.Error("ascendingOrdinal fails to sort Pods")
	}
}

func TestOverlappingStatefulSets(t *testing.T) {
	sets := make([]*apps.StatefulSet, 10)
	perm := rand.Perm(10)
	for i, v := range perm {
		sets[i] = newStatefulSet(10)
		sets[i].CreationTimestamp = metav1.NewTime(sets[i].CreationTimestamp.Add(time.Duration(v) * time.Second))
	}
	sort.Sort(overlappingStatefulSets(sets))
	if !sort.IsSorted(overlappingStatefulSets(sets)) {
		t.Error("ascendingOrdinal fails to sort Pods")
	}
	for i, v := range perm {
		sets[i] = newStatefulSet(10)
		sets[i].Name = strconv.FormatInt(int64(v), 10)
	}
	sort.Sort(overlappingStatefulSets(sets))
	if !sort.IsSorted(overlappingStatefulSets(sets)) {
		t.Error("ascendingOrdinal fails to sort Pods")
	}
}

func TestNewPodControllerRef(t *testing.T) {
	set := newStatefulSet(1)
	pod := newStatefulSetPod(set, 0)
	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil {
		t.Fatalf("No ControllerRef found on new pod")
	}
	if got, want := controllerRef.APIVersion, apps.SchemeGroupVersion.String(); got != want {
		t.Errorf("controllerRef.APIVersion = %q, want %q", got, want)
	}
	if got, want := controllerRef.Kind, "StatefulSet"; got != want {
		t.Errorf("controllerRef.Kind = %q, want %q", got, want)
	}
	if got, want := controllerRef.Name, set.Name; got != want {
		t.Errorf("controllerRef.Name = %q, want %q", got, want)
	}
	if got, want := controllerRef.UID, set.UID; got != want {
		t.Errorf("controllerRef.UID = %q, want %q", got, want)
	}
	if got, want := *controllerRef.Controller, true; got != want {
		t.Errorf("controllerRef.Controller = %v, want %v", got, want)
	}
}

func TestCreateApplyRevision(t *testing.T) {
	set := newStatefulSet(1)
	set.Status.CollisionCount = new(int32)
	revision, err := newRevision(set, 1, set.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}
	set.Spec.Template.Spec.Containers[0].Name = "foo"
	if set.Annotations == nil {
		set.Annotations = make(map[string]string)
	}
	key := "foo"
	expectedValue := "bar"
	set.Annotations[key] = expectedValue
	restoredSet, err := ApplyRevision(set, revision)
	if err != nil {
		t.Fatal(err)
	}
	restoredRevision, err := newRevision(restoredSet, 2, restoredSet.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}
	if !history.EqualRevision(revision, restoredRevision) {
		t.Errorf("wanted %v got %v", string(revision.Data.Raw), string(restoredRevision.Data.Raw))
	}
	value, ok := restoredRevision.Annotations[key]
	if !ok {
		t.Errorf("missing annotation %s", key)
	}
	if value != expectedValue {
		t.Errorf("for annotation %s wanted %s got %s", key, expectedValue, value)
	}
}

func TestRollingUpdateApplyRevision(t *testing.T) {
	set := newStatefulSet(1)
	set.Status.CollisionCount = new(int32)
	currentSet := set.DeepCopy()
	currentRevision, err := newRevision(set, 1, set.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}

	set.Spec.Template.Spec.Containers[0].Env = []v1.EnvVar{{Name: "foo", Value: "bar"}}
	updateSet := set.DeepCopy()
	updateRevision, err := newRevision(set, 2, set.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}

	restoredCurrentSet, err := ApplyRevision(set, currentRevision)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(currentSet.Spec.Template, restoredCurrentSet.Spec.Template) {
		t.Errorf("want %v got %v", currentSet.Spec.Template, restoredCurrentSet.Spec.Template)
	}

	restoredUpdateSet, err := ApplyRevision(set, updateRevision)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(updateSet.Spec.Template, restoredUpdateSet.Spec.Template) {
		t.Errorf("want %v got %v", updateSet.Spec.Template, restoredUpdateSet.Spec.Template)
	}
}

func TestGetPersistentVolumeClaims(t *testing.T) {

	// nil inherits statefulset labels
	pod := newPod()
	statefulSet := newStatefulSet(1)
	statefulSet.Spec.Selector.MatchLabels = nil
	claims := getPersistentVolumeClaims(statefulSet, pod)
	pvc := newPVC("datadir-foo-0")
	resultClaims := map[string]v1.PersistentVolumeClaim{"datadir": pvc}

	if !reflect.DeepEqual(claims, resultClaims) {
		t.Fatalf("Unexpected pvc:\n %+v\n, desired pvc:\n %+v", claims, resultClaims)
	}

	// nil inherits statefulset labels
	statefulSet.Spec.Selector.MatchLabels = map[string]string{"test": "test"}
	claims = getPersistentVolumeClaims(statefulSet, pod)
	pvc.SetLabels(map[string]string{"test": "test"})
	resultClaims = map[string]v1.PersistentVolumeClaim{"datadir": pvc}
	if !reflect.DeepEqual(claims, resultClaims) {
		t.Fatalf("Unexpected pvc:\n %+v\n, desired pvc:\n %+v", claims, resultClaims)
	}

	// non-nil with non-overlapping labels merge pvc and statefulset labels
	statefulSet.Spec.Selector.MatchLabels = map[string]string{"name": "foo"}
	statefulSet.Spec.VolumeClaimTemplates[0].ObjectMeta.Labels = map[string]string{"test": "test"}
	claims = getPersistentVolumeClaims(statefulSet, pod)
	pvc.SetLabels(map[string]string{"test": "test", "name": "foo"})
	resultClaims = map[string]v1.PersistentVolumeClaim{"datadir": pvc}
	if !reflect.DeepEqual(claims, resultClaims) {
		t.Fatalf("Unexpected pvc:\n %+v\n, desired pvc:\n %+v", claims, resultClaims)
	}

	// non-nil with overlapping labels merge pvc and statefulset labels and prefer statefulset labels
	statefulSet.Spec.Selector.MatchLabels = map[string]string{"test": "foo"}
	statefulSet.Spec.VolumeClaimTemplates[0].ObjectMeta.Labels = map[string]string{"test": "test"}
	claims = getPersistentVolumeClaims(statefulSet, pod)
	pvc.SetLabels(map[string]string{"test": "foo"})
	resultClaims = map[string]v1.PersistentVolumeClaim{"datadir": pvc}
	if !reflect.DeepEqual(claims, resultClaims) {
		t.Fatalf("Unexpected pvc:\n %+v\n, desired pvc:\n %+v", claims, resultClaims)
	}
}

func newPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo-0",
			Namespace: v1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: "nginx",
				},
			},
		},
	}
}

func newPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: v1.NamespaceDefault,
			Name:      name,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

func newStatefulSetWithVolumes(replicas int, name string, petMounts []v1.VolumeMount, podMounts []v1.VolumeMount) *apps.StatefulSet {
	mounts := append(petMounts, podMounts...)
	claims := []v1.PersistentVolumeClaim{}
	for _, m := range petMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []v1.Volume{}
	for _, m := range podMounts {
		vols = append(vols, v1.Volume{
			Name: m.Name,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	template := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:         "nginx",
					Image:        "nginx",
					VolumeMounts: mounts,
				},
			},
			Volumes: vols,
		},
	}

	template.Labels = map[string]string{"foo": "bar"}

	return &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			UID:       types.UID("test"),
		},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Replicas:             func() *int32 { i := int32(replicas); return &i }(),
			Template:             template,
			VolumeClaimTemplates: claims,
			ServiceName:          "governingsvc",
			UpdateStrategy:       apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			PersistentVolumeClaimRetentionPolicy: &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenScaled:  apps.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			RevisionHistoryLimit: func() *int32 {
				limit := int32(2)
				return &limit
			}(),
		},
	}
}

func newStatefulSet(replicas int) *apps.StatefulSet {
	petMounts := []v1.VolumeMount{
		{Name: "datadir", MountPath: "/tmp/zookeeper"},
	}
	podMounts := []v1.VolumeMount{
		{Name: "home", MountPath: "/home"},
	}
	return newStatefulSetWithVolumes(replicas, "foo", petMounts, podMounts)
}

func newStatefulSetWithLabels(replicas int, name string, uid types.UID, labels map[string]string) *apps.StatefulSet {
	// Converting all the map-only selectors to set-based selectors.
	var testMatchExpressions []metav1.LabelSelectorRequirement
	for key, value := range labels {
		sel := metav1.LabelSelectorRequirement{
			Key:      key,
			Operator: metav1.LabelSelectorOpIn,
			Values:   []string{value},
		}
		testMatchExpressions = append(testMatchExpressions, sel)
	}
	return &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			UID:       uid,
		},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				// Purposely leaving MatchLabels nil, so to ensure it will break if any link
				// in the chain ignores the set-based MatchExpressions.
				MatchLabels:      nil,
				MatchExpressions: testMatchExpressions,
			},
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			PersistentVolumeClaimRetentionPolicy: &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenScaled:  apps.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "nginx",
							VolumeMounts: []v1.VolumeMount{
								{Name: "datadir", MountPath: "/tmp/"},
								{Name: "home", MountPath: "/home"},
							},
						},
					},
					Volumes: []v1.Volume{{
						Name: "home",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: fmt.Sprintf("/tmp/%v", "home"),
							},
						}}},
				},
			},
			VolumeClaimTemplates: []v1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "datadir"},
					Spec: v1.PersistentVolumeClaimSpec{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
							},
						},
					},
				},
			},
			ServiceName: "governingsvc",
		},
	}
}

func TestGetStatefulSetMaxUnavailable(t *testing.T) {
	testCases := []struct {
		maxUnavailable         *intstr.IntOrString
		replicaCount           int
		expectedMaxUnavailable int
	}{
		// it wouldn't hurt to also test 0 and 0%, even if they should have been forbidden by API validation.
		{maxUnavailable: nil, replicaCount: 10, expectedMaxUnavailable: 1},
		{maxUnavailable: intOrStrP(intstr.FromInt(3)), replicaCount: 10, expectedMaxUnavailable: 3},
		{maxUnavailable: intOrStrP(intstr.FromInt(3)), replicaCount: 0, expectedMaxUnavailable: 3},
		{maxUnavailable: intOrStrP(intstr.FromInt(0)), replicaCount: 0, expectedMaxUnavailable: 1},
		{maxUnavailable: intOrStrP(intstr.FromString("10%")), replicaCount: 25, expectedMaxUnavailable: 2},
		{maxUnavailable: intOrStrP(intstr.FromString("100%")), replicaCount: 5, expectedMaxUnavailable: 5},
		{maxUnavailable: intOrStrP(intstr.FromString("50%")), replicaCount: 5, expectedMaxUnavailable: 2},
		{maxUnavailable: intOrStrP(intstr.FromString("10%")), replicaCount: 5, expectedMaxUnavailable: 1},
		{maxUnavailable: intOrStrP(intstr.FromString("1%")), replicaCount: 0, expectedMaxUnavailable: 1},
		{maxUnavailable: intOrStrP(intstr.FromString("0%")), replicaCount: 0, expectedMaxUnavailable: 1},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("case %d", i), func(t *testing.T) {
			gotMaxUnavailable, err := getStatefulSetMaxUnavailable(tc.maxUnavailable, tc.replicaCount)
			if err != nil {
				t.Fatal(err)
			}
			if gotMaxUnavailable != tc.expectedMaxUnavailable {
				t.Errorf("Expected maxUnavailable %v, got pods %v", tc.expectedMaxUnavailable, gotMaxUnavailable)
			}
		})
	}
}

func intOrStrP(v intstr.IntOrString) *intstr.IntOrString {
	return &v
}
