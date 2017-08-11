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
	"sort"
	"strconv"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/history"
)

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

func TestIsMemberOf(t *testing.T) {
	set := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod := newStatefulSetPod(set, 1)
	if !isMemberOf(set, pod) {
		t.Error("isMemberOf retruned false negative")
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
	pod.Spec.Hostname = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with no hostname")
	}
	pod = newStatefulSetPod(set, 1)
	pod.Spec.Subdomain = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with no subdomain")
	}
}

func TestStorageMatches(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !storageMatches(set, pod) {
		t.Error("Newly created Pod has a invalid stroage")
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
	pod = newStatefulSetPod(set, 1)
	pod.Spec.Hostname = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with no hostname")
	}
	updateIdentity(set, pod)
	if !identityMatches(set, pod) {
		t.Error("updateIdentity failed to update the Pod's hostname")
	}
	pod = newStatefulSetPod(set, 1)
	pod.Spec.Subdomain = ""
	if identityMatches(set, pod) {
		t.Error("identity matches for a Pod with no subdomain")
	}
	updateIdentity(set, pod)
	if !identityMatches(set, pod) {
		t.Error("updateIdentity failed to update the Pod's subdomain")
	}
}

func TestUpdateStorage(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 1)
	if !storageMatches(set, pod) {
		t.Error("Newly created Pod has a invalid stroage")
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
	revision, err := newRevision(set, 1)
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
	restoredSet, err := applyRevision(set, revision)
	if err != nil {
		t.Fatal(err)
	}
	restoredRevision, err := newRevision(restoredSet, 2)
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

func newPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
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
			APIVersion: "apps/v1beta1",
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
