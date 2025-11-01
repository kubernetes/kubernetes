/*
Copyright 2025 The Kubernetes Authors.

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
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/utils/ptr"
)

func TestStatefulSetCompatibility(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// These are what running statefulset/pod/pvc/controllerrevision objects look like from 1.33
	set := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			CreationTimestamp: metav1.Time{Time: time.Now().Add(-time.Hour)},
			Generation:        1,
			Labels:            map[string]string{"sslabel": "value"},
			Name:              "test",
			Namespace:         "test",
			ResourceVersion:   "123",
			UID:               "ssuid",
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"test": "value"},
					Labels:      map[string]string{"app": "foo"},
				},
				Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "test", Name: "test"}}}},
			VolumeClaimTemplates: []corev1.PersistentVolumeClaim{{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"key": "value"},
					Labels:      map[string]string{"key": "value"},
					Name:        "test",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
					Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{"storage": resource.MustParse("1Gi")}},
				},
			}},
		},
		Status: appsv1.StatefulSetStatus{
			AvailableReplicas:  1,
			CollisionCount:     ptr.To(int32(0)),
			CurrentReplicas:    1,
			CurrentRevision:    "test-c77f6d978",
			ObservedGeneration: 1,
			Replicas:           1,
			UpdateRevision:     "test-c77f6d978",
			UpdatedReplicas:    1,
			ReadyReplicas:      1,
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations:       map[string]string{"test": "value"},
			CreationTimestamp: metav1.Time{Time: time.Now().Add(-time.Hour)},
			GenerateName:      "test-",
			Generation:        1,
			Labels: map[string]string{
				"app":                                "foo",
				"apps.kubernetes.io/pod-index":       "0",
				"controller-revision-hash":           "test-c77f6d978",
				"statefulset.kubernetes.io/pod-name": "test-0",
			},
			Name:            "test-0",
			Namespace:       "test",
			OwnerReferences: []metav1.OwnerReference{{APIVersion: "apps/v1", BlockOwnerDeletion: ptr.To(true), Controller: ptr.To(true), Kind: "StatefulSet", Name: "test", UID: "ssuid"}},
			ResourceVersion: "345",
			UID:             "poduid",
		},

		Spec: corev1.PodSpec{
			Containers:               []corev1.Container{{Image: "test", Name: "test"}},
			NodeName:                 "mynode",
			Hostname:                 "test-0",
			PreemptionPolicy:         ptr.To(corev1.PreemptionPolicy("PreemptLowerPriority")),
			Priority:                 ptr.To(int32(0)),
			RestartPolicy:            "Always",
			SchedulerName:            "default-scheduler",
			DeprecatedServiceAccount: "default",
			ServiceAccountName:       "default",
			Volumes:                  []corev1.Volume{{Name: "test", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "test-test-0"}}}},
		},

		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{{
				Name:    "test",
				State:   corev1.ContainerState{Running: &corev1.ContainerStateRunning{StartedAt: metav1.Time{Time: time.Now().Add(-time.Hour)}}},
				Ready:   true,
				Started: ptr.To(true),
			}},
			ObservedGeneration: 1,
			Conditions: []corev1.PodCondition{{
				Type:               corev1.PodReady,
				ObservedGeneration: 1,
				Status:             corev1.ConditionTrue,
				LastProbeTime:      metav1.Time{Time: time.Now().Add(-time.Hour)},
			}},
		},
	}
	// c77f6d978 and the raw data was computed from the hash of the defaulted serialized statefulset spec in 1.33
	revision := &appsv1.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			CreationTimestamp: metav1.Time{Time: time.Now().Add(-time.Hour)},
			Labels:            map[string]string{"app": "foo", "controller.kubernetes.io/hash": "c77f6d978"},
			Name:              "test-c77f6d978",
			Namespace:         "test",
			OwnerReferences:   []metav1.OwnerReference{{APIVersion: "apps/v1", BlockOwnerDeletion: ptr.To(true), Controller: ptr.To(true), Kind: "StatefulSet", Name: "test", UID: "ssuid"}},
			ResourceVersion:   "234",
			UID:               "revuid",
		},
		Revision: 1,
		Data:     runtime.RawExtension{Raw: []byte(`{"spec":{"template":{"$patch":"replace","metadata":{"annotations":{"test":"value"},"creationTimestamp":null,"labels":{"app":"foo"}},"spec":{"containers":[{"image":"test","imagePullPolicy":"Always","name":"test","resources":{},"terminationMessagePath":"/dev/termination-log","terminationMessagePolicy":"File"}],"dnsPolicy":"ClusterFirst","restartPolicy":"Always","schedulerName":"default-scheduler","securityContext":{},"terminationGracePeriodSeconds":30}}}}`)},
	}
	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				"key": "value",
				"volume.beta.kubernetes.io/storage-provisioner": "kubernetes.io/host-path",
				"volume.kubernetes.io/storage-provisioner":      "kubernetes.io/host-path",
			},
			CreationTimestamp: metav1.Time{Time: time.Now().Add(-time.Hour)},
			Finalizers:        []string{"kubernetes.io/pvc-protection"},
			Labels:            map[string]string{"app": "foo", "key": "value"},
			Name:              "test-test-0",
			Namespace:         "test",
			ResourceVersion:   "2212",
			UID:               "pvcuid",
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes:      []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			Resources:        corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{"storage": resource.MustParse("1Gi")}},
			StorageClassName: ptr.To("standard"),
			VolumeMode:       ptr.To(corev1.PersistentVolumeFilesystem),
			VolumeName:       "pv",
		},
		Status: corev1.PersistentVolumeClaimStatus{Phase: corev1.ClaimBound},
	}

	// Apply current defaults to all objects.
	// This matches behavior when reading existing objects from etcd.
	legacyscheme.Scheme.Default(set)
	legacyscheme.Scheme.Default(pod)
	legacyscheme.Scheme.Default(revision)
	legacyscheme.Scheme.Default(pvc)

	// Set up and populate initial objects
	client := fake.NewSimpleClientset(set.DeepCopy(), pod.DeepCopy(), revision.DeepCopy(), pvc.DeepCopy())
	om, _, ssc := setupController(client)
	om.podsIndexer.Add(pod.DeepCopy())
	om.setsIndexer.Add(set.DeepCopy())
	om.claimsIndexer.Add(pvc.DeepCopy())
	om.revisionsIndexer.Add(revision.DeepCopy())

	// Clear actions before sync
	client.ClearActions()

	// Run a single sync
	_, err := ssc.UpdateStatefulSet(ctx, set, []*corev1.Pod{pod.DeepCopy()})
	if err != nil {
		t.Fatal(err)
	}

	// Assert nothing changed, no new revision was created, and no write requests were made to statefulset or pods
	pods, err := om.podsLister.List(labels.Everything())
	if err != nil {
		t.Error(err)
	}
	if len(pods) != 1 {
		t.Errorf("expected 1 pod, got %d: %v", len(pods), pods)
	} else if !reflect.DeepEqual(pod, pods[0]) {
		t.Errorf("expected pod with no diff, got: %s", cmp.Diff(pod, pods[0]))
	}

	sets, err := om.setsLister.List(labels.Everything())
	if err != nil {
		t.Error(err)
	}
	if len(sets) != 1 {
		t.Errorf("expected 1 set, got %d: %v", len(sets), sets)
	} else if !reflect.DeepEqual(set, sets[0]) {
		t.Errorf("expected set with no diff, got: %s", cmp.Diff(set, sets[0]))
	}

	revisions := om.revisionsIndexer.List()
	if err != nil {
		t.Error(err)
	}
	if len(revisions) != 1 {
		t.Errorf("expected 1 revision, got %d", len(revisions))
	} else if !reflect.DeepEqual(revision, revisions[0]) {
		t.Errorf("expected revision with no diff, got: %s", cmp.Diff(revision, revisions[0]))
	}

	if om.createPodTracker.requests != 0 {
		t.Error("unexpected create pod requests", om.createPodTracker.requests)
	}
	if om.updatePodTracker.requests != 0 {
		t.Error("unexpected update pod requests", om.updatePodTracker.requests)
	}
	if om.deletePodTracker.requests != 0 {
		t.Error("unexpected delete pod requests", om.deletePodTracker.requests)
	}

	for _, a := range client.Actions() {
		verb := a.GetVerb()
		if verb == "create" || verb == "update" || verb == "patch" || verb == "delete" {
			t.Errorf("unexpected verb %s: %#v", verb, a)
		} else {
			t.Logf("%#v", a)
		}
	}
}
