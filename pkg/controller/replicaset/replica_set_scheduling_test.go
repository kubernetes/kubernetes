/*
Copyright The Kubernetes Authors.

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

package replicaset

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

func newWorkload(name, namespace string, minCount int32, owners ...metav1.OwnerReference) *schedulingv1beta1.Workload {
	workload := &schedulingv1beta1.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			UID:             types.UID(name + "-uid"),
			OwnerReferences: append([]metav1.OwnerReference(nil), owners...),
		},
		Spec: schedulingv1beta1.WorkloadSpec{
			PodGroupTemplates: []schedulingv1beta1.PodGroupTemplate{{
				Name: name,
				SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
					Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
				},
			}},
		},
	}
	if owner := metav1.GetControllerOf(workload); owner != nil {
		workload.Spec.ControllerRef = &schedulingv1beta1.TypedLocalObjectReference{
			APIGroup: apps.GroupName,
			Kind:     owner.Kind,
			Name:     owner.Name,
		}
	}
	return workload
}

func newPodGroup(name, namespace string, minCount int32, owners ...metav1.OwnerReference) *schedulingv1beta1.PodGroup {
	return &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			UID:             types.UID(name + "-uid"),
			OwnerReferences: append([]metav1.OwnerReference(nil), owners...),
		},
		Spec: schedulingv1beta1.PodGroupSpec{
			SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
			},
		},
	}
}

func TestFindWorkloadForReplicaSetRequiresMatchingDeploymentOwner(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "new-deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	oldOwner := deploymentOwner
	oldOwner.UID = "old-deployment-uid"
	matching := newWorkload("inference-current", rs.Namespace, 1, deploymentOwner)
	stale := newWorkload("inference-stale", rs.Namespace, 1, oldOwner)
	duplicate := newWorkload("inference-duplicate", rs.Namespace, 1, deploymentOwner)
	orphaned := newWorkload("inference-orphaned", rs.Namespace, 1, deploymentOwner)
	orphaned.OwnerReferences = nil

	for _, tc := range []struct {
		name         string
		workloads    []*schedulingv1beta1.Workload
		wantWorkload *schedulingv1beta1.Workload
		wantError    string
	}{
		{
			name: "no Workloads",
		},
		{
			name:      "mismatched Deployment UID",
			workloads: []*schedulingv1beta1.Workload{stale},
			wantError: "no Workload controlled by Deployment",
		},
		{
			name:      "orphaned Workload",
			workloads: []*schedulingv1beta1.Workload{orphaned},
			wantError: "no Workload controlled by Deployment",
		},
		{
			name:         "same-name Deployment recreated",
			workloads:    []*schedulingv1beta1.Workload{stale, matching},
			wantWorkload: matching,
		},
		{
			name:      "multiple Workloads for matching Deployment UID",
			workloads: []*schedulingv1beta1.Workload{stale, matching, duplicate},
			wantError: "multiple Workloads controlled by Deployment",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
			for _, workload := range tc.workloads {
				if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
					t.Fatalf("adding Workload to informer: %v", err)
				}
			}

			workload, err := rsc.findWorkloadForReplicaSet(rs)
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("findWorkloadForReplicaSet() error = %v, want %q", err, tc.wantError)
				}
			} else if err != nil {
				t.Fatalf("findWorkloadForReplicaSet() error = %v", err)
			}
			if !reflect.DeepEqual(workload, tc.wantWorkload) {
				t.Errorf("findWorkloadForReplicaSet() = %#v, want %#v", workload, tc.wantWorkload)
			}
		})
	}
}

func TestEnqueueWorkloadReplicaSets(t *testing.T) {
	tCtx := ktesting.Init(t)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "new-deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference-workload", rs.Namespace, 1, deploymentOwner)
	rs.OwnerReferences[0].UID = "old-deployment-uid"
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	rsc.enqueueWorkloadReplicaSets(tCtx.Logger(), workload)
	if got := rsc.queue.Len(); got != 0 {
		t.Fatalf("queue length after mismatched Deployment UID = %d, want 0", got)
	}

	matchingRS := rs.DeepCopy()
	matchingRS.Name = "inference-matching"
	matchingRS.OwnerReferences[0].UID = workload.OwnerReferences[0].UID
	otherNamespaceRS := matchingRS.DeepCopy()
	otherNamespaceRS.Namespace = "other"
	for _, replicaSet := range []*apps.ReplicaSet{matchingRS, otherNamespaceRS} {
		if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(replicaSet); err != nil {
			t.Fatalf("adding ReplicaSet to informer: %v", err)
		}
	}
	rsc.enqueueWorkloadReplicaSets(tCtx.Logger(), workload)
	if got := rsc.queue.Len(); got != 1 {
		t.Fatalf("queue length after matching Deployment UID = %d, want 1", got)
	}
	key, _ := rsc.queue.Get()
	defer rsc.queue.Done(key)
	if want := matchingRS.Namespace + "/" + matchingRS.Name; key != want {
		t.Fatalf("queued key = %q, want %q", key, want)
	}
}

func TestWorkloadSpecChanged(t *testing.T) {
	workload := newWorkload("main", metav1.NamespaceDefault, 1)
	metadataUpdate := workload.DeepCopy()
	metadataUpdate.Labels = map[string]string{"updated": "true"}
	specUpdate := workload.DeepCopy()
	specUpdate.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = 2

	testCases := []struct {
		name string
		old  interface{}
		new  interface{}
		want bool
	}{
		{name: "metadata-only update", old: workload, new: metadataUpdate, want: false},
		{name: "spec update", old: workload, new: specUpdate, want: true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := workloadSpecChanged(tc.old, tc.new); got != tc.want {
				t.Fatalf("workloadSpecChanged() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestEnsurePodGroupForReplicaSetRejectsUnownedPodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference-workload", rs.Namespace, 1, deploymentOwner)
	podGroup := newPodGroup(rs.Name, rs.Namespace, 1)

	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}
	if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(podGroup); err != nil {
		t.Fatalf("adding PodGroup to informer: %v", err)
	}

	_, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
	if err == nil || !strings.Contains(err.Error(), "not controlled by ReplicaSet") {
		t.Fatalf("ensurePodGroupForReplicaSet() error = %v, want unowned PodGroup rejection", err)
	}
}

func TestEnsurePodGroupForReplicaSet(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	client := fake.NewSimpleClientset()
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(2, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference", rs.Namespace, 3, deploymentOwner)
	podGroupName := rs.Name
	if _, err := client.AppsV1().ReplicaSets(rs.Namespace).Create(ctx, rs, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating ReplicaSet: %v", err)
	}

	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}
	if err := rsc.ensureReplicaSetPodGroupFinalizer(ctx, rs); err != nil {
		t.Fatalf("ensureReplicaSetPodGroupFinalizer() error = %v", err)
	}
	updatedRS, err := client.AppsV1().ReplicaSets(rs.Namespace).Get(ctx, rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting ReplicaSet after adding finalizer: %v", err)
	}
	if !hasReplicaSetPodGroupFinalizer(updatedRS) {
		t.Fatalf("ReplicaSet finalizers = %v, want PodGroup cleanup finalizer", updatedRS.Finalizers)
	}
	ensuredPodGroup, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
	if err != nil {
		t.Fatalf("ensurePodGroupForReplicaSet() error = %v", err)
	}
	if ensuredPodGroup == nil {
		t.Fatal("ensurePodGroupForReplicaSet() returned a nil PodGroup")
	}

	created, err := client.SchedulingV1beta1().PodGroups("default").Get(ctx, podGroupName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting created PodGroup: %v", err)
	}
	owner := metav1.GetControllerOf(created)
	if owner == nil || owner.Kind != "ReplicaSet" || owner.Name != rs.Name || owner.UID != rs.UID {
		t.Fatalf("PodGroup owner = %#v, want ReplicaSet %s/%s", owner, rs.Namespace, rs.Name)
	}
	if created.Spec.WorkloadRef == nil || created.Spec.WorkloadRef.WorkloadName != workload.Name {
		t.Fatalf("PodGroup workloadRef = %#v, want Workload %q", created.Spec.WorkloadRef, workload.Name)
	}
	if created.Spec.SchedulingPolicy.Gang == nil || created.Spec.SchedulingPolicy.Gang.MinCount != 3 {
		t.Fatalf("PodGroup gang = %#v, want minCount 3", created.Spec.SchedulingPolicy.Gang)
	}

	t.Run("PodGroup not yet in cache", func(t *testing.T) {
		podGroup, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
		if !apierrors.IsAlreadyExists(err) {
			t.Fatalf("ensurePodGroupForReplicaSet() error = %v, want AlreadyExists", err)
		}
		if podGroup != nil {
			t.Fatalf("ensurePodGroupForReplicaSet() returned PodGroup %s on create failure", podGroup.Name)
		}

		if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(created); err != nil {
			t.Fatalf("adding created PodGroup to informer: %v", err)
		}
		podGroup, err = rsc.ensurePodGroupForReplicaSet(ctx, rs)
		if err != nil {
			t.Fatalf("ensurePodGroupForReplicaSet() after cache update: %v", err)
		}
		if podGroup == nil || podGroup.Name != created.Name {
			t.Fatalf("PodGroup after cache update = %#v, want PodGroup %q", podGroup, created.Name)
		}
		podGroups, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			t.Fatalf("listing PodGroups after retry: %v", err)
		}
		if len(podGroups.Items) != 1 {
			t.Fatalf("PodGroup count after retry = %d, want 1", len(podGroups.Items))
		}
	})

	t.Run("existing PodGroup with different minCount gets patched", func(t *testing.T) {
		existing := created.DeepCopy()
		existing.Spec.SchedulingPolicy.Gang.MinCount = 1
		if _, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Update(ctx, existing, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("setting existing PodGroup minCount: %v", err)
		}
		if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(existing); err != nil {
			t.Fatalf("adding existing PodGroup to informer: %v", err)
		}

		if _, err := rsc.ensurePodGroupForReplicaSet(ctx, rs); err != nil {
			t.Fatalf("ensurePodGroupForReplicaSet() error = %v", err)
		}
		updated, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, podGroupName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("getting updated PodGroup: %v", err)
		}
		if gang := updated.Spec.SchedulingPolicy.Gang; gang == nil || gang.MinCount != 3 {
			t.Fatalf("PodGroup gang = %#v, want minCount 3", gang)
		}
	})

	t.Run("terminating PodGroup", func(t *testing.T) {
		terminating := created.DeepCopy()
		terminating.DeletionTimestamp = new(metav1.NewTime(time.Now()))
		terminating.Spec.SchedulingPolicy.Gang.MinCount = 1
		if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(terminating); err != nil {
			t.Fatalf("adding terminating PodGroup to informer: %v", err)
		}
		client.ClearActions()

		podGroup, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
		if err == nil || !strings.Contains(err.Error(), "is being deleted") {
			t.Fatalf("ensurePodGroupForReplicaSet() error = %v, want terminating PodGroup rejection", err)
		}
		if podGroup != nil {
			t.Fatalf("ensurePodGroupForReplicaSet() returned terminating PodGroup %s", podGroup.Name)
		}
		if actions := client.Actions(); len(actions) != 0 {
			t.Fatalf("client actions = %#v, want no API calls for a terminating PodGroup", actions)
		}
	})
}

func TestEnsurePodGroupForReplicaSetDeletesAtZero(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference", rs.Namespace, 1, deploymentOwner)
	foreignOwner := replicaSetPodGroupOwner(rs)
	foreignOwner.UID = "foreign-rs-uid"

	for _, tc := range []struct {
		name            string
		ownerReferences []metav1.OwnerReference
		wantDeleted     bool
	}{
		{
			name:            "owned",
			ownerReferences: []metav1.OwnerReference{replicaSetPodGroupOwner(rs)},
			wantDeleted:     true,
		},
		{
			name:            "foreign",
			ownerReferences: []metav1.OwnerReference{foreignOwner},
		},
		{
			name: "ownerless",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			podGroup := newPodGroup(rs.Name, rs.Namespace, 1, tc.ownerReferences...)
			client := fake.NewSimpleClientset(podGroup)
			rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
			if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
				t.Fatalf("adding Workload to informer: %v", err)
			}
			if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(podGroup); err != nil {
				t.Fatalf("adding PodGroup to informer: %v", err)
			}

			ensuredPodGroup, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
			if err != nil {
				t.Fatalf("ensurePodGroupForReplicaSet() error = %v", err)
			}
			if ensuredPodGroup != nil {
				t.Fatalf("ensurePodGroupForReplicaSet() returned PodGroup %s at zero replicas", ensuredPodGroup.Name)
			}
			actions := client.Actions()
			wantActions := 0
			if tc.wantDeleted {
				wantActions = 1
			}
			if len(actions) != wantActions {
				t.Fatalf("client actions = %#v, want %d actions", actions, wantActions)
			}
			if tc.wantDeleted {
				action, ok := actions[0].(clienttesting.DeleteAction)
				if !ok {
					t.Fatalf("client action = %#v, want a DELETE", actions[0])
				}
				options := action.GetDeleteOptions()
				if options.Preconditions == nil || options.Preconditions.UID == nil || *options.Preconditions.UID != podGroup.UID {
					t.Fatalf("PodGroup delete preconditions = %#v, want UID %q", options.Preconditions, podGroup.UID)
				}
			}
			_, err = client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, podGroup.Name, metav1.GetOptions{})
			if tc.wantDeleted {
				if !apierrors.IsNotFound(err) {
					t.Fatalf("PodGroup GET error = %v, want NotFound", err)
				}
			} else if err != nil {
				t.Fatalf("getting preserved PodGroup: %v", err)
			}
		})
	}
}

func TestEnsurePodGroupForReplicaSetRecreatesOnScaleUp(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	tCtx := ktesting.Init(t)
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference", rs.Namespace, 3, deploymentOwner)
	oldPodGroup := newPodGroup(rs.Name, rs.Namespace, 3, replicaSetPodGroupOwner(rs), workloadPodGroupOwner(workload))
	client := fake.NewSimpleClientset(workload, oldPodGroup)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}
	podGroupIndexer := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer()
	if err := podGroupIndexer.Add(oldPodGroup); err != nil {
		t.Fatalf("adding PodGroup to informer: %v", err)
	}

	if podGroup, err := rsc.ensurePodGroupForReplicaSet(tCtx, rs); err != nil || podGroup != nil {
		t.Fatalf("ensurePodGroupForReplicaSet() at zero replicas = (%#v, %v), want (nil, nil)", podGroup, err)
	}
	if _, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, rs.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("PodGroup GET error at zero replicas = %v, want NotFound", err)
	}
	// Simulate the informer observing deletion before scaling back up.
	if err := podGroupIndexer.Delete(oldPodGroup); err != nil {
		t.Fatalf("removing deleted PodGroup from informer: %v", err)
	}

	rs.Spec.Replicas = new(int32(3))
	recreated, err := rsc.ensurePodGroupForReplicaSet(tCtx, rs)
	if err != nil {
		t.Fatalf("ensurePodGroupForReplicaSet() after scale-up: %v", err)
	}
	if recreated == nil {
		t.Fatal("ensurePodGroupForReplicaSet() returned no PodGroup after scale-up")
	}
	stored, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting recreated PodGroup: %v", err)
	}
	if !metav1.IsControlledBy(stored, rs) {
		t.Fatalf("recreated PodGroup ownerReferences = %#v, want ReplicaSet %s/%s with UID %q", stored.OwnerReferences, rs.Namespace, rs.Name, rs.UID)
	}
}

func TestSyncReplicaSetWaitsForMatchingWorkload(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "new-deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	oldOwner := deploymentOwner
	oldOwner.UID = "old-deployment-uid"
	stale := newWorkload("inference-stale", rs.Namespace, 1, oldOwner)
	matching := newWorkload("inference-current", rs.Namespace, 1, deploymentOwner)
	client := fake.NewSimpleClientset(rs, stale, matching)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	fakePodControl := &controller.FakePodControl{}
	rsc.podControl = fakePodControl
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	// Populate the informer cache with only the old Deployment's Workload
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(stale); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}

	if err := rsc.syncReplicaSet(context.Background(), rs.Namespace+"/"+rs.Name); err == nil {
		t.Fatal("syncReplicaSet() succeeded without a matching Workload in the cache")
	}
	if len(fakePodControl.Templates) != 0 {
		t.Fatalf("created Pod templates = %d, want 0 while the matching Workload is missing from the cache", len(fakePodControl.Templates))
	}
}

func TestSyncReplicaSetCreatesPodGroupBeforePods(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = []metav1.OwnerReference{deploymentOwner}
	workload := newWorkload("inference", rs.Namespace, 1, deploymentOwner)
	podGroupName := rs.Name

	const podGroupUID = "podgroup-uid"
	client := fake.NewSimpleClientset(rs, workload)
	client.PrependReactor("create", "podgroups", func(action clienttesting.Action) (bool, runtime.Object, error) {
		podGroup := action.(clienttesting.CreateAction).GetObject().(*schedulingv1beta1.PodGroup)
		podGroup.UID = podGroupUID
		return false, nil, nil
	})
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	fakePodControl := &controller.FakePodControl{}
	rsc.podControl = fakePodControl
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}

	if err := rsc.syncReplicaSet(ctx, "default/inference-abc123"); err != nil {
		t.Fatalf("syncReplicaSet() error = %v", err)
	}
	if len(fakePodControl.Templates) != 1 {
		t.Fatalf("created Pod templates = %d, want 1", len(fakePodControl.Templates))
	}
	createdTemplate := fakePodControl.Templates[0]
	if group := createdTemplate.Spec.SchedulingGroup; group == nil ||
		group.PodGroupName == nil || *group.PodGroupName != podGroupName {
		t.Fatalf("Pod schedulingGroup = %#v, want PodGroup %q",
			createdTemplate.Spec.SchedulingGroup, podGroupName)
	}
	var podGroupOwner *metav1.OwnerReference
	for i := range createdTemplate.OwnerReferences {
		owner := &createdTemplate.OwnerReferences[i]
		if owner.APIVersion == schedulingv1beta1.SchemeGroupVersion.String() &&
			owner.Kind == "PodGroup" && owner.Name == podGroupName {
			podGroupOwner = owner
			break
		}
	}
	if podGroupOwner == nil {
		t.Fatalf("Pod ownerReferences = %#v, want PodGroup %q",
			createdTemplate.OwnerReferences, podGroupName)
	}
	createdPodGroup, err := client.SchedulingV1beta1().PodGroups("default").Get(ctx, podGroupName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("PodGroup was not created before pod reconciliation: %v", err)
	}
	if podGroupOwner.Name != createdPodGroup.Name || podGroupOwner.UID != podGroupUID {
		t.Fatalf("PodGroup ownerRef = %#v, want name %q and UID %q",
			podGroupOwner, createdPodGroup.Name, podGroupUID)
	}
}

func TestCleanupDeletingReplicaSetRemovesFinalizerWhenFeatureDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.WorkloadWithDeployment: false,
	})

	tCtx := ktesting.Init(t)
	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = nil
	rs.DeletionTimestamp = new(metav1.NewTime(time.Now()))
	rs.Finalizers = []string{replicaSetPodGroupFinalizer}
	podGroup := newPodGroup(rs.Name, rs.Namespace, 1, replicaSetPodGroupOwner(rs))
	client := fake.NewSimpleClientset(rs, podGroup)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}

	if err := rsc.syncReplicaSet(tCtx, "default/"+rs.Name); err != nil {
		t.Fatalf("syncReplicaSet() error = %v", err)
	}
	podGroupGets := 0
	for _, action := range client.Actions() {
		if action.GetVerb() == "get" && action.GetResource().Resource == "podgroups" {
			podGroupGets++
		}
		if action.GetVerb() == "delete" && action.GetResource().Resource == "podgroups" {
			options := action.(clienttesting.DeleteAction).GetDeleteOptions()
			if options.Preconditions == nil || options.Preconditions.UID == nil || *options.Preconditions.UID != podGroup.UID {
				t.Fatalf("PodGroup delete preconditions = %#v, want UID %q", options.Preconditions, podGroup.UID)
			}
		}
	}
	if podGroupGets != 1 {
		t.Fatalf("PodGroup GET count = %d, want 1 with the feature disabled", podGroupGets)
	}
	if _, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, rs.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("PodGroup GET error = %v, want NotFound", err)
	}
	updatedRS, err := client.AppsV1().ReplicaSets(rs.Namespace).Get(tCtx, rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting ReplicaSet after cleanup: %v", err)
	}
	if hasReplicaSetPodGroupFinalizer(updatedRS) {
		t.Fatalf("ReplicaSet finalizers = %v, want cleanup finalizer removed", updatedRS.Finalizers)
	}
}

func TestPatchPodGroupMinCount(t *testing.T) {
	tCtx := ktesting.Init(t)
	podGroup := newPodGroup("inference-abc123", metav1.NamespaceDefault, 1)
	original := podGroup.DeepCopy()
	client := fake.NewSimpleClientset(podGroup)
	rsc := &ReplicaSetController{kubeClient: client}

	if err := rsc.patchPodGroupMinCount(tCtx, podGroup, 2); err != nil {
		t.Fatalf("patchPodGroupMinCount() error = %v", err)
	}
	actions := client.Actions()
	if len(actions) != 1 {
		t.Fatalf("client action count = %d, want 1", len(actions))
	}
	action, ok := actions[0].(clienttesting.PatchAction)
	if !ok || action.GetPatchType() != types.StrategicMergePatchType {
		t.Fatalf("client action = %#v, want a strategic merge patch", actions[0])
	}
	var patch map[string]interface{}
	if err := json.Unmarshal(action.GetPatch(), &patch); err != nil {
		t.Fatalf("decoding PodGroup patch: %v", err)
	}
	want := map[string]interface{}{
		"metadata": map[string]interface{}{
			"uid": string(podGroup.UID),
		},
		"spec": map[string]interface{}{
			"schedulingPolicy": map[string]interface{}{
				"gang": map[string]interface{}{"minCount": float64(2)},
			},
		},
	}
	if !reflect.DeepEqual(patch, want) {
		t.Fatalf("PodGroup patch = %#v, want %#v", patch, want)
	}
	updated, err := client.SchedulingV1beta1().PodGroups(podGroup.Namespace).Get(tCtx, podGroup.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting updated PodGroup: %v", err)
	}
	if gang := updated.Spec.SchedulingPolicy.Gang; gang == nil || gang.MinCount != 2 {
		t.Fatalf("PodGroup gang = %#v, want minCount 2", gang)
	}
	if !reflect.DeepEqual(podGroup, original) {
		t.Fatal("patchPodGroupMinCount modified the cached PodGroup")
	}
}

func TestCleanupReplicaSetPodGroupHandlesCacheMiss(t *testing.T) {
	tCtx := ktesting.Init(t)
	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.OwnerReferences = nil
	podGroup := newPodGroup(rs.Name, rs.Namespace, 1, replicaSetPodGroupOwner(rs))
	client := fake.NewSimpleClientset(podGroup)
	sharedInformers := informers.NewSharedInformerFactory(client, 0)
	rsc := &ReplicaSetController{
		kubeClient:     client,
		podGroupLister: sharedInformers.Scheduling().V1beta1().PodGroups().Lister(),
	}

	if err := rsc.cleanupReplicaSetPodGroup(tCtx, rs); err != nil {
		t.Fatalf("cleanupReplicaSetPodGroup() error = %v", err)
	}
	if _, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, podGroup.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("PodGroup GET error = %v, want NotFound after cleanup", err)
	}
}

func TestNewReplicaSetControllerSchedulingDisabledWhenAPINotServed(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	tCtx := ktesting.Init(t)
	client := fake.NewSimpleClientset()
	sharedInformers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	rsc, err := NewReplicaSetController(
		tCtx,
		sharedInformers.Apps().V1().ReplicaSets(),
		sharedInformers.Core().V1().Pods(),
		nil,
		nil,
		client,
		BurstReplicas,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rsc.schedulingEnabled {
		t.Error("schedulingEnabled = true, want false when the scheduling informers are nil")
	}
}
