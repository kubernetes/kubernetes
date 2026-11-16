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
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/ktesting"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
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

func TestEnsurePodGroupForReplicaSet(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})
	ctx := context.Background()

	rs := newReplicaSet(2, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 3)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	pgName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)

	ownedPodGroup := func(minCount int32) *schedulingv1beta1.PodGroup {
		pg := newPodGroup(pgName, rs.Namespace, minCount, *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet")))
		pg.Spec.WorkloadRef = &schedulingv1beta1.WorkloadReference{WorkloadName: workload.Name, TemplateName: templateName}
		return pg
	}
	terminatingPodGroup := func() *schedulingv1beta1.PodGroup {
		pg := ownedPodGroup(3)
		pg.DeletionTimestamp = new(metav1.NewTime(time.Now()))
		return pg
	}
	mismatchedPodGroup := func() *schedulingv1beta1.PodGroup {
		pg := newPodGroup(pgName, rs.Namespace, 3, *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet")))
		pg.Spec.WorkloadRef = &schedulingv1beta1.WorkloadReference{WorkloadName: "other", TemplateName: "other"}
		return pg
	}

	for _, tc := range []struct {
		name             string
		podGroup         *schedulingv1beta1.PodGroup
		inCache          bool
		inAPI            bool
		wantPodGroup     bool
		wantGangMinCount int32
		wantErrContains  string
	}{
		{name: "creates the PodGroup when none exists", wantPodGroup: true, wantGangMinCount: 3},
		{name: "returns the cached PodGroup and patches minCount to the template", podGroup: ownedPodGroup(1), inCache: true, inAPI: true, wantPodGroup: true, wantGangMinCount: 3},
		{name: "surfaces AlreadyExists when the PodGroup is in the API but not the cache", podGroup: ownedPodGroup(3), inAPI: true, wantErrContains: "already exists"},
		{name: "rejects a terminating PodGroup", podGroup: terminatingPodGroup(), inCache: true, wantErrContains: "is being deleted"},
		{name: "rejects a PodGroup that references a different workload", podGroup: mismatchedPodGroup(), inCache: true, wantErrContains: "does not reference workload"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var objs []runtime.Object
			if tc.podGroup != nil && tc.inAPI {
				objs = append(objs, tc.podGroup)
			}
			client := fake.NewSimpleClientset(objs...)
			rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
			if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
				t.Fatalf("adding Workload to informer: %v", err)
			}
			if tc.podGroup != nil && tc.inCache {
				if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(tc.podGroup); err != nil {
					t.Fatalf("adding PodGroup to informer: %v", err)
				}
			}

			got, err := rsc.ensurePodGroupForReplicaSet(ctx, rs)
			if tc.wantErrContains != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Fatalf("ensurePodGroupForReplicaSet() error = %v, want error containing %q", err, tc.wantErrContains)
				}
				if got != nil {
					t.Fatalf("ensurePodGroupForReplicaSet() = PodGroup %q, want nil on error", got.Name)
				}
				return
			}
			if err != nil {
				t.Fatalf("ensurePodGroupForReplicaSet() returned unexpected error: %v", err)
			}
			if got == nil {
				t.Fatal("ensurePodGroupForReplicaSet() = nil, want a PodGroup")
			}
			stored, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, pgName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("getting PodGroup %q after ensure: %v", pgName, err)
			}
			if !metav1.IsControlledBy(stored, rs) || len(stored.OwnerReferences) != 1 {
				t.Fatalf("PodGroup ownerReferences = %#v, want a single controller ref to ReplicaSet %q", stored.OwnerReferences, rs.Name)
			}
			if stored.Spec.WorkloadRef == nil || stored.Spec.WorkloadRef.WorkloadName != workload.Name {
				t.Fatalf("PodGroup WorkloadRef = %#v, want it to reference Workload %q", stored.Spec.WorkloadRef, workload.Name)
			}
			if stored.Spec.SchedulingPolicy.Gang == nil || stored.Spec.SchedulingPolicy.Gang.MinCount != tc.wantGangMinCount {
				t.Fatalf("PodGroup gang policy = %#v, want minCount %d from the Workload template", stored.Spec.SchedulingPolicy.Gang, tc.wantGangMinCount)
			}
		})
	}
}

func TestEnsurePodGroupForReplicaSetUsesReplicaSetPriorityClassName(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	client := fake.NewSimpleClientset()
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	rs := newReplicaSet(2, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rs.Spec.Template.Spec.PriorityClassName = "high-priority"
	workload := newWorkload("inference", rs.Namespace, 3)
	workload.Spec.PodGroupTemplates[0].PriorityClassName = "low-priority"
	workload.Spec.PodGroupTemplates[0].Priority = new(int32(100))
	workload.Spec.PodGroupTemplates[0].PreemptionPolicy = new(schedulingv1beta1.PreemptNever)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	podGroupName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)

	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}

	if _, err := rsc.ensurePodGroupForReplicaSet(ctx, rs); err != nil {
		t.Fatalf("ensurePodGroupForReplicaSet() returned error while ensuring the PodGroup: %v", err)
	}
	created, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, podGroupName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting PodGroup %q that ensurePodGroupForReplicaSet should have created: %v", podGroupName, err)
	}
	if got := created.Spec.PriorityClassName; got != "high-priority" {
		t.Fatalf("PodGroup priorityClassName = %q, want %q from the ReplicaSet pod template", got, "high-priority")
	}
	if created.Spec.Priority != nil {
		t.Errorf("PodGroup priority = %d, want nil for admission to resolve", *created.Spec.Priority)
	}
	if created.Spec.PreemptionPolicy != nil {
		t.Errorf("PodGroup preemptionPolicy = %q, want nil for admission to resolve", *created.Spec.PreemptionPolicy)
	}
}

func TestEnsurePodGroupForReplicaSetDeletesAtZero(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 1)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	pgName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)
	rsOwner := *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet"))
	foreignOwner := rsOwner
	foreignOwner.UID = "foreign-rs-uid"

	for _, tc := range []struct {
		name            string
		ownerReferences []metav1.OwnerReference
		wantDeleted     bool
	}{
		{name: "owned by the ReplicaSet", ownerReferences: []metav1.OwnerReference{rsOwner}, wantDeleted: true},
		{name: "owned by a different ReplicaSet", ownerReferences: []metav1.OwnerReference{foreignOwner}},
		{name: "ownerless", ownerReferences: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			podGroup := newPodGroup(pgName, rs.Namespace, 1, tc.ownerReferences...)
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
				t.Fatalf("ensurePodGroupForReplicaSet() returned error at zero replicas: %v", err)
			}
			if ensuredPodGroup != nil {
				t.Fatalf("ensurePodGroupForReplicaSet() = PodGroup %q at zero replicas, want nil", ensuredPodGroup.Name)
			}
			actions := client.Actions()
			wantActions := 0
			if tc.wantDeleted {
				wantActions = 1
			}
			if len(actions) != wantActions {
				t.Fatalf("controller made %d API calls (%#v), want %d", len(actions), actions, wantActions)
			}
			if tc.wantDeleted {
				action, ok := actions[0].(clienttesting.DeleteAction)
				if !ok {
					t.Fatalf("controller API call = %#v, want a PodGroup DELETE", actions[0])
				}
				options := action.GetDeleteOptions()
				if options.Preconditions == nil || options.Preconditions.UID == nil || *options.Preconditions.UID != podGroup.UID {
					t.Fatalf("PodGroup DELETE preconditions = %#v, want a UID precondition of %q", options.Preconditions, podGroup.UID)
				}
			}
			_, err = client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, podGroup.Name, metav1.GetOptions{})
			if tc.wantDeleted {
				if !apierrors.IsNotFound(err) {
					t.Fatalf("PodGroup still present after delete: GET error = %v, want NotFound", err)
				}
			} else if err != nil {
				t.Fatalf("getting the PodGroup that should have been preserved: %v", err)
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
	rs := newReplicaSet(0, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 3)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	pgName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)
	oldPodGroup := newPodGroup(pgName, rs.Namespace, 3, *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet")))
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
	if _, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, pgName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("old PodGroup still present at zero replicas: GET error = %v, want NotFound", err)
	}
	// Simulate the informer observing deletion before scaling back up.
	if err := podGroupIndexer.Delete(oldPodGroup); err != nil {
		t.Fatalf("removing deleted PodGroup from informer: %v", err)
	}

	rs.Spec.Replicas = new(int32(3))
	recreated, err := rsc.ensurePodGroupForReplicaSet(tCtx, rs)
	if err != nil {
		t.Fatalf("ensurePodGroupForReplicaSet() returned error after scaling back up: %v", err)
	}
	if recreated == nil {
		t.Fatal("ensurePodGroupForReplicaSet() = nil after scale-up, want a recreated PodGroup")
	}
	stored, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, pgName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting the recreated PodGroup: %v", err)
	}
	if !metav1.IsControlledBy(stored, rs) {
		t.Fatalf("recreated PodGroup ownerReferences = %#v, want a controller ref to ReplicaSet %q", stored.OwnerReferences, rs.Name)
	}
}

func TestSyncReplicaSetWaitsForMatchingWorkload(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 1)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	// Only an unrelated Workload is cached, so lookup by the annotated name still misses.
	otherWorkload := newWorkload("other-workload", rs.Namespace, 1)
	client := fake.NewSimpleClientset(rs)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	fakePodControl := &controller.FakePodControl{}
	rsc.podControl = fakePodControl
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(otherWorkload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}

	if err := rsc.syncReplicaSet(context.Background(), rs.Namespace+"/"+rs.Name); err == nil {
		t.Fatal("syncReplicaSet() = nil error, want an error while the named Workload is missing from the cache")
	}
	if len(fakePodControl.Templates) != 0 {
		t.Fatalf("syncReplicaSet() created %d Pods, want 0 while the named Workload is missing from the cache", len(fakePodControl.Templates))
	}
}

func TestSyncReplicaSetCreatesPodGroupBeforePods(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 1)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	podGroupName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)

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
	if _, err := client.SchedulingV1beta1().PodGroups("default").Get(ctx, podGroupName, metav1.GetOptions{}); err != nil {
		t.Fatalf("PodGroup was not created before pod reconciliation: %v", err)
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

func TestDeletePodGroupForReplicaSet(t *testing.T) {
	for _, tc := range []struct {
		name        string
		inCache     bool
		wantDeleted bool
	}{
		{name: "cached PodGroup is deleted", inCache: true, wantDeleted: true},
		{name: "cache miss is a no-op", inCache: false, wantDeleted: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			rs := newReplicaSet(0, map[string]string{"name": "foo"})
			rs.Name = "inference-abc123"
			pgName := "inference-abc123-pg"
			podGroup := newPodGroup(pgName, rs.Namespace, 1, *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet")))
			client := fake.NewSimpleClientset(podGroup)
			sharedInformers := informers.NewSharedInformerFactory(client, 0)
			if tc.inCache {
				if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(podGroup); err != nil {
					t.Fatal(err)
				}
			}
			rsc := &ReplicaSetController{
				kubeClient:     client,
				podGroupLister: sharedInformers.Scheduling().V1beta1().PodGroups().Lister(),
			}

			if err := rsc.deletePodGroupForReplicaSet(tCtx, rs, pgName); err != nil {
				t.Fatalf("deletePodGroupForReplicaSet() returned error: %v", err)
			}
			_, err := client.SchedulingV1beta1().PodGroups(rs.Namespace).Get(tCtx, pgName, metav1.GetOptions{})
			if tc.wantDeleted && !apierrors.IsNotFound(err) {
				t.Fatalf("PodGroup still present after delete: GET error = %v, want NotFound", err)
			}
			if !tc.wantDeleted && err != nil {
				t.Fatalf("PodGroup unexpectedly affected on cache miss: GET error = %v, want nil", err)
			}
		})
	}
}

func TestShouldManagePodGroupForReplicaSet(t *testing.T) {
	delegated := map[string]string{
		apischeduling.WorkloadNameAnnotation:      "inference",
		apischeduling.GroupTemplateNameAnnotation: "inference",
	}
	for _, tc := range []struct {
		name                     string
		workloadWithDeployment   bool
		enableWorkloadScheduling bool
		annotations              map[string]string
		schedulingGroup          *v1.PodSchedulingGroup
		want                     bool
	}{
		{name: "delegated and enabled", workloadWithDeployment: true, enableWorkloadScheduling: true, annotations: delegated, want: true},
		{name: "workload-name annotation only", workloadWithDeployment: true, enableWorkloadScheduling: true, annotations: map[string]string{apischeduling.WorkloadNameAnnotation: "inference"}, want: true},
		{name: "group-template-name annotation only", workloadWithDeployment: true, enableWorkloadScheduling: true, annotations: map[string]string{apischeduling.GroupTemplateNameAnnotation: "inference"}, want: true},
		{name: "no delegation annotations", workloadWithDeployment: true, enableWorkloadScheduling: true, annotations: nil, want: false},
		{name: "pod template already declares a schedulingGroup", workloadWithDeployment: true, enableWorkloadScheduling: true, annotations: delegated, schedulingGroup: &v1.PodSchedulingGroup{}, want: false},
		{name: "feature gate disabled", workloadWithDeployment: false, enableWorkloadScheduling: true, annotations: delegated, want: false},
		{name: "workload scheduling disabled for the controller", workloadWithDeployment: true, enableWorkloadScheduling: false, annotations: delegated, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:        tc.workloadWithDeployment,
				features.WorkloadWithDeployment: tc.workloadWithDeployment,
			})
			rsc, _ := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
			rsc.controllerFeatures.EnableWorkloadScheduling = tc.enableWorkloadScheduling
			rs := newReplicaSet(1, map[string]string{"name": "foo"})
			rs.Annotations = tc.annotations
			rs.Spec.Template.Spec.SchedulingGroup = tc.schedulingGroup
			if got := rsc.shouldManagePodGroupForReplicaSet(rs); got != tc.want {
				t.Fatalf("shouldManagePodGroupForReplicaSet() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestEnqueueReplicaSetForPodGroup(t *testing.T) {
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	rsOwner := *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet"))
	deploymentOwner := *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("Deployment"))
	staleOwner := rsOwner
	staleOwner.UID = "stale-uid"

	for _, tc := range []struct {
		name    string
		obj     interface{}
		wantLen int
	}{
		{name: "controlled by the ReplicaSet", obj: newPodGroup("pg", rs.Namespace, 1, rsOwner), wantLen: 1},
		{name: "controlled by a different kind", obj: newPodGroup("pg", rs.Namespace, 1, deploymentOwner), wantLen: 0},
		{name: "controller UID does not match", obj: newPodGroup("pg", rs.Namespace, 1, staleOwner), wantLen: 0},
		{name: "tombstone of a controlled PodGroup", obj: cache.DeletedFinalStateUnknown{Key: rs.Namespace + "/pg", Obj: newPodGroup("pg", rs.Namespace, 1, rsOwner)}, wantLen: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
			if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
				t.Fatalf("adding ReplicaSet to informer: %v", err)
			}

			rsc.enqueueReplicaSetForPodGroup(tc.obj)

			if got := rsc.queue.Len(); got != tc.wantLen {
				t.Fatalf("queue length = %d, want %d", got, tc.wantLen)
			}
			if tc.wantLen > 0 {
				key, _ := rsc.queue.Get()
				if want := rs.Namespace + "/" + rs.Name; key != want {
					t.Fatalf("enqueued key = %q, want %q", key, want)
				}
			}
		})
	}
}

func TestEnqueueWorkloadReplicaSets(t *testing.T) {
	deploymentOwner := metav1.OwnerReference{
		APIVersion: apps.SchemeGroupVersion.String(),
		Kind:       "Deployment",
		Name:       "inference",
		UID:        "deployment-uid",
		Controller: new(true),
	}
	replicaSet := func(name, namespace, workloadName string, owners ...metav1.OwnerReference) *apps.ReplicaSet {
		rs := newReplicaSet(1, map[string]string{"name": "foo"})
		rs.Name = name
		rs.Namespace = namespace
		rs.OwnerReferences = owners
		if workloadName != "" {
			rs.Annotations = map[string]string{
				apischeduling.WorkloadNameAnnotation:      workloadName,
				apischeduling.GroupTemplateNameAnnotation: "workers",
			}
		}
		return rs
	}

	testCases := []struct {
		name           string
		replicaSets    []*apps.ReplicaSet
		workloadOwners []metav1.OwnerReference
		wantKeys       []string
	}{
		{
			name: "standalone ReplicaSet delegating to an unowned Workload is enqueued",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-standalone", metav1.NamespaceDefault, "inference-workload"),
			},
			wantKeys: []string{metav1.NamespaceDefault + "/inference-standalone"},
		},
		{
			name: "Deployment-owned ReplicaSet with matching Workload annotation is enqueued",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-matching", metav1.NamespaceDefault, "inference-workload", deploymentOwner),
			},
			workloadOwners: []metav1.OwnerReference{deploymentOwner},
			wantKeys:       []string{metav1.NamespaceDefault + "/inference-matching"},
		},
		{
			name: "matching Deployment owner with another Workload annotation is ignored",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-unrelated", metav1.NamespaceDefault, "other-workload", deploymentOwner),
			},
			workloadOwners: []metav1.OwnerReference{deploymentOwner},
		},
		{
			name: "matching Deployment owner without delegation annotations is ignored",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-undelegated", metav1.NamespaceDefault, "", deploymentOwner),
			},
			workloadOwners: []metav1.OwnerReference{deploymentOwner},
		},
		{
			name: "matching Workload annotation in another namespace is ignored",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-other-ns", "other", "inference-workload", deploymentOwner),
			},
			workloadOwners: []metav1.OwnerReference{deploymentOwner},
		},
		{
			name: "all ReplicaSets delegating to the Workload are enqueued",
			replicaSets: []*apps.ReplicaSet{
				replicaSet("inference-a", metav1.NamespaceDefault, "inference-workload"),
				replicaSet("inference-b", metav1.NamespaceDefault, "inference-workload", deploymentOwner),
				replicaSet("inference-unrelated", metav1.NamespaceDefault, "other-workload", deploymentOwner),
			},
			wantKeys: []string{
				metav1.NamespaceDefault + "/inference-a",
				metav1.NamespaceDefault + "/inference-b",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, fake.NewSimpleClientset(), nil, BurstReplicas)
			for _, rs := range tc.replicaSets {
				if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
					t.Fatalf("adding ReplicaSet to informer: %v", err)
				}
			}
			workload := newWorkload("inference-workload", metav1.NamespaceDefault, 1, tc.workloadOwners...)

			rsc.enqueueWorkloadReplicaSets(workload)

			var gotKeys []string
			for rsc.queue.Len() > 0 {
				key, _ := rsc.queue.Get()
				rsc.queue.Done(key)
				gotKeys = append(gotKeys, key)
			}
			slices.Sort(gotKeys)
			if !slices.Equal(gotKeys, tc.wantKeys) {
				t.Fatalf("queued keys = %v, want %v", gotKeys, tc.wantKeys)
			}
		})
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

func TestSyncReplicaSetScalesDownOnPodGroupError(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	rs := newReplicaSet(1, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 1)
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: workload.Spec.PodGroupTemplates[0].Name,
	}
	client := fake.NewSimpleClientset(rs)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	fakePodControl := &controller.FakePodControl{}
	rsc.podControl = fakePodControl
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	// Two active Pods against one desired replica: a scale-down is due even though
	// the Workload named by the annotations is absent from the cache.
	for i := range 2 {
		pod := newPod(fmt.Sprintf("pod-%d", i), rs, v1.PodRunning, nil, true)
		if err := sharedInformers.Core().V1().Pods().Informer().GetIndexer().Add(pod); err != nil {
			t.Fatalf("adding Pod to informer: %v", err)
		}
	}

	err := rsc.syncReplicaSet(ctx, rs.Namespace+"/"+rs.Name)
	if err == nil {
		t.Fatal("syncReplicaSet() = nil error, want the PodGroup error surfaced")
	}
	if len(fakePodControl.Templates) != 0 {
		t.Fatalf("syncReplicaSet() created %d Pods, want 0 while the PodGroup could not be ensured", len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != 1 {
		t.Fatalf("syncReplicaSet() deleted %d Pods, want 1 excess Pod scaled down", len(fakePodControl.DeletePodName))
	}
}

func TestSyncReplicaSetMixedSchedulingGroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	ctx := context.Background()
	rs := newReplicaSet(2, map[string]string{"name": "foo"})
	rs.Name = "inference-abc123"
	workload := newWorkload("inference", rs.Namespace, 1)
	templateName := workload.Spec.PodGroupTemplates[0].Name
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workload.Name,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	pgName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, rs.UID)
	podGroup := newPodGroup(pgName, rs.Namespace, 1, *metav1.NewControllerRef(rs, apps.SchemeGroupVersion.WithKind("ReplicaSet")))
	podGroup.Spec.WorkloadRef = &schedulingv1beta1.WorkloadReference{WorkloadName: workload.Name, TemplateName: templateName}

	client := fake.NewSimpleClientset(rs, workload, podGroup)
	rsc, sharedInformers := testNewReplicaSetControllerFromClient(t, client, nil, BurstReplicas)
	fakePodControl := &controller.FakePodControl{}
	rsc.podControl = fakePodControl
	fakeRecorder := record.NewFakeRecorder(10)
	rsc.recorder = fakeRecorder
	if err := sharedInformers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs); err != nil {
		t.Fatalf("adding ReplicaSet to informer: %v", err)
	}
	if err := sharedInformers.Scheduling().V1beta1().Workloads().Informer().GetIndexer().Add(workload); err != nil {
		t.Fatalf("adding Workload to informer: %v", err)
	}
	if err := sharedInformers.Scheduling().V1beta1().PodGroups().Informer().GetIndexer().Add(podGroup); err != nil {
		t.Fatalf("adding PodGroup to informer: %v", err)
	}
	// One active Pod lacks a schedulingGroup (for example after a gate toggle).
	ungrouped := newPod("ungrouped", rs, v1.PodRunning, nil, true)
	if err := sharedInformers.Core().V1().Pods().Informer().GetIndexer().Add(ungrouped); err != nil {
		t.Fatalf("adding Pod to informer: %v", err)
	}

	if err := rsc.syncReplicaSet(ctx, rs.Namespace+"/"+rs.Name); err != nil {
		t.Fatalf("syncReplicaSet() error = %v", err)
	}
	if len(fakePodControl.Templates) != 1 {
		t.Fatalf("expected 1 pod creation, got %d", len(fakePodControl.Templates))
	}
	var gotEvents []string
	for len(fakeRecorder.Events) > 0 {
		fields := strings.SplitN(<-fakeRecorder.Events, " ", 3)
		gotEvents = append(gotEvents, strings.Join(fields[:2], " "))
	}
	if wantEvents := []string{"Warning MixedSchedulingGroup"}; !reflect.DeepEqual(gotEvents, wantEvents) {
		t.Errorf("events = %v, want %v", gotEvents, wantEvents)
	}
	for i, tmpl := range fakePodControl.Templates {
		if tmpl.Spec.SchedulingGroup != nil {
			t.Errorf("pod template [%d]: expected nil schedulingGroup, got %v", i, tmpl.Spec.SchedulingGroup)
		}
		if slices.ContainsFunc(tmpl.OwnerReferences, func(ref metav1.OwnerReference) bool { return ref.Kind == "PodGroup" }) {
			t.Errorf("pod template [%d]: ownerReferences = %+v, want no PodGroup owner", i, tmpl.OwnerReferences)
		}
		if ref := fakePodControl.ControllerRefs[i]; ref.Kind != "ReplicaSet" || ref.UID != rs.UID {
			t.Errorf("pod [%d]: controllerRef = %+v, want the ReplicaSet", i, ref)
		}
	}
}
