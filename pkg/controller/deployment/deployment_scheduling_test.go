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

package deployment

import (
	"strings"
	"testing"

	apps "k8s.io/api/apps/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	testclient "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func newGangDeployment(name string, replicas int32, minCount *int32) *apps.Deployment {
	d := newDeployment(name, replicas, nil, nil, nil, map[string]string{"app": "test"})
	d.Spec.Strategy = apps.DeploymentStrategy{Type: apps.RecreateDeploymentStrategyType}
	d.Spec.Scheduling = &apps.DeploymentSchedulingConfiguration{
		SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: minCount},
		},
	}
	return d
}

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

func TestResolveMinCount(t *testing.T) {
	tests := []struct {
		name     string
		replicas *int32
		minCount *int32
		expected int32
	}{
		{
			name: "user-set minCount is used", replicas: new(int32(4)), minCount: new(int32(2)), expected: 2,
		},
		{
			name: "no user minCount derives from replicas", replicas: new(int32(5)), expected: 5,
		},
		{
			name: "replicas=0 without user minCount defaults to 1", replicas: new(int32(0)), expected: 1,
		},
		{
			name: "replicas=0 with user minCount preserves user value", replicas: new(int32(0)), minCount: new(int32(3)), expected: 3,
		},
		{
			name: "nil replicas defaults to 1", expected: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := newGangDeployment("test-deployment", 0, tt.minCount)
			d.Spec.Replicas = tt.replicas
			if got := resolveMinCount(d); got != tt.expected {
				t.Errorf("resolveMinCount() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func TestBuildWorkloadForDeployment(t *testing.T) {
	d := newGangDeployment("test-deployment", 3, new(int32(3)))
	d.Spec.Template.Spec.PriorityClassName = "high-priority"
	d.Spec.Scheduling.SchedulingConstraints = &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
		Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
	}
	d.Spec.Scheduling.DisruptionMode = &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
		All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
	}
	d.Spec.Scheduling.ResourceClaims = []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{
		Name: "gpu", ResourceClaimName: new("shared-gpu"),
	}}

	workload, err := buildWorkloadForDeployment(d)
	if err != nil {
		t.Fatalf("buildWorkloadForDeployment() error = %v", err)
	}

	expectedName := workloadbuilder.GenerateWorkloadName(d.Name, d.UID)
	if workload.Name != expectedName {
		t.Errorf("Workload.Name = %v, want %v", workload.Name, expectedName)
	}
	if workload.Namespace != "default" {
		t.Errorf("Workload.Namespace = %v, want default", workload.Namespace)
	}
	owner := metav1.GetControllerOf(workload)
	if owner == nil || owner.APIVersion != apps.SchemeGroupVersion.String() || owner.Kind != "Deployment" || owner.Name != d.Name || owner.UID != d.UID {
		t.Errorf("Workload controller owner = %#v, want Deployment %s/%s with UID %q", owner, d.Namespace, d.Name, d.UID)
	}
	controllerRef := workload.Spec.ControllerRef
	if controllerRef == nil || controllerRef.APIGroup != apps.GroupName || controllerRef.Kind != "Deployment" || controllerRef.Name != d.Name {
		t.Errorf("Workload controllerRef = %#v, want apps/Deployment %q", controllerRef, d.Name)
	}
	if len(workload.Spec.PodGroupTemplates) != 1 {
		t.Fatalf("Workload PodGroupTemplate count = %d, want 1", len(workload.Spec.PodGroupTemplates))
	}
	template := workload.Spec.PodGroupTemplates[0]
	if want := podGroupTemplateName(d); template.Name != want {
		t.Errorf("Workload template name = %q, want %q", template.Name, want)
	}
	if gang := template.SchedulingPolicy.Gang; gang == nil || gang.MinCount != 3 {
		t.Errorf("Workload template gang = %#v, want minCount 3", gang)
	}
	if template.PriorityClassName != "high-priority" {
		t.Errorf("Workload template priorityClassName = %q, want high-priority", template.PriorityClassName)
	}
	if constraints := template.SchedulingConstraints; constraints == nil || len(constraints.Topology) != 1 || constraints.Topology[0].Key != "topology.kubernetes.io/zone" {
		t.Errorf("Workload template schedulingConstraints = %+v, want topology key topology.kubernetes.io/zone", constraints)
	}
	if mode := template.DisruptionMode; mode == nil || mode.All == nil {
		t.Errorf("Workload template disruptionMode = %+v, want All", mode)
	}
	if len(template.ResourceClaims) != 1 {
		t.Fatalf("Workload template resource claim count = %d, want 1", len(template.ResourceClaims))
	}
	claim := template.ResourceClaims[0]
	if claim.Name != "gpu" {
		t.Errorf("Workload template claim name = %q, want gpu", claim.Name)
	}
	if name := ptr.Deref(claim.ResourceClaimName, ""); name != "shared-gpu" {
		t.Errorf("Workload template resourceClaimName = %q, want shared-gpu", name)
	}
}

func TestPodGroupTemplateName(t *testing.T) {
	tests := map[string]struct {
		deploymentName string
		want           string
	}{
		"short name is unchanged": {
			deploymentName: "test-deployment",
			want:           "test-deployment-pgt-0",
		},
		"name and suffix exactly fit the limit": {
			deploymentName: strings.Repeat("a", 57),
			want:           strings.Repeat("a", 57) + "-pgt-0",
		},
		"name and suffix exceed the limit by one": {
			deploymentName: strings.Repeat("a", 58),
			want:           strings.Repeat("a", 57) + "-pgt-0",
		},
		"maximum length name": {
			deploymentName: strings.Repeat("a", 63),
			want:           strings.Repeat("a", 57) + "-pgt-0",
		},
		"truncation preserves the name prefix": {
			deploymentName: strings.Repeat("a", 56) + "bcdefgh",
			want:           strings.Repeat("a", 56) + "b-pgt-0",
		},
		"dots are replaced with hyphens": {
			deploymentName: "app.v1",
			want:           "app-v1-pgt-0",
		},
		"multiple dots are replaced with hyphens": {
			deploymentName: "app.v1.4.0",
			want:           "app-v1-4-0-pgt-0",
		},
		"dot at truncation boundary is replaced with a hyphen": {
			deploymentName: strings.Repeat("a", 56) + ".daily",
			want:           strings.Repeat("a", 56) + "--pgt-0",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			d := &apps.Deployment{ObjectMeta: metav1.ObjectMeta{Name: tc.deploymentName}}
			if got := podGroupTemplateName(d); got != tc.want {
				t.Errorf("podGroupTemplateName() = %q (len %d), want %q", got, len(got), tc.want)
			}
		})
	}
}

func TestSyncDeployment_CreatesWorkloadBeforeReplicaSet(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	_, ctx := ktesting.NewTestContext(t)
	d := newGangDeployment("test-deployment", 3, nil)
	f := newFixture(t)
	f.objects = []runtime.Object{d}
	f.dLister = []*apps.Deployment{d}
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	if err := dc.syncDeployment(ctx, d.Namespace+"/"+d.Name); err != nil {
		t.Fatalf("syncDeployment() error = %v", err)
	}

	wlIndex, rsIndex := -1, -1
	var rs *apps.ReplicaSet
	for i, action := range f.client.Actions() {
		if createAction, ok := action.(testclient.CreateAction); ok {
			switch created := createAction.GetObject().(type) {
			case *schedulingv1beta1.Workload:
				if wlIndex == -1 {
					wlIndex = i
				}
			case *apps.ReplicaSet:
				if rs == nil {
					rs = created
					rsIndex = i
				}
			}
		}
	}
	if rs == nil {
		t.Fatal("syncDeployment() did not create a ReplicaSet")
	}

	if rs.Spec.Template.Spec.SchedulingGroup != nil {
		t.Error("ReplicaSet template should NOT have SchedulingGroup set (RS controller sets it at pod creation)")
	}

	if wlIndex == -1 {
		t.Fatal("syncDeployment() did not create a Workload")
	}
	if wlIndex >= rsIndex {
		t.Errorf("Workload (idx %d) must be created before ReplicaSet (idx %d)", wlIndex, rsIndex)
	}
}

func TestSyncDeployment_CreatesWorkloadAtZeroReplicas(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	for _, tc := range []struct {
		name         string
		minCount     *int32
		wantMinCount int32
	}{
		{name: "unset minCount defaults to 1", wantMinCount: 1},
		{name: "explicit minCount is preserved", minCount: new(int32(3)), wantMinCount: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			d := newGangDeployment("test-deployment", 0, tc.minCount)
			f := newFixture(t)
			f.objects = []runtime.Object{d}
			f.dLister = []*apps.Deployment{d}
			dc, _, err := f.newController(ctx)
			if err != nil {
				t.Fatalf("error creating Deployment controller: %v", err)
			}

			if err := dc.syncDeployment(ctx, d.Namespace+"/"+d.Name); err != nil {
				t.Fatalf("syncDeployment() error = %v", err)
			}
			workload, err := f.client.SchedulingV1beta1().Workloads(d.Namespace).Get(ctx, workloadbuilder.GenerateWorkloadName(d.Name, d.UID), metav1.GetOptions{})
			if err != nil {
				t.Fatalf("getting Workload created at zero replicas: %v", err)
			}
			if len(workload.Spec.PodGroupTemplates) != 1 {
				t.Fatalf("Workload PodGroupTemplate count = %d, want 1", len(workload.Spec.PodGroupTemplates))
			}
			if gang := workload.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang; gang == nil || gang.MinCount != tc.wantMinCount {
				t.Fatalf("Workload template gang = %#v, want minCount %d", gang, tc.wantMinCount)
			}
		})
	}
}

func TestSyncDeployment_WithoutScheduling(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	_, ctx := ktesting.NewTestContext(t)
	d := newDeployment("test-deployment", 3, nil, nil, nil, map[string]string{"app": "test"})
	d.Spec.Strategy = apps.DeploymentStrategy{Type: apps.RecreateDeploymentStrategyType}
	f := newFixture(t)
	f.objects = []runtime.Object{d}
	f.dLister = []*apps.Deployment{d}
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	if err := dc.syncDeployment(ctx, d.Namespace+"/"+d.Name); err != nil {
		t.Fatalf("syncDeployment() error = %v", err)
	}
	replicaSets, err := f.client.AppsV1().ReplicaSets(d.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("listing created ReplicaSets: %v", err)
	}
	if len(replicaSets.Items) != 1 {
		t.Fatalf("ReplicaSet count = %d, want 1", len(replicaSets.Items))
	}

	for _, action := range f.client.Actions() {
		if action.Matches("create", "workloads") {
			t.Error("Workload should not be created when Deployment has no scheduling")
		}
	}
}

func TestSyncDeployment_ReconcilesWorkloadMinCount(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	for _, tc := range []struct {
		name             string
		existingMinCount int32
		wantPatchCount   int
	}{
		{name: "updates mismatched minCount", existingMinCount: 3, wantPatchCount: 1},
		{name: "matching minCount needs no patch", existingMinCount: 2, wantPatchCount: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			d := newGangDeployment("test-deployment", 2, nil)
			workload := newWorkload(workloadbuilder.GenerateWorkloadName(d.Name, d.UID), d.Namespace, tc.existingMinCount, *metav1.NewControllerRef(d, controllerKind))
			f := newFixture(t)
			f.objects = []runtime.Object{d, workload}
			f.dLister = []*apps.Deployment{d}
			dc, informers, err := f.newController(ctx)
			if err != nil {
				t.Fatalf("error creating Deployment controller: %v", err)
			}
			if err := informers.Scheduling().V1beta1().Workloads().Informer().GetStore().Add(workload); err != nil {
				t.Fatalf("adding Workload to informer store: %v", err)
			}

			if err := dc.syncDeployment(ctx, d.Namespace+"/"+d.Name); err != nil {
				t.Fatalf("syncDeployment() error = %v", err)
			}
			patchCount := 0
			for _, action := range f.client.Actions() {
				if action.Matches("patch", "workloads") {
					patchCount++
				}
			}
			if patchCount != tc.wantPatchCount {
				t.Fatalf("Workload patch count = %d, want %d", patchCount, tc.wantPatchCount)
			}
			updated, err := f.client.SchedulingV1beta1().Workloads(d.Namespace).Get(ctx, workload.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("getting reconciled Workload: %v", err)
			}
			if len(updated.Spec.PodGroupTemplates) != 1 {
				t.Fatalf("Workload PodGroupTemplate count = %d, want 1", len(updated.Spec.PodGroupTemplates))
			}
			if gang := updated.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang; gang == nil || gang.MinCount != 2 {
				t.Fatalf("Workload template gang = %#v, want minCount 2", gang)
			}
		})
	}
}

func TestEnsureWorkloadForDeploymentRetriesAfterAlreadyExists(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	_, ctx := ktesting.NewTestContext(t)
	d := newGangDeployment("test-deployment", 3, nil)
	existing := newWorkload(workloadbuilder.GenerateWorkloadName(d.Name, d.UID), d.Namespace, 3, *metav1.NewControllerRef(d, controllerKind))
	f := newFixture(t)
	f.objects = []runtime.Object{existing}
	dc, informers, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	workload, err := dc.ensureWorkloadForDeployment(ctx, d)
	if !apierrors.IsAlreadyExists(err) {
		t.Fatalf("ensureWorkloadForDeployment() error = %v, want AlreadyExists", err)
	}
	if workload != nil {
		t.Fatalf("ensureWorkloadForDeployment() returned Workload %s on create failure", workload.Name)
	}

	if err := informers.Scheduling().V1beta1().Workloads().Informer().GetStore().Add(existing); err != nil {
		t.Fatalf("adding existing Workload to informer store: %v", err)
	}
	workload, err = dc.ensureWorkloadForDeployment(ctx, d)
	if err != nil {
		t.Fatalf("ensureWorkloadForDeployment() after cache update: %v", err)
	}
	if workload == nil || workload.Name != existing.Name {
		t.Fatalf("Workload after cache update = %#v, want Workload %q", workload, existing.Name)
	}
	workloads, err := f.client.SchedulingV1beta1().Workloads(d.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("listing Workloads after retry: %v", err)
	}
	if len(workloads.Items) != 1 {
		t.Fatalf("Workload count after retry = %d, want 1", len(workloads.Items))
	}
}

func TestEnsureWorkloadForDeploymentRejectsWorkloadWithMismatchedOwnership(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	_, ctx := ktesting.NewTestContext(t)
	d := newGangDeployment("test-deployment", 1, nil)
	d.UID = "deployment-uid"
	foreignDeployment := d.DeepCopy()
	foreignDeployment.UID = "foreign-uid"
	correctControllerRef := &schedulingv1beta1.TypedLocalObjectReference{
		APIGroup: apps.GroupName, Kind: "Deployment", Name: d.Name,
	}
	foreignControllerRef := &schedulingv1beta1.TypedLocalObjectReference{
		APIGroup: apps.GroupName, Kind: "Deployment", Name: "other-deployment",
	}
	cases := []struct {
		name          string
		owner         *apps.Deployment
		controllerRef *schedulingv1beta1.TypedLocalObjectReference
	}{
		{name: "different owner UID", owner: foreignDeployment, controllerRef: correctControllerRef},
		{name: "different spec controllerRef", owner: d, controllerRef: foreignControllerRef},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			workload := newWorkload(workloadbuilder.GenerateWorkloadName(d.Name, d.UID), d.Namespace, 2, *metav1.NewControllerRef(tc.owner, controllerKind))
			workload.Spec.ControllerRef = tc.controllerRef
			f := newFixture(t)
			f.objects = []runtime.Object{workload}
			dc, informers, err := f.newController(ctx)
			if err != nil {
				t.Fatalf("error creating Deployment controller: %v", err)
			}
			if err := informers.Scheduling().V1beta1().Workloads().Informer().GetStore().Add(workload); err != nil {
				t.Fatalf("adding Workload to informer store: %v", err)
			}

			if _, err := dc.ensureWorkloadForDeployment(ctx, d); err == nil || !strings.Contains(err.Error(), "is not controlled by Deployment") {
				t.Fatalf("ensureWorkloadForDeployment() error = %v, want ownership error", err)
			}
			for _, action := range f.client.Actions() {
				if action.Matches("patch", "workloads") {
					t.Fatal("ensureWorkloadForDeployment() patched a Workload with mismatched ownership")
				}
			}
		})
	}
}

func TestEnsureWorkloadForDeploymentRejectsUnexpectedPodGroupTemplateCount(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadWithDeployment, true)

	_, ctx := ktesting.NewTestContext(t)
	d := newGangDeployment("test-deployment", 5, nil)
	d.UID = "deployment-uid"
	firstTemplate := schedulingv1beta1.PodGroupTemplate{
		Name: "first",
		SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
			Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
		},
	}
	secondTemplate := firstTemplate
	secondTemplate.Name = "second"
	cases := []struct {
		name      string
		templates []schedulingv1beta1.PodGroupTemplate
		want      string
	}{
		{name: "zero templates", want: "has 0 PodGroupTemplates, want exactly 1"},
		{name: "multiple templates", templates: []schedulingv1beta1.PodGroupTemplate{firstTemplate, secondTemplate}, want: "has 2 PodGroupTemplates, want exactly 1"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			workload := newWorkload(workloadbuilder.GenerateWorkloadName(d.Name, d.UID), d.Namespace, 1, *metav1.NewControllerRef(d, controllerKind))
			workload.Spec.PodGroupTemplates = tc.templates
			f := newFixture(t)
			f.objects = []runtime.Object{workload}
			dc, informers, err := f.newController(ctx)
			if err != nil {
				t.Fatalf("error creating Deployment controller: %v", err)
			}
			if err := informers.Scheduling().V1beta1().Workloads().Informer().GetStore().Add(workload); err != nil {
				t.Fatalf("adding Workload to informer store: %v", err)
			}

			if _, err := dc.ensureWorkloadForDeployment(ctx, d); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ensureWorkloadForDeployment() error = %v, want error containing %q", err, tc.want)
			}
			for _, action := range f.client.Actions() {
				if action.Matches("patch", "workloads") {
					t.Fatal("ensureWorkloadForDeployment() patched a Workload with an unexpected template count")
				}
			}
		})
	}
}
