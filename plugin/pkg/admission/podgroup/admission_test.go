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

package podgroup

import (
	"testing"

	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

const (
	testNamespace    = "test"
	testWorkloadName = "my-workload"
	testTemplateName = "worker"
)

func newPodGroup(mutators ...func(*scheduling.PodGroup)) admission.Attributes {
	pg := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: testNamespace},
		Spec: scheduling.PodGroupSpec{
			PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{
				Workload: &scheduling.WorkloadPodGroupTemplateReference{
					WorkloadName:         testWorkloadName,
					PodGroupTemplateName: testTemplateName,
				},
			},
		},
	}
	for _, m := range mutators {
		m(pg)
	}
	return admission.NewAttributesRecord(
		pg, nil,
		scheduling.Kind("PodGroup").WithVersion("version"),
		testNamespace, "pg",
		scheduling.Resource("podgroups").WithVersion("version"),
		"", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
	)
}

func withWorkloadRef(workloadName, templateName string) func(*scheduling.PodGroup) {
	return func(pg *scheduling.PodGroup) {
		pg.Spec.PodGroupTemplateRef.Workload.WorkloadName = workloadName
		pg.Spec.PodGroupTemplateRef.Workload.PodGroupTemplateName = templateName
	}
}

func newWorkload(name string, templateNames ...string) *schedulingv1alpha2.Workload {
	w := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace},
		Spec: schedulingv1alpha2.WorkloadSpec{
			PodGroupTemplates: make([]schedulingv1alpha2.PodGroupTemplate, 0, len(templateNames)),
		},
	}
	for _, tn := range templateNames {
		w.Spec.PodGroupTemplates = append(w.Spec.PodGroupTemplates, schedulingv1alpha2.PodGroupTemplate{Name: tn})
	}
	return w
}

func TestHandles(t *testing.T) {
	plugin := NewPodGroupWorkloadExists()
	for op, shouldHandle := range map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  false,
		admission.Delete:  false,
		admission.Connect: false,
	} {
		if e, a := shouldHandle, plugin.Handles(op); e != a {
			t.Errorf("%v: shouldHandle=%t, handles=%t", op, e, a)
		}
	}
}

func TestValidate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	now := metav1.Now()

	deletingWorkload := newWorkload("deleting-workload", testTemplateName)
	deletingWorkload.DeletionTimestamp = &now

	workload := newWorkload(testWorkloadName, testTemplateName, "driver")

	cases := map[string]struct {
		enableFeatureGate bool
		workloads         []*schedulingv1alpha2.Workload
		attrs             admission.Attributes
		wantErr           bool
	}{
		"feature gate disabled, always allow": {
			attrs: newPodGroup(withWorkloadRef("non-existent", "worker")),
		},
		"no podGroupTemplateRef, allow": {
			enableFeatureGate: true,
			attrs: newPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef = nil
			}),
		},
		"nil workload ref, allow": {
			enableFeatureGate: true,
			attrs: newPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{}
			}),
		},
		"workload found, template exists": {
			enableFeatureGate: true,
			workloads:         []*schedulingv1alpha2.Workload{workload},
			attrs:             newPodGroup(),
		},
		"workload not found": {
			enableFeatureGate: true,
			attrs:             newPodGroup(withWorkloadRef("non-existent", testTemplateName)),
			wantErr:           true,
		},
		"workload being deleted": {
			enableFeatureGate: true,
			workloads:         []*schedulingv1alpha2.Workload{deletingWorkload},
			attrs:             newPodGroup(withWorkloadRef("deleting-workload", testTemplateName)),
			wantErr:           true,
		},
		"workload found but template name does not match": {
			enableFeatureGate: true,
			workloads:         []*schedulingv1alpha2.Workload{workload},
			attrs:             newPodGroup(withWorkloadRef(testWorkloadName, "non-existent-template")),
			wantErr:           true,
		},
		"ignores non-PodGroup resources": {
			enableFeatureGate: true,
			attrs: admission.NewAttributesRecord(
				nil, nil,
				scheduling.Kind("Workload").WithVersion("version"),
				testNamespace, "w1",
				scheduling.Resource("workloads").WithVersion("version"),
				"", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
			),
		},
		"ignores subresources": {
			enableFeatureGate: true,
			attrs: admission.NewAttributesRecord(
				&scheduling.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: testNamespace}}, nil,
				scheduling.Kind("PodGroup").WithVersion("version"),
				testNamespace, "pg",
				scheduling.Resource("podgroups").WithVersion("version"),
				"status", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
			),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t,
				utilfeature.DefaultFeatureGate, features.GenericWorkload, tc.enableFeatureGate)

			plugin := NewPodGroupWorkloadExists()
			plugin.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			plugin.SetExternalKubeInformerFactory(informerFactory)
			for _, w := range tc.workloads {
				if err := informerFactory.Scheduling().V1alpha2().Workloads().Informer().GetStore().Add(w); err != nil {
					t.Fatalf("failed to add Workload: %v", err)
				}
			}
			plugin.SetReadyFunc(func() bool { return true })
			plugin.SetExternalKubeClientSet(fake.NewSimpleClientset())

			err := plugin.Validate(ctx, tc.attrs, nil)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
