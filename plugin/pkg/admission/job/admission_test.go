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

package job

import (
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/controller"
)

func TestGangSchedulingParallelism(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	indexedMode := batch.IndexedCompletion
	oldJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-job",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("test-job-uid"),
		},
		Spec: batch.JobSpec{
			CompletionMode: &indexedMode,
			Completions:    new(int32(4)),
			Parallelism:    new(int32(4)),
		},
	}
	newJob := oldJob.DeepCopy()
	newJob.Spec.Parallelism = new(int32(2))

	p := NewPlugin()
	p.InspectFeatureGates(utilfeature.DefaultFeatureGate)
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	p.SetExternalKubeInformerFactory(informerFactory)

	// A gang PodGroup owned by the Job exists: the plugin must still allow the
	// parallelism change (the v1.36 block is gone).
	gangPodGroup := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gang-pg",
			Namespace: metav1.NamespaceDefault,
			OwnerReferences: []metav1.OwnerReference{
				{APIVersion: "batch/v1", Kind: "Job", Name: "test-job", UID: types.UID("test-job-uid"), Controller: new(true)},
			},
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4},
			},
		},
	}
	if err := informerFactory.Scheduling().V1alpha3().PodGroups().Informer().GetStore().Add(gangPodGroup); err != nil {
		t.Fatalf("failed to add PodGroup: %v", err)
	}

	attrs := admission.NewAttributesRecord(
		newJob, oldJob,
		schema.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
		metav1.NamespaceDefault, "test-job",
		batch.Resource("jobs").WithVersion("v1"),
		"", admission.Update, &metav1.UpdateOptions{}, false, nil,
	)

	if err := p.Validate(ctx, attrs, nil); err != nil {
		t.Errorf("expected parallelism change to be allowed for gang-scheduled Job, got: %v", err)
	}
}
