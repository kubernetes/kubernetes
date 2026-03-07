/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"testing"

	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/stretchr/testify/assert"
)

func newPluginForTest(
	enabled bool,
	listerObject *schedulingv1alpha2.Workload,
	clientObject *schedulingv1alpha2.Workload,
) *PodGroupWorkloadExists {
	plugin := NewPodGroupWorkloadExists()
	plugin.enabled = enabled

	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	plugin.SetExternalKubeInformerFactory(informerFactory)
	if listerObject != nil {
		informerFactory.Scheduling().V1alpha2().Workloads().Informer().GetStore().Add(listerObject)
	}
	// Override ready func so tests don't block waiting for informer sync.
	plugin.SetReadyFunc(func() bool { return true })

	var client kubernetes.Interface
	if clientObject != nil {
		client = fake.NewSimpleClientset(clientObject)
	} else {
		client = fake.NewSimpleClientset()
	}
	plugin.SetExternalKubeClientSet(client)

	return plugin
}

func newPodGroup(name, namespace, workloadName, templateName string) *scheduling.PodGroup {
	pg := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: scheduling.PodGroupSpec{
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{MinCount: 3},
			},
		},
	}
	if workloadName != "" {
		pg.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{
			Workload: &scheduling.WorkloadPodGroupTemplateReference{
				WorkloadName:         workloadName,
				PodGroupTemplateName: templateName,
			},
		}
	}
	return pg
}

func newWorkload(name, namespace string, templateNames ...string) *schedulingv1alpha2.Workload {
	w := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: schedulingv1alpha2.WorkloadSpec{
			PodGroupTemplates: make([]schedulingv1alpha2.PodGroupTemplate, 0, len(templateNames)),
		},
	}
	for _, tn := range templateNames {
		w.Spec.PodGroupTemplates = append(w.Spec.PodGroupTemplates, schedulingv1alpha2.PodGroupTemplate{
			Name: tn,
			SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 3},
			},
		})
	}
	return w
}

func podGroupAttributes(pg *scheduling.PodGroup) admission.Attributes {
	return admission.NewAttributesRecord(
		pg, nil,
		scheduling.Kind("PodGroup").WithVersion("version"),
		pg.Namespace, pg.Name,
		scheduling.Resource("podgroups").WithVersion("version"),
		"", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
	)
}

func TestValidateInitialization(t *testing.T) {
	tests := []struct {
		name        string
		setup       func() *PodGroupWorkloadExists
		expectError bool
	}{
		{
			name: "fully initialized",
			setup: func() *PodGroupWorkloadExists {
				return newPluginForTest(true, nil, nil)
			},
			expectError: false,
		},
		{
			name: "missing lister",
			setup: func() *PodGroupWorkloadExists {
				p := NewPodGroupWorkloadExists()
				p.SetExternalKubeClientSet(fake.NewSimpleClientset())
				return p
			},
			expectError: true,
		},
		{
			name: "missing client",
			setup: func() *PodGroupWorkloadExists {
				p := NewPodGroupWorkloadExists()
				f := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
				p.SetExternalKubeInformerFactory(f)
				return p
			},
			expectError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.setup().ValidateInitialization()
			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	workload := newWorkload("my-workload", "test", "worker", "driver")

	tests := []struct {
		name          string
		enabled       bool
		listerObject  *schedulingv1alpha2.Workload
		clientObject  *schedulingv1alpha2.Workload
		podGroup      *scheduling.PodGroup
		expectError   bool
		errorContains string
	}{
		{
			name:         "feature gate disabled, always allow",
			enabled:      false,
			listerObject: nil,
			clientObject: nil,
			podGroup:     newPodGroup("pg1", "test", "non-existent", "worker"),
			expectError:  false,
		},
		{
			name:         "no podGroupTemplateRef, allow",
			enabled:      true,
			listerObject: nil,
			clientObject: nil,
			podGroup: &scheduling.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "test"},
				Spec: scheduling.PodGroupSpec{
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 3},
					},
				},
			},
			expectError: false,
		},
		{
			name:         "nil workload ref, allow",
			enabled:      true,
			listerObject: nil,
			clientObject: nil,
			podGroup: &scheduling.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "test"},
				Spec: scheduling.PodGroupSpec{
					PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{},
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 3},
					},
				},
			},
			expectError: false,
		},
		{
			name:         "workload found by lister, template matches",
			enabled:      true,
			listerObject: workload,
			clientObject: nil,
			podGroup:     newPodGroup("pg1", "test", "my-workload", "worker"),
			expectError:  false,
		},
		{
			name:         "workload found by client (lister cache miss)",
			enabled:      true,
			listerObject: nil,
			clientObject: workload,
			podGroup:     newPodGroup("pg1", "test", "my-workload", "driver"),
			expectError:  false,
		},
		{
			name:          "workload not found",
			enabled:       true,
			listerObject:  nil,
			clientObject:  nil,
			podGroup:      newPodGroup("pg1", "test", "non-existent", "worker"),
			expectError:   true,
			errorContains: "Workload \"non-existent\" not found",
		},
		{
			name:          "workload found but template name does not match",
			enabled:       true,
			listerObject:  workload,
			clientObject:  nil,
			podGroup:      newPodGroup("pg1", "test", "my-workload", "non-existent-template"),
			expectError:   true,
			errorContains: "PodGroupTemplate \"non-existent-template\" not found in Workload \"my-workload\"",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plugin := newPluginForTest(tc.enabled, tc.listerObject, tc.clientObject)
			attrs := podGroupAttributes(tc.podGroup)
			err := plugin.Validate(context.TODO(), attrs, nil)
			if tc.expectError {
				assert.Error(t, err)
				if tc.errorContains != "" {
					assert.Contains(t, err.Error(), tc.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateIgnoresNonPodGroupResources(t *testing.T) {
	plugin := newPluginForTest(true, nil, nil)

	attrs := admission.NewAttributesRecord(
		nil, nil,
		scheduling.Kind("Workload").WithVersion("version"),
		"test", "w1",
		scheduling.Resource("workloads").WithVersion("version"),
		"", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
	)
	err := plugin.Validate(context.TODO(), attrs, nil)
	assert.NoError(t, err)
}

func TestValidateIgnoresSubresources(t *testing.T) {
	plugin := newPluginForTest(true, nil, nil)

	pg := newPodGroup("pg1", "test", "non-existent", "worker")
	attrs := admission.NewAttributesRecord(
		pg, nil,
		scheduling.Kind("PodGroup").WithVersion("version"),
		pg.Namespace, pg.Name,
		scheduling.Resource("podgroups").WithVersion("version"),
		"status", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{},
	)
	err := plugin.Validate(context.TODO(), attrs, nil)
	assert.NoError(t, err)
}
