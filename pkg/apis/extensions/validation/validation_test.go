/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestValidateDaemonSetStatusUpdate(t *testing.T) {
	type dsUpdateTest struct {
		old    extensions.DaemonSet
		update extensions.DaemonSet
	}

	successCases := []dsUpdateTest{
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
				},
			},
		},
	}

	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateDaemonSetStatusUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]dsUpdateTest{
		"negative values": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: -1,
					NumberMisscheduled:     -1,
					DesiredNumberScheduled: -3,
					NumberReady:            -1,
					ObservedGeneration:     -3,
				},
			},
		},
		"negative CurrentNumberScheduled": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: -1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
		},
		"negative NumberMisscheduled": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     -1,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
		},
		"negative DesiredNumberScheduled": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: -3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
		},
		"negative NumberReady": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: 3,
					NumberReady:            -1,
					ObservedGeneration:     3,
				},
			},
		},
		"negative ObservedGeneration": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					ObservedGeneration:     -3,
				},
			},
		},
	}

	for testName, errorCase := range errorCases {
		if errs := ValidateDaemonSetStatusUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateDaemonSetUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validSelector2 := map[string]string{"c": "d"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}

	validPodSpecAbc := api.PodSpec{
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecDef := api.PodSpec{
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecNodeSelector := api.PodSpec{
		NodeSelector:  validSelector,
		NodeName:      "xyz",
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecVolume := api.PodSpec{
		Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}

	validPodTemplateAbc := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecAbc,
		},
	}
	validPodTemplateNodeSelector := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecNodeSelector,
		},
	}
	validPodTemplateAbc2 := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector2,
			},
			Spec: validPodSpecAbc,
		},
	}
	validPodTemplateDef := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector2,
			},
			Spec: validPodSpecDef,
		},
	}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecVolume,
		},
	}

	type dsUpdateTest struct {
		old    extensions.DaemonSet
		update extensions.DaemonSet
	}
	successCases := []dsUpdateTest{
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector2},
					Template: validPodTemplateAbc2.Template,
				},
			},
		},
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateNodeSelector.Template,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateDaemonSetUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]dsUpdateTest{
		"change daemon name": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		"invalid selector": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: invalidSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		"invalid pod": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: invalidPodTemplate.Template,
				},
			},
		},
		"change container image": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateDef.Template,
				},
			},
		},
		"read-write volume": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid update strategy": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: invalidSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateDaemonSetUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateDaemonSet(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	successCases := []extensions.DaemonSet{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateDaemonSet(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]extensions.DaemonSet{
		"zero-length ID": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"nil selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Template: validPodTemplate.Template,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{},
				Template: validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				Template: validPodTemplate.Template,
			},
		},
		"invalid template": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
			},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"invalid_label 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.DaemonSetSpec{
				Template: invalidPodTemplate.Template,
			},
		},
		"invalid_annotation": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validSelector},
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateDaemonSet(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].Field
			if !strings.HasPrefix(field, "spec.template.") &&
				!strings.HasPrefix(field, "spec.updateStrategy") &&
				field != "metadata.name" &&
				field != "metadata.namespace" &&
				field != "spec.selector" &&
				field != "spec.template" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "spec.template.labels" &&
				field != "metadata.annotations" &&
				field != "metadata.labels" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func validDeployment() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": "abc",
				},
			},
			Strategy: extensions.DeploymentStrategy{
				Type: extensions.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &extensions.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt(1),
					MaxUnavailable: intstr.FromInt(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:      "abc",
					Namespace: api.NamespaceDefault,
					Labels: map[string]string{
						"name": "abc",
					},
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					Containers: []api.Container{
						{
							Name:            "nginx",
							Image:           "image",
							ImagePullPolicy: api.PullNever,
						},
					},
				},
			},
			RollbackTo: &extensions.RollbackConfig{
				Revision: 1,
			},
		},
	}
}

func TestValidateDeployment(t *testing.T) {
	successCases := []*extensions.Deployment{
		validDeployment(),
	}
	for _, successCase := range successCases {
		if errs := ValidateDeployment(successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]*extensions.Deployment{}
	errorCases["metadata.name: Required value"] = &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
		},
	}
	// selector should match the labels in pod template.
	invalidSelectorDeployment := validDeployment()
	invalidSelectorDeployment.Spec.Selector = &metav1.LabelSelector{
		MatchLabels: map[string]string{
			"name": "def",
		},
	}
	errorCases["`selector` does not match template `labels`"] = invalidSelectorDeployment

	// RestartPolicy should be always.
	invalidRestartPolicyDeployment := validDeployment()
	invalidRestartPolicyDeployment.Spec.Template.Spec.RestartPolicy = api.RestartPolicyNever
	errorCases["Unsupported value: \"Never\""] = invalidRestartPolicyDeployment

	// must have valid strategy type
	invalidStrategyDeployment := validDeployment()
	invalidStrategyDeployment.Spec.Strategy.Type = extensions.DeploymentStrategyType("randomType")
	errorCases["supported values: Recreate, RollingUpdate"] = invalidStrategyDeployment

	// rollingUpdate should be nil for recreate.
	invalidRecreateDeployment := validDeployment()
	invalidRecreateDeployment.Spec.Strategy = extensions.DeploymentStrategy{
		Type:          extensions.RecreateDeploymentStrategyType,
		RollingUpdate: &extensions.RollingUpdateDeployment{},
	}
	errorCases["may not be specified when strategy `type` is 'Recreate'"] = invalidRecreateDeployment

	// MaxSurge should be in the form of 20%.
	invalidMaxSurgeDeployment := validDeployment()
	invalidMaxSurgeDeployment.Spec.Strategy = extensions.DeploymentStrategy{
		Type: extensions.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &extensions.RollingUpdateDeployment{
			MaxSurge: intstr.FromString("20Percent"),
		},
	}
	errorCases["a valid percent string must be"] = invalidMaxSurgeDeployment

	// MaxSurge and MaxUnavailable cannot both be zero.
	invalidRollingUpdateDeployment := validDeployment()
	invalidRollingUpdateDeployment.Spec.Strategy = extensions.DeploymentStrategy{
		Type: extensions.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &extensions.RollingUpdateDeployment{
			MaxSurge:       intstr.FromString("0%"),
			MaxUnavailable: intstr.FromInt(0),
		},
	}
	errorCases["may not be 0 when `maxSurge` is 0"] = invalidRollingUpdateDeployment

	// MaxUnavailable should not be more than 100%.
	invalidMaxUnavailableDeployment := validDeployment()
	invalidMaxUnavailableDeployment.Spec.Strategy = extensions.DeploymentStrategy{
		Type: extensions.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &extensions.RollingUpdateDeployment{
			MaxUnavailable: intstr.FromString("110%"),
		},
	}
	errorCases["must not be greater than 100%"] = invalidMaxUnavailableDeployment

	// Rollback.Revision must be non-negative
	invalidRollbackRevisionDeployment := validDeployment()
	invalidRollbackRevisionDeployment.Spec.RollbackTo.Revision = -3
	errorCases["must be greater than or equal to 0"] = invalidRollbackRevisionDeployment

	// ProgressDeadlineSeconds should be greater than MinReadySeconds
	invalidProgressDeadlineDeployment := validDeployment()
	seconds := int32(600)
	invalidProgressDeadlineDeployment.Spec.ProgressDeadlineSeconds = &seconds
	invalidProgressDeadlineDeployment.Spec.MinReadySeconds = seconds
	errorCases["must be greater than minReadySeconds"] = invalidProgressDeadlineDeployment

	for k, v := range errorCases {
		errs := ValidateDeployment(v)
		if len(errs) == 0 {
			t.Errorf("[%s] expected failure", k)
		} else if !strings.Contains(errs[0].Error(), k) {
			t.Errorf("unexpected error: %q, expected: %q", errs[0].Error(), k)
		}
	}
}

func validDeploymentRollback() *extensions.DeploymentRollback {
	return &extensions.DeploymentRollback{
		Name: "abc",
		UpdatedAnnotations: map[string]string{
			"created-by": "abc",
		},
		RollbackTo: extensions.RollbackConfig{
			Revision: 1,
		},
	}
}

func TestValidateDeploymentRollback(t *testing.T) {
	noAnnotation := validDeploymentRollback()
	noAnnotation.UpdatedAnnotations = nil
	successCases := []*extensions.DeploymentRollback{
		validDeploymentRollback(),
		noAnnotation,
	}
	for _, successCase := range successCases {
		if errs := ValidateDeploymentRollback(successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]*extensions.DeploymentRollback{}
	invalidNoName := validDeploymentRollback()
	invalidNoName.Name = ""
	errorCases["name: Required value"] = invalidNoName

	for k, v := range errorCases {
		errs := ValidateDeploymentRollback(v)
		if len(errs) == 0 {
			t.Errorf("[%s] expected failure", k)
		} else if !strings.Contains(errs[0].Error(), k) {
			t.Errorf("unexpected error: %q, expected: %q", errs[0].Error(), k)
		}
	}
}

type ingressRules map[string]string

func TestValidateIngress(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}
	servicelessBackend := newValid()
	servicelessBackend.Spec.Backend.ServiceName = ""
	invalidNameBackend := newValid()
	invalidNameBackend.Spec.Backend.ServiceName = "defaultBackend"
	noPortBackend := newValid()
	noPortBackend.Spec.Backend = &extensions.IngressBackend{ServiceName: defaultBackend.ServiceName}
	noForwardSlashPath := newValid()
	noForwardSlashPath.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{
		{
			Path:    "invalid",
			Backend: defaultBackend,
		},
	}
	noPaths := newValid()
	noPaths.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{}
	badHost := newValid()
	badHost.Spec.Rules[0].Host = "foobar:80"
	badRegexPath := newValid()
	badPathExpr := "/invalid["
	badRegexPath.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{
		{
			Path:    badPathExpr,
			Backend: defaultBackend,
		},
	}
	badPathErr := fmt.Sprintf("spec.rules[0].http.paths[0].path: Invalid value: '%v'", badPathExpr)
	hostIP := "127.0.0.1"
	badHostIP := newValid()
	badHostIP.Spec.Rules[0].Host = hostIP
	badHostIPErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", hostIP)

	errorCases := map[string]extensions.Ingress{
		"spec.backend.serviceName: Required value":        servicelessBackend,
		"spec.backend.serviceName: Invalid value":         invalidNameBackend,
		"spec.backend.servicePort: Invalid value":         noPortBackend,
		"spec.rules[0].host: Invalid value":               badHost,
		"spec.rules[0].http.paths: Required value":        noPaths,
		"spec.rules[0].http.paths[0].path: Invalid value": noForwardSlashPath,
	}
	errorCases[badPathErr] = badRegexPath
	errorCases[badHostIPErr] = badHostIP

	wildcardHost := "foo.*.bar.com"
	badWildcard := newValid()
	badWildcard.Spec.Rules[0].Host = wildcardHost
	badWildcardErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardErr] = badWildcard

	for k, v := range errorCases {
		errs := ValidateIngress(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressTLS(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}

	errorCases := map[string]extensions.Ingress{}

	wildcardHost := "foo.*.bar.com"
	badWildcardTLS := newValid()
	badWildcardTLS.Spec.Rules[0].Host = "*.foo.bar.com"
	badWildcardTLS.Spec.TLS = []extensions.IngressTLS{
		{
			Hosts: []string{wildcardHost},
		},
	}
	badWildcardTLSErr := fmt.Sprintf("spec.tls[0].hosts: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardTLSErr] = badWildcardTLS

	for k, v := range errorCases {
		errs := ValidateIngress(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressStatusUpdate(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       api.NamespaceDefault,
				ResourceVersion: "9",
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1", Hostname: "foo.bar.com"},
					},
				},
			},
		}
	}
	oldValue := newValid()
	newValue := newValid()
	newValue.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2", Hostname: "foo.com"},
			},
		},
	}
	invalidIP := newValid()
	invalidIP.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "abcd", Hostname: "foo.com"},
			},
		},
	}
	invalidHostname := newValid()
	invalidHostname.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.1", Hostname: "127.0.0.1"},
			},
		},
	}

	errs := ValidateIngressStatusUpdate(&newValue, &oldValue)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}

	errorCases := map[string]extensions.Ingress{
		"status.loadBalancer.ingress[0].ip: Invalid value":       invalidIP,
		"status.loadBalancer.ingress[0].hostname: Invalid value": invalidHostname,
	}
	for k, v := range errorCases {
		errs := ValidateIngressStatusUpdate(&v, &oldValue)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateScale(t *testing.T) {
	successCases := []extensions.Scale{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ScaleSpec{
				Replicas: 1,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ScaleSpec{
				Replicas: 10,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ScaleSpec{
				Replicas: 0,
			},
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateScale(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		scale extensions.Scale
		msg   string
	}{
		{
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					Name:      "frontend",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.ScaleSpec{
					Replicas: -1,
				},
			},
			msg: "must be greater than or equal to 0",
		},
	}

	for _, c := range errorCases {
		if errs := ValidateScale(&c.scale); len(errs) == 0 {
			t.Errorf("expected failure for %s", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], c.msg)
		}
	}
}

func TestValidateReplicaSetStatusUpdate(t *testing.T) {
	validLabels := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validLabels,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	type rcUpdateTest struct {
		old    extensions.ReplicaSet
		update extensions.ReplicaSet
	}
	successCases := []rcUpdateTest{
		{
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
				Status: extensions.ReplicaSetStatus{
					Replicas: 2,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 3,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
				Status: extensions.ReplicaSetStatus{
					Replicas: 4,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicaSetStatusUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"negative replicas": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
				Status: extensions.ReplicaSetStatus{
					Replicas: 3,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
				Status: extensions.ReplicaSetStatus{
					Replicas: -3,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicaSetStatusUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}

}

func TestValidateReplicaSetUpdate(t *testing.T) {
	validLabels := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validLabels,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validLabels,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
			},
		},
	}
	invalidLabels := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidLabels,
			},
		},
	}
	type rcUpdateTest struct {
		old    extensions.ReplicaSet
		update extensions.ReplicaSet
	}
	successCases := []rcUpdateTest{
		{
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 3,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
		},
		{
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 1,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: readWriteVolumePodTemplate.Template,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicaSetUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"more than one read/write": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid selector": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &metav1.LabelSelector{MatchLabels: invalidLabels},
					Template: validPodTemplate.Template,
				},
			},
		},
		"invalid pod": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: invalidPodTemplate.Template,
				},
			},
		},
		"negative replicas": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: -1,
					Selector: &metav1.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicaSetUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateReplicaSet(t *testing.T) {
	validLabels := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validLabels,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validLabels,
			},
			Spec: api.PodSpec{
				Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	invalidLabels := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidLabels,
			},
		},
	}
	successCases := []extensions.ReplicaSet{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Replicas: 1,
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: readWriteVolumePodTemplate.Template,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateReplicaSet(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]extensions.ReplicaSet{
		"zero-length ID": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Template: validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				Template: validPodTemplate.Template,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
			},
		},
		"read-write persistent disk with > 1 pod": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Spec: extensions.ReplicaSetSpec{
				Replicas: 2,
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: readWriteVolumePodTemplate.Template,
			},
		},
		"negative_replicas": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Replicas: -1,
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
			},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"invalid_label 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.ReplicaSetSpec{
				Template: invalidPodTemplate.Template,
			},
		},
		"invalid_annotation": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validLabels,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: validLabels},
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validLabels,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateReplicaSet(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].Field
			if !strings.HasPrefix(field, "spec.template.") &&
				field != "metadata.name" &&
				field != "metadata.namespace" &&
				field != "spec.selector" &&
				field != "spec.template" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "spec.replicas" &&
				field != "spec.template.labels" &&
				field != "metadata.annotations" &&
				field != "metadata.labels" &&
				field != "status.replicas" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func TestValidatePodSecurityPolicy(t *testing.T) {
	validPSP := func() *extensions.PodSecurityPolicy {
		return &extensions.PodSecurityPolicy{
			ObjectMeta: api.ObjectMeta{
				Name:        "foo",
				Annotations: map[string]string{},
			},
			Spec: extensions.PodSecurityPolicySpec{
				SELinux: extensions.SELinuxStrategyOptions{
					Rule: extensions.SELinuxStrategyRunAsAny,
				},
				RunAsUser: extensions.RunAsUserStrategyOptions{
					Rule: extensions.RunAsUserStrategyRunAsAny,
				},
				FSGroup: extensions.FSGroupStrategyOptions{
					Rule: extensions.FSGroupStrategyRunAsAny,
				},
				SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
					Rule: extensions.SupplementalGroupsStrategyRunAsAny,
				},
			},
		}
	}

	noUserOptions := validPSP()
	noUserOptions.Spec.RunAsUser.Rule = ""

	noSELinuxOptions := validPSP()
	noSELinuxOptions.Spec.SELinux.Rule = ""

	invalidUserStratType := validPSP()
	invalidUserStratType.Spec.RunAsUser.Rule = "invalid"

	invalidSELinuxStratType := validPSP()
	invalidSELinuxStratType.Spec.SELinux.Rule = "invalid"

	invalidUIDPSP := validPSP()
	invalidUIDPSP.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAs
	invalidUIDPSP.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{Min: -1, Max: 1},
	}

	missingObjectMetaName := validPSP()
	missingObjectMetaName.ObjectMeta.Name = ""

	noFSGroupOptions := validPSP()
	noFSGroupOptions.Spec.FSGroup.Rule = ""

	invalidFSGroupStratType := validPSP()
	invalidFSGroupStratType.Spec.FSGroup.Rule = "invalid"

	noSupplementalGroupsOptions := validPSP()
	noSupplementalGroupsOptions.Spec.SupplementalGroups.Rule = ""

	invalidSupGroupStratType := validPSP()
	invalidSupGroupStratType.Spec.SupplementalGroups.Rule = "invalid"

	invalidRangeMinGreaterThanMax := validPSP()
	invalidRangeMinGreaterThanMax.Spec.FSGroup.Ranges = []extensions.IDRange{
		{Min: 2, Max: 1},
	}

	invalidRangeNegativeMin := validPSP()
	invalidRangeNegativeMin.Spec.FSGroup.Ranges = []extensions.IDRange{
		{Min: -1, Max: 10},
	}

	invalidRangeNegativeMax := validPSP()
	invalidRangeNegativeMax.Spec.FSGroup.Ranges = []extensions.IDRange{
		{Min: 1, Max: -10},
	}

	requiredCapAddAndDrop := validPSP()
	requiredCapAddAndDrop.Spec.DefaultAddCapabilities = []api.Capability{"foo"}
	requiredCapAddAndDrop.Spec.RequiredDropCapabilities = []api.Capability{"foo"}

	allowedCapListedInRequiredDrop := validPSP()
	allowedCapListedInRequiredDrop.Spec.RequiredDropCapabilities = []api.Capability{"foo"}
	allowedCapListedInRequiredDrop.Spec.AllowedCapabilities = []api.Capability{"foo"}

	invalidAppArmorDefault := validPSP()
	invalidAppArmorDefault.Annotations = map[string]string{
		apparmor.DefaultProfileAnnotationKey: "not-good",
	}
	invalidAppArmorAllowed := validPSP()
	invalidAppArmorAllowed.Annotations = map[string]string{
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault + ",not-good",
	}

	invalidSysctlPattern := validPSP()
	invalidSysctlPattern.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "a.*.b"

	invalidSeccompDefault := validPSP()
	invalidSeccompDefault.Annotations = map[string]string{
		seccomp.DefaultProfileAnnotationKey: "not-good",
	}
	invalidSeccompAllowed := validPSP()
	invalidSeccompAllowed.Annotations = map[string]string{
		seccomp.AllowedProfilesAnnotationKey: "docker/default,not-good",
	}

	type testCase struct {
		psp         *extensions.PodSecurityPolicy
		errorType   field.ErrorType
		errorDetail string
	}
	errorCases := map[string]testCase{
		"no user options": {
			psp:         noUserOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, MustRunAsNonRoot, RunAsAny",
		},
		"no selinux options": {
			psp:         noSELinuxOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"no fsgroup options": {
			psp:         noFSGroupOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"no sup group options": {
			psp:         noSupplementalGroupsOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"invalid user strategy type": {
			psp:         invalidUserStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, MustRunAsNonRoot, RunAsAny",
		},
		"invalid selinux strategy type": {
			psp:         invalidSELinuxStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"invalid sup group strategy type": {
			psp:         invalidSupGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"invalid fs group strategy type": {
			psp:         invalidFSGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"invalid uid": {
			psp:         invalidUIDPSP,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"missing object meta name": {
			psp:         missingObjectMetaName,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "name or generateName is required",
		},
		"invalid range min greater than max": {
			psp:         invalidRangeMinGreaterThanMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be greater than max",
		},
		"invalid range negative min": {
			psp:         invalidRangeNegativeMin,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"invalid range negative max": {
			psp:         invalidRangeNegativeMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "max cannot be negative",
		},
		"invalid required caps": {
			psp:         requiredCapAddAndDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in defaultAddCapabilities and requiredDropCapabilities",
		},
		"allowed cap listed in required drops": {
			psp:         allowedCapListedInRequiredDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in allowedCapabilities and requiredDropCapabilities",
		},
		"invalid AppArmor default profile": {
			psp:         invalidAppArmorDefault,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid AppArmor profile name: \"not-good\"",
		},
		"invalid AppArmor allowed profile": {
			psp:         invalidAppArmorAllowed,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid AppArmor profile name: \"not-good\"",
		},
		"invalid sysctl pattern": {
			psp:         invalidSysctlPattern,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("must have at most 253 characters and match regex %s", SysctlPatternFmt),
		},
		"invalid seccomp default profile": {
			psp:         invalidSeccompDefault,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be a valid seccomp profile",
		},
		"invalid seccomp allowed profile": {
			psp:         invalidSeccompAllowed,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be a valid seccomp profile",
		},
	}

	for k, v := range errorCases {
		errs := ValidatePodSecurityPolicy(v.psp)
		if len(errs) == 0 {
			t.Errorf("%s expected errors but got none", k)
			continue
		}
		if errs[0].Type != v.errorType {
			t.Errorf("[%s] received an unexpected error type.  Expected: '%s' got: '%s'", k, v.errorType, errs[0].Type)
		}
		if errs[0].Detail != v.errorDetail {
			t.Errorf("[%s] received an unexpected error detail.  Expected '%s' got: '%s'", k, v.errorDetail, errs[0].Detail)
		}
	}

	// Update error is different for 'missing object meta name'.
	errorCases["missing object meta name"] = testCase{
		psp:         errorCases["missing object meta name"].psp,
		errorType:   field.ErrorTypeInvalid,
		errorDetail: "field is immutable",
	}

	// Should not be able to update to an invalid policy.
	for k, v := range errorCases {
		v.psp.ResourceVersion = "444" // Required for updates.
		errs := ValidatePodSecurityPolicyUpdate(validPSP(), v.psp)
		if len(errs) == 0 {
			t.Errorf("[%s] expected update errors but got none", k)
			continue
		}
		if errs[0].Type != v.errorType {
			t.Errorf("[%s] received an unexpected error type.  Expected: '%s' got: '%s'", k, v.errorType, errs[0].Type)
		}
		if errs[0].Detail != v.errorDetail {
			t.Errorf("[%s] received an unexpected error detail.  Expected '%s' got: '%s'", k, v.errorDetail, errs[0].Detail)
		}
	}

	mustRunAs := validPSP()
	mustRunAs.Spec.FSGroup.Rule = extensions.FSGroupStrategyMustRunAs
	mustRunAs.Spec.SupplementalGroups.Rule = extensions.SupplementalGroupsStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{Min: 1, Max: 1},
	}
	mustRunAs.Spec.SELinux.Rule = extensions.SELinuxStrategyMustRunAs

	runAsNonRoot := validPSP()
	runAsNonRoot.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAsNonRoot

	caseInsensitiveAddDrop := validPSP()
	caseInsensitiveAddDrop.Spec.DefaultAddCapabilities = []api.Capability{"foo"}
	caseInsensitiveAddDrop.Spec.RequiredDropCapabilities = []api.Capability{"FOO"}

	caseInsensitiveAllowedDrop := validPSP()
	caseInsensitiveAllowedDrop.Spec.RequiredDropCapabilities = []api.Capability{"FOO"}
	caseInsensitiveAllowedDrop.Spec.AllowedCapabilities = []api.Capability{"foo"}

	validAppArmor := validPSP()
	validAppArmor.Annotations = map[string]string{
		apparmor.DefaultProfileAnnotationKey:  apparmor.ProfileRuntimeDefault,
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault + "," + apparmor.ProfileNamePrefix + "foo",
	}

	withSysctl := validPSP()
	withSysctl.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "net.*"

	validSeccomp := validPSP()
	validSeccomp.Annotations = map[string]string{
		seccomp.DefaultProfileAnnotationKey:  "docker/default",
		seccomp.AllowedProfilesAnnotationKey: "docker/default,unconfined,localhost/foo",
	}

	successCases := map[string]struct {
		psp *extensions.PodSecurityPolicy
	}{
		"must run as": {
			psp: mustRunAs,
		},
		"run as any": {
			psp: validPSP(),
		},
		"run as non-root (user only)": {
			psp: runAsNonRoot,
		},
		"comparison for add -> drop is case sensitive": {
			psp: caseInsensitiveAddDrop,
		},
		"comparison for allowed -> drop is case sensitive": {
			psp: caseInsensitiveAllowedDrop,
		},
		"valid AppArmor annotations": {
			psp: validAppArmor,
		},
		"with network sysctls": {
			psp: withSysctl,
		},
		"valid seccomp annotations": {
			psp: validSeccomp,
		},
	}

	for k, v := range successCases {
		if errs := ValidatePodSecurityPolicy(v.psp); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}

		// Should be able to update to a valid PSP.
		v.psp.ResourceVersion = "444" // Required for updates.
		if errs := ValidatePodSecurityPolicyUpdate(validPSP(), v.psp); len(errs) != 0 {
			t.Errorf("Expected success for %s update, got %v", k, errs)
		}
	}
}

func TestValidatePSPVolumes(t *testing.T) {
	validPSP := func() *extensions.PodSecurityPolicy {
		return &extensions.PodSecurityPolicy{
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: extensions.PodSecurityPolicySpec{
				SELinux: extensions.SELinuxStrategyOptions{
					Rule: extensions.SELinuxStrategyRunAsAny,
				},
				RunAsUser: extensions.RunAsUserStrategyOptions{
					Rule: extensions.RunAsUserStrategyRunAsAny,
				},
				FSGroup: extensions.FSGroupStrategyOptions{
					Rule: extensions.FSGroupStrategyRunAsAny,
				},
				SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
					Rule: extensions.SupplementalGroupsStrategyRunAsAny,
				},
			},
		}
	}

	volumes := psputil.GetAllFSTypesAsSet()
	// add in the * value since that is a pseudo type that is not included by default
	volumes.Insert(string(extensions.All))

	for _, strVolume := range volumes.List() {
		psp := validPSP()
		psp.Spec.Volumes = []extensions.FSType{extensions.FSType(strVolume)}
		errs := ValidatePodSecurityPolicy(psp)
		if len(errs) != 0 {
			t.Errorf("%s validation expected no errors but received %v", strVolume, errs)
		}
	}
}

func TestValidateNetworkPolicy(t *testing.T) {
	successCases := []extensions.NetworkPolicy{
		{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []extensions.NetworkPolicyIngressRule{},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From:  []extensions.NetworkPolicyPeer{},
						Ports: []extensions.NetworkPolicyPort{},
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
			},
		},
	}

	// Success cases are expected to pass validation.
	for k, v := range successCases {
		if errs := ValidateNetworkPolicy(&v); len(errs) != 0 {
			t.Errorf("Expected success for %d, got %v", k, errs)
		}
	}

	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	errorCases := map[string]extensions.NetworkPolicy{
		"namespaceSelector and podSelector": {
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
			},
		},
		"invalid spec.podSelector": {
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: invalidSelector,
				},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
			},
		},
		"invalid ingress.from.podSelector": {
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: invalidSelector,
								},
							},
						},
					},
				},
			},
		},
		"invalid ingress.from.namespaceSelector": {
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: extensions.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []extensions.NetworkPolicyIngressRule{
					{
						From: []extensions.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: invalidSelector,
								},
							},
						},
					},
				},
			},
		},
	}

	// Error cases are not expected to pass validation.
	for testName, networkPolicy := range errorCases {
		if errs := ValidateNetworkPolicy(&networkPolicy); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}

func TestValidateNetworkPolicyUpdate(t *testing.T) {
	type npUpdateTest struct {
		old    extensions.NetworkPolicy
		update extensions.NetworkPolicy
	}
	successCases := []npUpdateTest{
		{
			old: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []extensions.NetworkPolicyIngressRule{},
				},
			},
			update: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []extensions.NetworkPolicyIngressRule{},
				},
			},
		},
	}

	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateNetworkPolicyUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]npUpdateTest{
		"change name": {
			old: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []extensions.NetworkPolicyIngressRule{},
				},
			},
			update: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []extensions.NetworkPolicyIngressRule{},
				},
			},
		},
		"change spec": {
			old: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []extensions.NetworkPolicyIngressRule{},
				},
			},
			update: extensions.NetworkPolicy{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: extensions.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []extensions.NetworkPolicyIngressRule{},
				},
			},
		},
	}

	for testName, errorCase := range errorCases {
		if errs := ValidateNetworkPolicyUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestIsValidSysctlPattern(t *testing.T) {
	valid := []string{
		"a.b.c.d",
		"a",
		"a_b",
		"a-b",
		"abc",
		"abc.def",
		"*",
		"a.*",
		"*",
		"abc*",
		"a.abc*",
		"a.b.*",
	}
	invalid := []string{
		"",
		"ä",
		"a_",
		"_",
		"_a",
		"_a._b",
		"__",
		"-",
		".",
		"a.",
		".a",
		"a.b.",
		"a*.b",
		"a*b",
		"*a",
		"Abc",
		func(n int) string {
			x := make([]byte, n)
			for i := range x {
				x[i] = byte('a')
			}
			return string(x)
		}(256),
	}
	for _, s := range valid {
		if !IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be a valid sysctl pattern", s)
		}
	}
	for _, s := range invalid {
		if IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be an invalid sysctl pattern", s)
		}
	}
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
