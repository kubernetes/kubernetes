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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateReplicationControllerStatus(t *testing.T) {
	tests := []struct {
		name string

		replicas             int32
		fullyLabeledReplicas int32
		readyReplicas        int32
		availableReplicas    int32
		observedGeneration   int64

		expectedErr bool
	}{
		{
			name:                 "valid status",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        2,
			availableReplicas:    1,
			observedGeneration:   2,
			expectedErr:          false,
		},
		{
			name:                 "invalid replicas",
			replicas:             -1,
			fullyLabeledReplicas: 3,
			readyReplicas:        2,
			availableReplicas:    1,
			observedGeneration:   2,
			expectedErr:          true,
		},
		{
			name:                 "invalid fullyLabeledReplicas",
			replicas:             3,
			fullyLabeledReplicas: -1,
			readyReplicas:        2,
			availableReplicas:    1,
			observedGeneration:   2,
			expectedErr:          true,
		},
		{
			name:                 "invalid readyReplicas",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        -1,
			availableReplicas:    1,
			observedGeneration:   2,
			expectedErr:          true,
		},
		{
			name:                 "invalid availableReplicas",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        3,
			availableReplicas:    -1,
			observedGeneration:   2,
			expectedErr:          true,
		},
		{
			name:                 "invalid observedGeneration",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        3,
			availableReplicas:    3,
			observedGeneration:   -1,
			expectedErr:          true,
		},
		{
			name:                 "fullyLabeledReplicas greater than replicas",
			replicas:             3,
			fullyLabeledReplicas: 4,
			readyReplicas:        3,
			availableReplicas:    3,
			observedGeneration:   1,
			expectedErr:          true,
		},
		{
			name:                 "readyReplicas greater than replicas",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        4,
			availableReplicas:    3,
			observedGeneration:   1,
			expectedErr:          true,
		},
		{
			name:                 "availableReplicas greater than replicas",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        3,
			availableReplicas:    4,
			observedGeneration:   1,
			expectedErr:          true,
		},
		{
			name:                 "availableReplicas greater than readyReplicas",
			replicas:             3,
			fullyLabeledReplicas: 3,
			readyReplicas:        2,
			availableReplicas:    3,
			observedGeneration:   1,
			expectedErr:          true,
		},
	}

	for _, test := range tests {
		status := core.ReplicationControllerStatus{
			Replicas:             test.replicas,
			FullyLabeledReplicas: test.fullyLabeledReplicas,
			ReadyReplicas:        test.readyReplicas,
			AvailableReplicas:    test.availableReplicas,
			ObservedGeneration:   test.observedGeneration,
		}

		if hasErr := len(ValidateReplicationControllerStatus(status, field.NewPath("status"))) > 0; hasErr != test.expectedErr {
			t.Errorf("%s: expected error: %t, got error: %t", test.name, test.expectedErr, hasErr)
		}
	}
}

func TestValidateReplicationControllerStatusUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
	}
	type rcUpdateTest struct {
		old    core.ReplicationController
		update core.ReplicationController
	}
	successCases := []rcUpdateTest{
		{
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: core.ReplicationControllerStatus{
					Replicas: 2,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 3,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: core.ReplicationControllerStatus{
					Replicas: 4,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicationControllerStatusUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"negative replicas": {
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: core.ReplicationControllerStatus{
					Replicas: 3,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: core.ReplicationControllerStatus{
					Replicas: -3,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicationControllerStatusUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}

}

func TestValidateReplicationControllerUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
	}
	readWriteVolumePodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				Volumes:       []core.Volume{{Name: "gcepd", VolumeSource: core.VolumeSource{GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ObjectMeta: metav1.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	type rcUpdateTest struct {
		old    core.ReplicationController
		update core.ReplicationController
	}
	successCases := []rcUpdateTest{
		{
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 3,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
		{
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 1,
					Selector: validSelector,
					Template: &readWriteVolumePodTemplate.Template,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicationControllerUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"more than one read/write": {
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid selector": {
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 2,
					Selector: invalidSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
		"invalid pod": {
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &invalidPodTemplate.Template,
				},
			},
		},
		"negative replicas": {
			old: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: core.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: core.ReplicationControllerSpec{
					Replicas: -1,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicationControllerUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateReplicationController(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
	}
	readWriteVolumePodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: core.PodSpec{
				Volumes:       []core.Volume{{Name: "gcepd", VolumeSource: core.VolumeSource{GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := core.PodTemplate{
		Template: core.PodTemplateSpec{
			Spec: core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ObjectMeta: metav1.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	successCases := []core.ReplicationController{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "abc-123", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "abc-123", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Replicas: 1,
				Selector: validSelector,
				Template: &readWriteVolumePodTemplate.Template,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateReplicationController(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]core.ReplicationController{
		"zero-length ID": {
			ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc-123"},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"empty selector": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Template: &validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Selector: map[string]string{"foo": "bar"},
				Template: &validPodTemplate.Template,
			},
		},
		"invalid manifest": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
			},
		},
		"read-write persistent disk with > 1 pod": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc"},
			Spec: core.ReplicationControllerSpec{
				Replicas: 2,
				Selector: validSelector,
				Template: &readWriteVolumePodTemplate.Template,
			},
		},
		"negative_replicas": {
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: core.ReplicationControllerSpec{
				Replicas: -1,
				Selector: validSelector,
			},
		},
		"invalid_label": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc-123",
				Namespace: metav1.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid_label 2": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc-123",
				Namespace: metav1.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: core.ReplicationControllerSpec{
				Template: &invalidPodTemplate.Template,
			},
		},
		"invalid_annotation": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc-123",
				Namespace: metav1.NamespaceDefault,
				Annotations: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc-123",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &core.PodTemplateSpec{
					Spec: core.PodSpec{
						RestartPolicy: core.RestartPolicyOnFailure,
						DNSPolicy:     core.DNSClusterFirst,
						Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					},
					ObjectMeta: metav1.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc-123",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: core.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &core.PodTemplateSpec{
					Spec: core.PodSpec{
						RestartPolicy: core.RestartPolicyNever,
						DNSPolicy:     core.DNSClusterFirst,
						Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					},
					ObjectMeta: metav1.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateReplicationController(&v)
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
