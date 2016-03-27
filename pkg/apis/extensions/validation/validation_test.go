/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestValidateHorizontalPodAutoscaler(t *testing.T) {
	successCases := []extensions.HorizontalPodAutoscaler{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.HorizontalPodAutoscalerSpec{
				ScaleRef: extensions.SubresourceReference{
					Kind:        "ReplicationController",
					Name:        "myrc",
					Subresource: "scale",
				},
				MinReplicas:    newInt(1),
				MaxReplicas:    5,
				CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.HorizontalPodAutoscalerSpec{
				ScaleRef: extensions.SubresourceReference{
					Kind:        "ReplicationController",
					Name:        "myrc",
					Subresource: "scale",
				},
				MinReplicas: newInt(1),
				MaxReplicas: 5,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"name\":\"qps\",\"value\":\"20\"}]}",
				},
			},
			Spec: extensions.HorizontalPodAutoscalerSpec{
				ScaleRef: extensions.SubresourceReference{
					Kind:        "ReplicationController",
					Name:        "myrc",
					Subresource: "scale",
				},
				MinReplicas: newInt(1),
				MaxReplicas: 5,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateHorizontalPodAutoscaler(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		horizontalPodAutoscaler extensions.HorizontalPodAutoscaler
		msg                     string
	}{
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Name: "myrc", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.kind: Required",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "..", Name: "myrc", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.kind: Invalid",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "ReplicationController", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.name: Required",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "ReplicationController", Name: "..", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.name: Invalid",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: ""},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Required",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: ".."},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Invalid",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef:       extensions.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: "randomsubresource"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Unsupported",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas: newInt(-1),
					MaxReplicas: 5,
				},
			},
			msg: "must be greater than 0",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas: newInt(7),
					MaxReplicas: 5,
				},
			},
			msg: "must be greater than or equal to `minReplicas`",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: -70},
				},
			},
			msg: "must be greater than 0",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "broken",
					},
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Kind:        "ReplicationController",
						Name:        "myrc",
						Subresource: "scale",
					},
					MinReplicas: newInt(1),
					MaxReplicas: 5,
				},
			},
			msg: "failed to parse custom metrics target annotation",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{}",
					},
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Kind:        "ReplicationController",
						Name:        "myrc",
						Subresource: "scale",
					},
					MinReplicas: newInt(1),
					MaxReplicas: 5,
				},
			},
			msg: "custom metrics target must not be empty",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"value\":\"20\"}]}",
					},
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Kind:        "ReplicationController",
						Name:        "myrc",
						Subresource: "scale",
					},
					MinReplicas: newInt(1),
					MaxReplicas: 5,
				},
			},
			msg: "missing custom metric target name",
		},
		{
			horizontalPodAutoscaler: extensions.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"name\":\"qps\",\"value\":\"0\"}]}",
					},
				},
				Spec: extensions.HorizontalPodAutoscalerSpec{
					ScaleRef: extensions.SubresourceReference{
						Kind:        "ReplicationController",
						Name:        "myrc",
						Subresource: "scale",
					},
					MinReplicas: newInt(1),
					MaxReplicas: 5,
				},
			},
			msg: "custom metric target value must be greater than 0",
		},
	}

	for _, c := range errorCases {
		errs := ValidateHorizontalPodAutoscaler(&c.horizontalPodAutoscaler)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %q, expected: %q", errs[0], c.msg)
		}
	}
}

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
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     1,
					DesiredNumberScheduled: 3,
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
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 1,
					NumberMisscheduled:     2,
					DesiredNumberScheduled: 3,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: -1,
					NumberMisscheduled:     -1,
					DesiredNumberScheduled: -3,
				},
			},
		},
	}

	for testName, errorCase := range errorCases {
		if errs := ValidateDaemonSetStatusUpdate(&errorCase.old, &errorCase.update); len(errs) == 0 {
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
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector2},
					Template: validPodTemplateAbc2.Template,
				},
			},
		},
		{
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		"invalid selector": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: invalidSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
		},
		"invalid pod": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: invalidPodTemplate.Template,
				},
			},
		},
		"change container image": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateDef.Template,
				},
			},
		},
		"read-write volume": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid update strategy": {
			old: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
					Template: validPodTemplateAbc.Template,
				},
			},
			update: extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.DaemonSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: invalidSelector},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: extensions.DaemonSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
				Selector: &unversioned.LabelSelector{},
				Template: validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				Template: validPodTemplate.Template,
			},
		},
		"invalid template": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.DaemonSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
				Template: validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.DaemonSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validSelector},
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
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{
					"name": "abc",
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
	invalidSelectorDeployment.Spec.Selector = &unversioned.LabelSelector{
		MatchLabels: map[string]string{
			"name": "def",
		},
	}
	errorCases["`selector` does not match template `labels`"] = invalidSelectorDeployment

	// RestartPolicy should be always.
	invalidRestartPolicyDeployment := validDeployment()
	invalidRestartPolicyDeployment.Spec.Template.Spec.RestartPolicy = api.RestartPolicyNever
	errorCases["Unsupported value: \"Never\""] = invalidRestartPolicyDeployment

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
	errorCases["must be an integer or percentage"] = invalidMaxSurgeDeployment

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

func TestValidateJob(t *testing.T) {
	validManualSelector := &unversioned.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validGeneratedSelector := &unversioned.LabelSelector{
		MatchLabels: map[string]string{"controller-uid": "1a2b3c", "job-name": "myjob"},
	}
	validPodTemplateSpecForManual := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validManualSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	validPodTemplateSpecForGenerated := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validGeneratedSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	successCases := map[string]extensions.Job{
		"manual selector": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template:       validPodTemplateSpecForManual,
			},
		},
		"generated selector": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Selector:       validGeneratedSelector,
				ManualSelector: newBool(false),
				Template:       validPodTemplateSpecForGenerated,
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateJob(&v); len(errs) != 0 {
			t.Errorf("expected success for %s: %v", k, errs)
		}
	}
	negative := -1
	negative64 := int64(-1)
	errorCases := map[string]extensions.Job{
		"spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Parallelism:    &negative,
				ManualSelector: newBool(true),
				Template:       validPodTemplateSpecForGenerated,
			},
		},
		"spec.completions:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Completions:    &negative,
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template:       validPodTemplateSpecForGenerated,
			},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				ActiveDeadlineSeconds: &negative64,
				Selector:              validManualSelector,
				ManualSelector:        newBool(true),
				Template:              validPodTemplateSpecForGenerated,
			},
		},
		"spec.selector:Required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Template: validPodTemplateSpecForGenerated,
			},
		},
		"spec.template.metadata.labels: Invalid value: {\"y\":\"z\"}: `selector` does not match template `labels`": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{"y": "z"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
		"spec.template.metadata.labels: Invalid value: {\"controller-uid\":\"4d5e6f\"}: `selector` does not match template `labels`": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{"controller-uid": "4d5e6f"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
		"spec.template.spec.restartPolicy: Unsupported value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: extensions.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: validManualSelector.MatchLabels,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateJob(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %v, expected: %s", err, k)
			}
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
	noSecretName := newValid()
	noSecretName.Spec.TLS = []extensions.IngressTLS{{SecretName: ""}}

	errorCases := map[string]extensions.Ingress{
		"spec.backend.serviceName: Required value":        servicelessBackend,
		"spec.backend.serviceName: Invalid value":         invalidNameBackend,
		"spec.backend.servicePort: Invalid value":         noPortBackend,
		"spec.rules[0].host: Invalid value":               badHost,
		"spec.rules[0].http.paths: Required value":        noPaths,
		"spec.rules[0].http.paths[0].path: Invalid value": noForwardSlashPath,
		"spec.tls[0].secretName: Required value":          noSecretName,
	}
	errorCases[badPathErr] = badRegexPath
	errorCases[badHostIPErr] = badHostIP

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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 3,
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
		},
		{
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 1,
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid selector": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &unversioned.LabelSelector{MatchLabels: invalidLabels},
					Template: validPodTemplate.Template,
				},
			},
		},
		"invalid pod": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: 2,
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: invalidPodTemplate.Template,
				},
			},
		},
		"negative replicas": {
			old: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
					Template: validPodTemplate.Template,
				},
			},
			update: extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: extensions.ReplicaSetSpec{
					Replicas: -1,
					Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Replicas: 1,
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: extensions.ReplicaSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				Template: validPodTemplate.Template,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
			},
		},
		"read-write persistent disk with > 1 pod": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Spec: extensions.ReplicaSetSpec{
				Replicas: 2,
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
				Template: readWriteVolumePodTemplate.Template,
			},
		},
		"negative_replicas": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: extensions.ReplicaSetSpec{
				Replicas: -1,
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
				Template: validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.ReplicaSetSpec{
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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
				Selector: &unversioned.LabelSelector{MatchLabels: validLabels},
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

func newInt(val int) *int {
	p := new(int)
	*p = val
	return p
}

func TestValidatePodSecurityPolicy(t *testing.T) {
	validSCC := func() *extensions.PodSecurityPolicy {
		return &extensions.PodSecurityPolicy{
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: extensions.PodSecurityPolicySpec{
				SELinux: extensions.SELinuxStrategyOptions{
					Rule: extensions.SELinuxStrategyRunAsAny,
				},
				RunAsUser: extensions.RunAsUserStrategyOptions{
					Rule: extensions.RunAsUserStrategyRunAsAny,
				},
			},
		}
	}

	noUserOptions := validSCC()
	noUserOptions.Spec.RunAsUser.Rule = ""

	noSELinuxOptions := validSCC()
	noSELinuxOptions.Spec.SELinux.Rule = ""

	invalidUserStratRule := validSCC()
	invalidUserStratRule.Spec.RunAsUser.Rule = "invalid"

	invalidSELinuxStratRule := validSCC()
	invalidSELinuxStratRule.Spec.SELinux.Rule = "invalid"

	missingObjectMetaName := validSCC()
	missingObjectMetaName.ObjectMeta.Name = ""

	invalidRangeMinGreaterThanMax := validSCC()
	invalidRangeMinGreaterThanMax.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{Min: 2, Max: 1},
	}

	invalidRangeNegativeMin := validSCC()
	invalidRangeNegativeMin.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{Min: -1, Max: 10},
	}

	invalidRangeNegativeMax := validSCC()
	invalidRangeNegativeMax.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{Min: 1, Max: -10},
	}

	errorCases := map[string]struct {
		scc         *extensions.PodSecurityPolicy
		errorDetail string
	}{
		"no user options": {
			scc:         noUserOptions,
			errorDetail: "supported values: MustRunAs, MustRunAsNonRoot, RunAsAny",
		},
		"no selinux options": {
			scc:         noSELinuxOptions,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"invalid user strategy rule": {
			scc:         invalidUserStratRule,
			errorDetail: "supported values: MustRunAs, MustRunAsNonRoot, RunAsAny",
		},
		"invalid selinux strategy rule": {
			scc:         invalidSELinuxStratRule,
			errorDetail: "supported values: MustRunAs, RunAsAny",
		},
		"missing object meta name": {
			scc:         missingObjectMetaName,
			errorDetail: "name or generateName is required",
		},
		"invalid range min greater than max": {
			scc:         invalidRangeMinGreaterThanMax,
			errorDetail: "min cannot be greater than max",
		},
		"invalid range negative min": {
			scc:         invalidRangeNegativeMin,
			errorDetail: "min cannot be negative",
		},
		"invalid range negative max": {
			scc:         invalidRangeNegativeMax,
			errorDetail: "max cannot be negative",
		},
	}

	for k, v := range errorCases {
		if errs := ValidatePodSecurityPolicy(v.scc); len(errs) == 0 || errs[0].Detail != v.errorDetail {
			t.Errorf("Expected error with detail %s for %s, got %v", v.errorDetail, k, errs[0].Detail)
		}
	}

	mustRunAs := validSCC()
	mustRunAs.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Ranges = []extensions.IDRange{
		{
			Min: 1,
			Max: 1,
		},
	}
	mustRunAs.Spec.SELinux.Rule = extensions.SELinuxStrategyMustRunAs

	runAsNonRoot := validSCC()
	runAsNonRoot.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAsNonRoot

	successCases := map[string]struct {
		scc *extensions.PodSecurityPolicy
	}{
		"must run as": {
			scc: mustRunAs,
		},
		"run as any": {
			scc: validSCC(),
		},
		"run as non-root (user only)": {
			scc: runAsNonRoot,
		},
	}

	for k, v := range successCases {
		if errs := ValidatePodSecurityPolicy(v.scc); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
