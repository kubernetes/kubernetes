/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta1

import (
	"fmt"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/core"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	if err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("StatefulSet"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "metadata.namespace", "status.successful":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported for appsv1beta1.StatefulSet: %s", label)
			}
		}); err != nil {
		return err
	}

	return nil
}

func Convert_autoscaling_ScaleStatus_To_v1beta1_ScaleStatus(in *autoscaling.ScaleStatus, out *appsv1beta1.ScaleStatus, s conversion.Scope) error {
	out.Replicas = int32(in.Replicas)
	out.TargetSelector = in.Selector

	out.Selector = nil
	selector, err := metav1.ParseToLabelSelector(in.Selector)
	if err != nil {
		return fmt.Errorf("failed to parse selector: %v", err)
	}
	if len(selector.MatchExpressions) == 0 {
		out.Selector = selector.MatchLabels
	}

	return nil
}

func Convert_v1beta1_ScaleStatus_To_autoscaling_ScaleStatus(in *appsv1beta1.ScaleStatus, out *autoscaling.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	if in.TargetSelector != "" {
		out.Selector = in.TargetSelector
	} else if in.Selector != nil {
		set := labels.Set{}
		for key, val := range in.Selector {
			set[key] = val
		}
		out.Selector = labels.SelectorFromSet(set).String()
	} else {
		out.Selector = ""
	}
	return nil
}

// Convert_v1beta1_StatefulSetSpec_To_apps_StatefulSetSpec augments auto-conversion to preserve < 1.17 behavior
// setting apiVersion/kind in nested persistent volume claim objects.
func Convert_v1beta1_StatefulSetSpec_To_apps_StatefulSetSpec(in *appsv1beta1.StatefulSetSpec, out *apps.StatefulSetSpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_StatefulSetSpec_To_apps_StatefulSetSpec(in, out, s); err != nil {
		return err
	}
	// set APIVersion/Kind to behave the same as reflective conversion < 1.17.
	// see https://issue.k8s.io/87583
	if out.VolumeClaimTemplates != nil {
		// copy so we don't modify the input
		templatesCopy := make([]core.PersistentVolumeClaim, len(out.VolumeClaimTemplates))
		copy(templatesCopy, out.VolumeClaimTemplates)
		out.VolumeClaimTemplates = templatesCopy
		for i := range out.VolumeClaimTemplates {
			out.VolumeClaimTemplates[i].APIVersion = ""
			out.VolumeClaimTemplates[i].Kind = ""
		}
	}
	return nil
}

// Convert_apps_StatefulSetSpec_To_v1beta1_StatefulSetSpec augments auto-conversion to preserve < 1.17 behavior
// setting apiVersion/kind in nested persistent volume claim objects.
func Convert_apps_StatefulSetSpec_To_v1beta1_StatefulSetSpec(in *apps.StatefulSetSpec, out *appsv1beta1.StatefulSetSpec, s conversion.Scope) error {
	if err := autoConvert_apps_StatefulSetSpec_To_v1beta1_StatefulSetSpec(in, out, s); err != nil {
		return err
	}
	// set APIVersion/Kind to behave the same as reflective conversion < 1.17.
	// see https://issue.k8s.io/87583
	if out.VolumeClaimTemplates != nil {
		// copy so we don't modify the input
		templatesCopy := make([]corev1.PersistentVolumeClaim, len(out.VolumeClaimTemplates))
		copy(templatesCopy, out.VolumeClaimTemplates)
		out.VolumeClaimTemplates = templatesCopy
		for i := range out.VolumeClaimTemplates {
			out.VolumeClaimTemplates[i].APIVersion = "v1"
			out.VolumeClaimTemplates[i].Kind = "PersistentVolumeClaim"
		}
	}
	return nil
}
