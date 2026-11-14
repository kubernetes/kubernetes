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

package v1beta2

import (
	"fmt"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	if err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("StatefulSet"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "metadata.namespace", "status.successful":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported for appsv1beta2.StatefulSet: %s", label)
			}
		}); err != nil {
		return err
	}

	return nil
}

func Convert_autoscaling_ScaleStatus_To_v1beta2_ScaleStatus(in *autoscaling.ScaleStatus, out *appsv1beta2.ScaleStatus, s conversion.Scope) error {
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

func Convert_v1beta2_ScaleStatus_To_autoscaling_ScaleStatus(in *appsv1beta2.ScaleStatus, out *autoscaling.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	// Normally when 2 fields map to the same internal value we favor the old field, since
	// old clients can't be expected to know about new fields but clients that know about the
	// new field can be expected to know about the old field (though that's not quite true, due
	// to kubectl apply). However, these fields are readonly, so any non-nil value should work.
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
