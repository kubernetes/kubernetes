/*
Copyright 2017 The Kubernetes Authors.

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

package appsv1beta2

import (
	"fmt"

	v1beta2 "k8s.io/api/apps/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	scheme "k8s.io/client-go/scale/scheme"
)

func Convert_scheme_ScaleStatus_To_v1beta2_ScaleStatus(in *scheme.ScaleStatus, out *v1beta2.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.Selector = nil
	out.TargetSelector = ""
	if in.Selector != nil {
		if in.Selector.MatchExpressions == nil || len(in.Selector.MatchExpressions) == 0 {
			out.Selector = in.Selector.MatchLabels
		}

		selector, err := metav1.LabelSelectorAsSelector(in.Selector)
		if err != nil {
			return fmt.Errorf("invalid label selector: %v", err)
		}
		out.TargetSelector = selector.String()
	}

	return nil
}

func Convert_v1beta2_ScaleStatus_To_scheme_ScaleStatus(in *v1beta2.ScaleStatus, out *scheme.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	// Normally when 2 fields map to the same internal value we favor the old field, since
	// old clients can't be expected to know about new fields but clients that know about the
	// new field can be expected to know about the old field (though that's not quite true, due
	// to kubectl apply). However, these fields are readonly, so any non-nil value should work.
	if in.TargetSelector != "" {
		labelSelector, err := metav1.ParseToLabelSelector(in.TargetSelector)
		if err != nil {
			out.Selector = nil
			return fmt.Errorf("failed to parse target selector: %v", err)
		}
		out.Selector = labelSelector
	} else if in.Selector != nil {
		out.Selector = new(metav1.LabelSelector)
		selector := make(map[string]string)
		for key, val := range in.Selector {
			selector[key] = val
		}
		out.Selector.MatchLabels = selector
	} else {
		out.Selector = nil
	}

	return nil
}
