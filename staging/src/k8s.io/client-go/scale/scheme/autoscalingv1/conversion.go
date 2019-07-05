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

package autoscalingv1

import (
	"fmt"

	v1 "k8s.io/api/autoscaling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	scheme "k8s.io/client-go/scale/scheme"
)

// addConversions registers conversions between the internal version
// of Scale and supported external versions of Scale.
func addConversionFuncs(scheme *runtime.Scheme) error {
	err := scheme.AddConversionFuncs(
		Convert_scheme_ScaleStatus_To_v1_ScaleStatus,
		Convert_v1_ScaleStatus_To_scheme_ScaleStatus,
	)
	if err != nil {
		return err
	}

	return nil
}

func Convert_scheme_ScaleStatus_To_v1_ScaleStatus(in *scheme.ScaleStatus, out *v1.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.Selector = ""
	if in.Selector != nil {
		selector, err := metav1.LabelSelectorAsSelector(in.Selector)
		if err != nil {
			return fmt.Errorf("invalid label selector: %v", err)
		}
		out.Selector = selector.String()
	}

	return nil
}

func Convert_v1_ScaleStatus_To_scheme_ScaleStatus(in *v1.ScaleStatus, out *scheme.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	if in.Selector != "" {
		labelSelector, err := metav1.ParseToLabelSelector(in.Selector)
		if err != nil {
			out.Selector = nil
			return fmt.Errorf("failed to parse target selector: %v", err)
		}
		out.Selector = labelSelector
	}

	return nil
}
