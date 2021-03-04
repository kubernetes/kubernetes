/*
Copyright 2015 The Kubernetes Authors.

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

package versioned

import (
	"fmt"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubectl/pkg/generate"
)

// HorizontalPodAutoscalerGeneratorV1 supports stable generation of a horizontal pod autoscaler.
type HorizontalPodAutoscalerGeneratorV1 struct {
	Name               string
	ScaleRefKind       string
	ScaleRefName       string
	ScaleRefAPIVersion string
	MinReplicas        int32
	MaxReplicas        int32
	CPUPercent         int32
}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ generate.StructuredGenerator = &HorizontalPodAutoscalerGeneratorV1{}

// StructuredGenerate outputs a horizontal pod autoscaler object using the configured fields.
func (s *HorizontalPodAutoscalerGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	scaler := autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name: s.Name,
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind:       s.ScaleRefKind,
				Name:       s.ScaleRefName,
				APIVersion: s.ScaleRefAPIVersion,
			},
			MaxReplicas: s.MaxReplicas,
		},
	}

	if s.MinReplicas > 0 {
		v := int32(s.MinReplicas)
		scaler.Spec.MinReplicas = &v
	}
	if s.CPUPercent >= 0 {
		c := int32(s.CPUPercent)
		scaler.Spec.TargetCPUUtilizationPercentage = &c
	}

	return &scaler, nil
}

// validate check if the caller has set the right fields.
func (s HorizontalPodAutoscalerGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if s.MaxReplicas < 1 {
		return fmt.Errorf("'max' is a required parameter and must be at least 1")
	}
	if s.MinReplicas > s.MaxReplicas {
		return fmt.Errorf("'max' must be greater than or equal to 'min'")
	}
	return nil
}
