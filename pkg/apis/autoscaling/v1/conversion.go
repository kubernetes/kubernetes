/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package v1

import (
	"reflect"

	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func addConversionFuncs(scheme *runtime.Scheme) {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_extensions_SubresourceReference_To_v1_CrossVersionObjectReference,
		Convert_v1_CrossVersionObjectReference_To_extensions_SubresourceReference,
		Convert_extensions_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec,
		Convert_v1_HorizontalPodAutoscalerSpec_To_extensions_HorizontalPodAutoscalerSpec,
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}

func Convert_extensions_SubresourceReference_To_v1_CrossVersionObjectReference(in *extensions.SubresourceReference, out *CrossVersionObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*extensions.SubresourceReference))(in)
	}
	out.Kind = in.Kind
	out.Name = in.Name
	out.APIVersion = in.APIVersion
	return nil
}

func Convert_v1_CrossVersionObjectReference_To_extensions_SubresourceReference(in *CrossVersionObjectReference, out *extensions.SubresourceReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*CrossVersionObjectReference))(in)
	}
	out.Kind = in.Kind
	out.Name = in.Name
	out.APIVersion = in.APIVersion
	out.Subresource = "scale"
	return nil
}

func Convert_extensions_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec(in *extensions.HorizontalPodAutoscalerSpec, out *HorizontalPodAutoscalerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*extensions.HorizontalPodAutoscalerSpec))(in)
	}
	if err := Convert_extensions_SubresourceReference_To_v1_CrossVersionObjectReference(&in.ScaleRef, &out.ScaleTargetRef, s); err != nil {
		return err
	}
	if in.MinReplicas != nil {
		out.MinReplicas = new(int32)
		*out.MinReplicas = int32(*in.MinReplicas)
	} else {
		out.MinReplicas = nil
	}
	out.MaxReplicas = int32(in.MaxReplicas)
	if in.CPUUtilization != nil {
		out.TargetCPUUtilizationPercentage = new(int32)
		*out.TargetCPUUtilizationPercentage = int32(in.CPUUtilization.TargetPercentage)
	}
	return nil
}

func Convert_v1_HorizontalPodAutoscalerSpec_To_extensions_HorizontalPodAutoscalerSpec(in *HorizontalPodAutoscalerSpec, out *extensions.HorizontalPodAutoscalerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*HorizontalPodAutoscalerSpec))(in)
	}
	if err := Convert_v1_CrossVersionObjectReference_To_extensions_SubresourceReference(&in.ScaleTargetRef, &out.ScaleRef, s); err != nil {
		return err
	}
	if in.MinReplicas != nil {
		out.MinReplicas = new(int)
		*out.MinReplicas = int(*in.MinReplicas)
	} else {
		out.MinReplicas = nil
	}
	out.MaxReplicas = int(in.MaxReplicas)
	if in.TargetCPUUtilizationPercentage != nil {
		out.CPUUtilization = &extensions.CPUTargetUtilization{TargetPercentage: int(*in.TargetCPUUtilizationPercentage)}
	}
	return nil
}
