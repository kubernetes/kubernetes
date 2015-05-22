/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func addConversionFuncs() {
	// Add non-generated conversion functions
	err := api.Scheme.AddConversionFuncs(
		convert_api_StatusDetails_To_v1_StatusDetails,
		convert_v1_StatusDetails_To_api_StatusDetails,
		convert_v1_ReplicationControllerSpec_To_api_ReplicationControllerSpec,
		convert_api_ReplicationControllerSpec_To_v1_ReplicationControllerSpec,
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}

	// Add field conversion funcs.
	err = api.Scheme.AddFieldLabelConversionFunc("v1", "Pod",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"metadata.namespace",
				"status.phase",
				"spec.host":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1", "Node",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name":
				return label, value, nil
			case "spec.unschedulable":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1", "ReplicationController",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"status.replicas":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1", "Event",
		func(label, value string) (string, string, error) {
			switch label {
			case "involvedObject.kind",
				"involvedObject.namespace",
				"involvedObject.name",
				"involvedObject.uid",
				"involvedObject.apiVersion",
				"involvedObject.resourceVersion",
				"involvedObject.fieldPath",
				"reason",
				"source":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1", "Namespace",
		func(label, value string) (string, string, error) {
			switch label {
			case "status.phase":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}

func convert_v1_StatusDetails_To_api_StatusDetails(in *StatusDetails, out *api.StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusDetails))(in)
	}
	out.ID = in.Name
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]api.StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_v1_StatusCause_To_api_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_api_StatusDetails_To_v1_StatusDetails(in *api.StatusDetails, out *StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.StatusDetails))(in)
	}
	out.Name = in.ID
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_api_StatusCause_To_v1_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_v1_StatusCause_To_api_StatusCause(in *StatusCause, out *api.StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusCause))(in)
	}
	out.Type = api.CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_api_StatusCause_To_v1_StatusCause(in *api.StatusCause, out *StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.StatusCause))(in)
	}
	out.Type = CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_api_ReplicationControllerSpec_To_v1_ReplicationControllerSpec(in *api.ReplicationControllerSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ReplicationControllerSpec))(in)
	}
	out.Replicas = new(int)
	*out.Replicas = in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	if in.TemplateRef != nil {
		out.TemplateRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1_ObjectReference(in.TemplateRef, out.TemplateRef, s); err != nil {
			return err
		}
	} else {
		out.TemplateRef = nil
	}
	if in.Template != nil {
		out.Template = new(PodTemplateSpec)
		if err := convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func convert_v1_ReplicationControllerSpec_To_api_ReplicationControllerSpec(in *ReplicationControllerSpec, out *api.ReplicationControllerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerSpec))(in)
	}
	out.Replicas = *in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	if in.TemplateRef != nil {
		out.TemplateRef = new(api.ObjectReference)
		if err := convert_v1_ObjectReference_To_api_ObjectReference(in.TemplateRef, out.TemplateRef, s); err != nil {
			return err
		}
	} else {
		out.TemplateRef = nil
	}
	if in.Template != nil {
		out.Template = new(api.PodTemplateSpec)
		if err := convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}
