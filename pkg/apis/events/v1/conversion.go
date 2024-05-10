/*
Copyright 2020 The Kubernetes Authors.

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

	v1 "k8s.io/api/events/v1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	k8s_api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

func init() {
	localSchemeBuilder.Register(addFieldLabelConversionFuncs)
}

func addFieldLabelConversionFuncs(scheme *runtime.Scheme) error {
	if err := AddFieldLabelConversionsForEvent(scheme); err != nil {
		return err
	}
	return nil
}

func Convert_v1_Event_To_core_Event(in *v1.Event, out *k8s_api.Event, s conversion.Scope) error {
	if err := autoConvert_v1_Event_To_core_Event(in, out, s); err != nil {
		return err
	}
	if err := k8s_api_v1.Convert_v1_ObjectReference_To_core_ObjectReference(&in.Regarding, &out.InvolvedObject, s); err != nil {
		return err
	}
	if err := k8s_api_v1.Convert_v1_EventSource_To_core_EventSource(&in.DeprecatedSource, &out.Source, s); err != nil {
		return err
	}
	out.Message = in.Note
	out.FirstTimestamp = in.DeprecatedFirstTimestamp
	out.LastTimestamp = in.DeprecatedLastTimestamp
	out.Count = in.DeprecatedCount
	return nil
}

func Convert_core_Event_To_v1_Event(in *k8s_api.Event, out *v1.Event, s conversion.Scope) error {
	if err := autoConvert_core_Event_To_v1_Event(in, out, s); err != nil {
		return err
	}
	if err := k8s_api_v1.Convert_core_ObjectReference_To_v1_ObjectReference(&in.InvolvedObject, &out.Regarding, s); err != nil {
		return err
	}
	if err := k8s_api_v1.Convert_core_EventSource_To_v1_EventSource(&in.Source, &out.DeprecatedSource, s); err != nil {
		return err
	}
	out.Note = in.Message
	out.DeprecatedFirstTimestamp = in.FirstTimestamp
	out.DeprecatedLastTimestamp = in.LastTimestamp
	out.DeprecatedCount = in.Count
	return nil
}

func AddFieldLabelConversionsForEvent(scheme *runtime.Scheme) error {
	mapping := map[string]string{
		"reason":                    "reason",
		"regarding.kind":            "involvedObject.kind",            // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.namespace":       "involvedObject.namespace",       // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.name":            "involvedObject.name",            // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.uid":             "involvedObject.uid",             // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.apiVersion":      "involvedObject.apiVersion",      // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.resourceVersion": "involvedObject.resourceVersion", // map events.k8s.io field to fieldset returned by ToSelectableFields
		"regarding.fieldPath":       "involvedObject.fieldPath",       // map events.k8s.io field to fieldset returned by ToSelectableFields
		"reportingController":       "reportingComponent",             // map events.k8s.io field to fieldset returned by ToSelectableFields
		"source":                    "source",                         // TODO: Do we still need to support selection by source? There's no source field any more.
		"type":                      "type",
		"metadata.namespace":        "metadata.namespace",
		"metadata.name":             "metadata.name",
	}
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Event"),
		func(label, value string) (string, string, error) {
			mappedLabel, ok := mapping[label]
			if !ok {
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
			return mappedLabel, value, nil
		},
	)
}
