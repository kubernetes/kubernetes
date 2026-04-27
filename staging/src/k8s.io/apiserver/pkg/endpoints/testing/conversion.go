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

package testing

import (
	"net/url"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

func convertUrlValuesToSimpleGetOptions(in *url.Values, out *SimpleGetOptions, s conversion.Scope) error {
	// Skip TypeMeta field.

	if values, ok := map[string][]string(*in)["param1"]; ok && len(values) > 0 {
		if err := runtime.Convert_Slice_string_To_string(&values, &out.Param1, s); err != nil {
			return err
		}
	} else {
		out.Param1 = ""
	}
	if values, ok := map[string][]string(*in)["param2"]; ok && len(values) > 0 {
		if err := runtime.Convert_Slice_string_To_string(&values, &out.Param2, s); err != nil {
			return err
		}
	} else {
		out.Param2 = ""
	}
	if values, ok := map[string][]string(*in)["atAPath"]; ok && len(values) > 0 {
		if err := runtime.Convert_Slice_string_To_string(&values, &out.Path, s); err != nil {
			return err
		}
	} else {
		out.Path = ""
	}
	return nil
}

func RegisterConversions(s *runtime.Scheme) error {
	if err := s.AddConversionFunc((*url.Values)(nil), (*SimpleGetOptions)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertUrlValuesToSimpleGetOptions(a.(*url.Values), b.(*SimpleGetOptions), scope)
	}); err != nil {
		return err
	}
	return nil
}
