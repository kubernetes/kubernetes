/*
Copyright 2023 The Kubernetes Authors.

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

package v1beta3

import (
	"k8s.io/api/flowcontrol/v1beta3"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

func Convert_v1beta3_PriorityLevelConfiguration_To_flowcontrol_PriorityLevelConfiguration(in *v1beta3.PriorityLevelConfiguration, out *flowcontrol.PriorityLevelConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1beta3_PriorityLevelConfiguration_To_flowcontrol_PriorityLevelConfiguration(in, out, nil); err != nil {
		return err
	}

	// during v1beta3 -> internal conversion:
	//  - remove the roundtrip annotation for the 'NominalConcurrencyShares' field
	//  - make sure we don't mutate the source (v1beta3) object's annotations
	annotations, copied := dropPriorityLevelConcurrencyShareDefaultAnnotation(out.ObjectMeta.Annotations)
	if copied {
		out.ObjectMeta.Annotations = annotations
	}
	return nil
}

func Convert_flowcontrol_PriorityLevelConfiguration_To_v1beta3_PriorityLevelConfiguration(in *flowcontrol.PriorityLevelConfiguration, out *v1beta3.PriorityLevelConfiguration, s conversion.Scope) error {
	if err := autoConvert_flowcontrol_PriorityLevelConfiguration_To_v1beta3_PriorityLevelConfiguration(in, out, nil); err != nil {
		return err
	}

	// during internal -> v1beta3 conversion:
	//  - add the roundtrip annotation for the 'NominalConcurrencyShares' field,
	//      IIF the 'NominalConcurrencyShares' field has a value of zero.
	//  - make sure we don't mutate the source (internal) object's annotations
	if limited := in.Spec.Limited; limited != nil && limited.NominalConcurrencyShares == 0 {
		annotations, copied := addPriorityLevelConcurrencyShareDefaultAnnotation(out.ObjectMeta.Annotations)
		if copied {
			out.ObjectMeta.Annotations = annotations
		}
	}

	return nil
}

func dropPriorityLevelConcurrencyShareDefaultAnnotation(in map[string]string) (map[string]string, bool) {
	if _, ok := in[v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey]; !ok {
		return in, false
	}

	out := copyStringMap(in)
	delete(out, v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey)
	return out, true
}

func addPriorityLevelConcurrencyShareDefaultAnnotation(in map[string]string) (map[string]string, bool) {
	if _, ok := in[v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey]; ok {
		return in, false
	}

	out := copyStringMap(in)
	out[v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey] = ""
	return out, true
}

// copyStringMap returns a copy of the input map.
// If input is nil, an empty map is returned.
func copyStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
