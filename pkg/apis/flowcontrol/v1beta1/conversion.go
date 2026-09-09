/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/api/flowcontrol/v1beta1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

// LimitedPriorityLevelConfiguration.AssuredConcurrencyShares has been
// renamed to NominalConcurrencyShares in v1beta3.
func Convert_v1beta1_LimitedPriorityLevelConfiguration_To_flowcontrol_LimitedPriorityLevelConfiguration(in *v1beta1.LimitedPriorityLevelConfiguration, out *flowcontrol.LimitedPriorityLevelConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1beta1_LimitedPriorityLevelConfiguration_To_flowcontrol_LimitedPriorityLevelConfiguration(in, out, nil); err != nil {
		return err
	}

	out.NominalConcurrencyShares = in.AssuredConcurrencyShares
	return nil
}

// LimitedPriorityLevelConfiguration.AssuredConcurrencyShares has been
// renamed to NominalConcurrencyShares in v1beta3.
func Convert_flowcontrol_LimitedPriorityLevelConfiguration_To_v1beta1_LimitedPriorityLevelConfiguration(in *flowcontrol.LimitedPriorityLevelConfiguration, out *v1beta1.LimitedPriorityLevelConfiguration, s conversion.Scope) error {
	if err := autoConvert_flowcontrol_LimitedPriorityLevelConfiguration_To_v1beta1_LimitedPriorityLevelConfiguration(in, out, nil); err != nil {
		return err
	}

	out.AssuredConcurrencyShares = in.NominalConcurrencyShares
	return nil
}
