/*
Copyright 2025 The Kubernetes Authors.

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

package v1alpha1

import (
	conversion "k8s.io/apimachinery/pkg/conversion"
	config "k8s.io/kubectl/pkg/config"
)

// v1alpha1 Preference does not have `CredentialPluginPolicy` or `CredentialPluginAllowlist` fields. They can be left blank, so the autoConvert functions will suffice.
func Convert_config_Preference_To_v1alpha1_Preference(in *config.Preference, out *Preference, s conversion.Scope) error {
	return autoConvert_config_Preference_To_v1alpha1_Preference(in, out, s)
}

func Convert_v1alpha1_Preference_To_config_Preference(in *Preference, out *config.Preference, s conversion.Scope) error {
	return autoConvert_v1alpha1_Preference_To_config_Preference(in, out, s)
}
