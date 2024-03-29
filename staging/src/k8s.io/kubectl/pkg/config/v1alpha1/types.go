/*
Copyright 2024 The Kubernetes Authors.

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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// PreferencesCommandOverrideFlag stores the name and the specified default
// value of the flag.
type PreferencesCommandOverrideFlag struct {
	// Flags name without dashes as prefix.
	Name string `json:"name"`

	// In a string format of a default value. It will be parsed
	// by kubectl to the compatible value of the flag.
	Default string `json:"default"`
}

// PreferencesCommandOverride stores the commands and their associated flag's
// default values.
type PreferencesCommandOverride struct {
	// Command refers to a command whose flag's default value is changed.
	Command string `json:"command"`
	// Flags is a list of flags storing different default values.
	Flags []PreferencesCommandOverrideFlag `json:"flags"`
}

// PreferencesSpec stores the overrides
type PreferencesSpec struct {
	// Overrides is used to change the defaults values of flags of commands.
	// This is especially useful, when user doesn't want to explicitly
	// set flags each time.
	Overrides []PreferencesCommandOverride `json:"overrides"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Preferences stores elements of KubeRC configuration file
type Preferences struct {
	metav1.TypeMeta `json:",inline"`

	// Spec stores the overrides and the other preferences
	Spec PreferencesSpec `json:"spec,omitempty"`
}
