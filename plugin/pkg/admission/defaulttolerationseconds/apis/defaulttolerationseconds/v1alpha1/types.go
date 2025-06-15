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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Configuration provides configuration for the DefaultTolerationSeconds admission controller.
type Configuration struct {
	metav1.TypeMeta `json:",inline"`

	DefaultTolerationSecondsConfig DefaultTolerationSecondsConfig `json:"defaultTolerationSecondsConfig"`
}

type DefaultTolerationSecondsConfig struct {
	// NotReadyTolerationSeconds indicates the tolerationSeconds of the toleration for notReady:NoExecute
	// that is added by default to every pod that does not already have such a toleration.
	NotReadyTolerationSeconds *int64 `json:"notReadyTolerationSeconds"`
	// UnreachableTolerationSeconds indicates the tolerationSeconds of the toleration for unreachable:NoExecute
	// that is added by default to every pod that does not already have such a toleration.
	UnreachableTolerationSeconds *int64 `json:"unreachableTolerationSeconds"`
}
