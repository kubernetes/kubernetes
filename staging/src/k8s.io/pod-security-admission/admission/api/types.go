/*
Copyright 2021 The Kubernetes Authors.

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

package api

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type PodSecurityConfiguration struct {
	metav1.TypeMeta
	Defaults   PodSecurityDefaults
	Exemptions PodSecurityExemptions
}

type PodSecurityDefaults struct {
	Enforce        string
	EnforceVersion string
	Audit          string
	AuditVersion   string
	Warn           string
	WarnVersion    string
}

type PodSecurityExemptions struct {
	Usernames      []string
	Namespaces     []string
	RuntimeClasses []string
}
