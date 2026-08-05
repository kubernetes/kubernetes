/*
Copyright The Kubernetes Authors.

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

// Package types holds the shared input types generated into ../registered and
// ../external and referenced by ../consumer.
package types

type T1 struct {
	TypeMeta int

	// +k8s:validateFalse="field T1.T2"
	T2 T2 `json:"t2"`

	// +k8s:eachVal=+k8s:validateFalse="field T1.List[*]"
	List []T2 `json:"list"`
}

type T2 struct {
	// +k8s:validateFalse="field T2.S"
	S string `json:"s"`
}

// T3 has no TypeMeta, so only external's validation-gen=* selects it.
type T3 struct {
	// +k8s:validateFalse="field T3.S"
	S string `json:"s"`
}
