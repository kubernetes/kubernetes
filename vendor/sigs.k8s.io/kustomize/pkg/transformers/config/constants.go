/*
Copyright 2018 The Kubernetes Authors.

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

package config

// Annotation is to mark a field as annotations.
// "x-kubernetes-annotation": ""
const Annotation = "x-kubernetes-annotation"

// LabelSelector is to mark a field as LabelSelector
// "x-kubernetes-label-selector": ""
const LabelSelector = "x-kubernetes-label-selector"

// Identity is to mark a field as Identity
// "x-kubernetes-identity": ""
const Identity = "x-kubernetes-identity"

// Version marks the type version of an object ref field
// "x-kubernetes-object-ref-api-version": <apiVersion name>
const Version = "x-kubernetes-object-ref-api-version"

// Kind marks the type name of an object ref field
// "x-kubernetes-object-ref-kind": <kind name>
const Kind = "x-kubernetes-object-ref-kind"

// NameKey marks the field key that refers to an object of an object ref field
// "x-kubernetes-object-ref-name-key": "name"
// default is "name"
const NameKey = "x-kubernetes-object-ref-name-key"
