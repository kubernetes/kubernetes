/*
Copyright 2017 The Kubernetes Authors.

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
	v1beta1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	// SchemeBuilder registers the types and defaulting functions of
	// k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1 and the conversions to the
	// internal version. It is separate from that package's builder so that
	// importing this package doesn't change what the other one registers.
	SchemeBuilder      = runtime.NewSchemeBuilder(v1beta1.AddToScheme)
	localSchemeBuilder = &SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)
