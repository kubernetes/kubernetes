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

package scheme

import (
	scheme "k8s.io/apiextensions-api/pkg/client/clientset/clientset/scheme"
)

var (
	// Deprecated: Use k8s.io/apiextensions-api/pkg/client/clientset/clientset/scheme.AddToScheme instead.
	AddToScheme = scheme.AddToScheme
	// Deprecated: Use k8s.io/apiextensions-api/pkg/client/clientset/clientset/scheme.Codecs instead.
	Codecs = scheme.Codecs
	// Deprecated: Use k8s.io/apiextensions-api/pkg/client/clientset/clientset/scheme.ParameterCodec instead.
	ParameterCodec = scheme.ParameterCodec
	// Deprecated: Use k8s.io/apiextensions-api/pkg/client/clientset/clientset/scheme.Scheme instead.
	Scheme = scheme.Scheme
)
