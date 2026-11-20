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

package testing

import (
	"k8s.io/apimachinery/pkg/runtime"
	openapiutil "k8s.io/kube-openapi/pkg/util"
)

// runtime.OpenAPIModelNamer must stay identical to kube-openapi's, which generated model names implement.
var (
	_ runtime.OpenAPIModelNamer     = openapiutil.OpenAPIModelNamer(nil)
	_ openapiutil.OpenAPIModelNamer = runtime.OpenAPIModelNamer(nil)
)
