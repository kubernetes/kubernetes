/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/conversion"
)

func Convert_apiextensions_JSON_To_v1beta1_JSON(in *apiextensions.JSON, out *JSON, s conversion.Scope) error {
	out.Object = in.Object
	return nil
}

func Convert_v1beta1_JSON_To_apiextensions_JSON(in *JSON, out *apiextensions.JSON, s conversion.Scope) error {
	out.Object = in.Object
	return nil
}
