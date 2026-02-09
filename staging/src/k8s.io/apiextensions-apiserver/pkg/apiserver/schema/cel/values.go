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

package cel

import (
	"github.com/google/cel-go/common/types/ref"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	celopenapi "k8s.io/apiserver/pkg/cel/common"
)

// UnstructuredToVal converts a Kubernetes unstructured data element to a CEL Val.
// The root schema of custom resource schema is expected contain type meta and object meta schemas.
// If Embedded resources do not contain type meta and object meta schemas, they will be added automatically.
func UnstructuredToVal(unstructured interface{}, schema *structuralschema.Structural) ref.Val {
	return celopenapi.UnstructuredToVal(unstructured, &model.Structural{Structural: schema})
}
