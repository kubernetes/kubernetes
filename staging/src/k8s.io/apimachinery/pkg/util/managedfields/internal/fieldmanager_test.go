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

package internal_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var fakeTypeConverter = func() internal.TypeConverter {
	data, err := os.ReadFile(filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 8),
		"api", "openapi-spec", "swagger.json"))
	if err != nil {
		panic(err)
	}
	convertedDefs := map[string]*spec.Schema{}
	spec := spec.Swagger{}
	if err := json.Unmarshal(data, &spec); err != nil {
		panic(err)
	}

	for k, v := range spec.Definitions {
		vCopy := v
		convertedDefs[k] = &vCopy
	}

	typeConverter, err := internal.NewTypeConverter(convertedDefs, false)
	if err != nil {
		panic(err)
	}
	return typeConverter
}()
