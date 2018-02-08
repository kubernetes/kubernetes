/*
Copyright 2016 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ServiceAccountGeneratorV1 supports stable generation of a service account
type ServiceAccountGeneratorV1 struct {
	// Name of service account
	Name string
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &ServiceAccountGeneratorV1{}

// StructuredGenerate outputs a service account object using the configured fields
func (g *ServiceAccountGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := g.validate(); err != nil {
		return nil, err
	}
	serviceAccount := &v1.ServiceAccount{}
	serviceAccount.Name = g.Name
	return serviceAccount, nil
}

// validate validates required fields are set to support structured generation
func (g *ServiceAccountGeneratorV1) validate() error {
	if len(g.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	return nil
}
