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

package federatedtypes

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// FederatedType configures federation for a kubernetes type
type FederatedType struct {
	Kind              string
	ControllerName    string
	RequiredResources []schema.GroupVersionResource
	AdapterFactory    AdapterFactory
}

var typeRegistry = make(map[string]FederatedType)

// RegisterFederatedType ensures that configuration for the given kind will be returned by the FederatedTypes method.
func RegisterFederatedType(kind, controllerName string, requiredResources []schema.GroupVersionResource, factory AdapterFactory) {
	_, ok := typeRegistry[kind]
	if ok {
		// TODO Is panicking ok given that this is part of a type-registration mechanism
		panic(fmt.Sprintf("Federated type %q has already been registered", kind))
	}
	typeRegistry[kind] = FederatedType{
		Kind:              kind,
		ControllerName:    controllerName,
		RequiredResources: requiredResources,
		AdapterFactory:    factory,
	}
}

// FederatedTypes returns a mapping of kind (e.g. "secret") to the
// type information required to configure its federation.
func FederatedTypes() map[string]FederatedType {
	// TODO copy RequiredResources to avoid accidental mutation
	result := make(map[string]FederatedType)
	for key, value := range typeRegistry {
		result[key] = value
	}
	return result
}
