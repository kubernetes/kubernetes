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

package registrytest

import (
	"fmt"

	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
)

// ValidateStorageStrategies ensures any instances of the generic registry.Store in the given storage map
// have expected strategies defined.
func ValidateStorageStrategies(storageMap map[string]rest.Storage) []error {
	errs := []error{}

	for k, storage := range storageMap {
		switch t := storage.(type) {
		case registry.GenericStore:
			// At this point it appears all uses of the generic registry store should have a create, update, and
			// delete strategy set:
			if t.GetCreateStrategy() == nil {
				errs = append(errs, fmt.Errorf("store for type [%v] does not have a CreateStrategy", k))
			}
			if t.GetUpdateStrategy() == nil {
				errs = append(errs, fmt.Errorf("store for type [%v] does not have an UpdateStrategy", k))
			}
			if t.GetDeleteStrategy() == nil {
				errs = append(errs, fmt.Errorf("store for type [%v] does not have a DeleteStrategy", k))
			}
		}
	}

	return errs
}
