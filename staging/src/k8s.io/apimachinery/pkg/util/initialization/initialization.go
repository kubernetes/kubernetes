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

package initialization

import (
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// IsInitialized returns if the initializers indicates means initialized.
func IsInitialized(initializers *metav1.Initializers) bool {
	if initializers == nil {
		return true
	}
	// Persisted objects will never be in this state. The initializer admission
	// plugin will override metadata.initializers to nil. If the initializer
	// admissio plugin is disabled, the generic registry always set
	// metadata.initializers to nil. However, this function
	// might be called before the object persisted, thus the check.
	if len(initializers.Pending) == 0 && initializers.Result == nil {
		return true
	}
	return false
}

// IsObjectInitialized returns if the object is initialized.
func IsObjectInitialized(obj runtime.Object) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	return IsInitialized(accessor.GetInitializers()), nil
}
