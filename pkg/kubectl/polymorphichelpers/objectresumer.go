/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func defaultObjectResumer(obj runtime.Object) ([]byte, error) {
	switch obj := obj.(type) {
	case *extensions.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(internalVersionJSONEncoder(), obj)
	default:
		return nil, fmt.Errorf("resuming is not supported")
	}
}
