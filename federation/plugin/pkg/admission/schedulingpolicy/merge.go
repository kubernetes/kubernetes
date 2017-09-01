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

package schedulingpolicy

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// mergeAnnotations updates obj so that the provided annotations supersede the
// existing annotations.
func mergeAnnotations(obj runtime.Object, annotations map[string]string) error {
	if len(annotations) == 0 {
		return nil
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}

	orig := accessor.GetAnnotations()
	for k := range orig {
		if _, ok := annotations[k]; !ok {
			annotations[k] = orig[k]
		}
	}

	accessor.SetAnnotations(annotations)
	return nil
}
