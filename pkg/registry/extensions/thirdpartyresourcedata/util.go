/*
Copyright 2015 The Kubernetes Authors.

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

package thirdpartyresourcedata

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func ExtractGroupVersionKind(list *extensions.ThirdPartyResourceList) ([]schema.GroupVersion, []schema.GroupVersionKind, error) {
	gvs := []schema.GroupVersion{}
	gvks := []schema.GroupVersionKind{}
	for ix := range list.Items {
		rsrc := &list.Items[ix]
		gvk := schema.GroupVersionKind{Group: rsrc.Spec.Group, Version: rsrc.Spec.Version, Kind: rsrc.Spec.Kind}
		gvs = append(gvs, gvk.GroupVersion())
		gvks = append(gvks, gvk)
	}
	return gvs, gvks, nil
}
