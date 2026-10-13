/*
Copyright The Kubernetes Authors.

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
	v1beta1 "k8s.io/apiextensions/pkg/client/listers/apiextensions/v1beta1"
	"k8s.io/client-go/tools/cache"
)

//go:fix inline
type CustomResourceDefinitionLister = v1beta1.CustomResourceDefinitionLister

//go:fix inline
type CustomResourceDefinitionListerExpansion = v1beta1.CustomResourceDefinitionListerExpansion

//go:fix inline
func NewCustomResourceDefinitionLister(indexer cache.Indexer) v1beta1.CustomResourceDefinitionLister {
	return v1beta1.NewCustomResourceDefinitionLister(indexer)
}
