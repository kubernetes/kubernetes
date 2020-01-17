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

package metadata

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
)

// Interface allows a caller to get the metadata (in the form of PartialObjectMetadata objects)
// from any Kubernetes compatible resource API.
type Interface interface {
	Resource(resource schema.GroupVersionResource) Getter
}

// ResourceInterface contains the set of methods that may be invoked on objects by their metadata.
// Update is not supported by the server, but Patch can be used for the actions Update would handle.
type ResourceInterface interface {
	Delete(name string, options *metav1.DeleteOptions, subresources ...string) error
	DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(name string, options metav1.GetOptions, subresources ...string) (*metav1.PartialObjectMetadata, error)
	List(opts metav1.ListOptions) (*metav1.PartialObjectMetadataList, error)
	Watch(opts metav1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, options metav1.PatchOptions, subresources ...string) (*metav1.PartialObjectMetadata, error)
}

// Getter handles both namespaced and non-namespaced resource types consistently.
type Getter interface {
	Namespace(string) ResourceInterface
	ResourceInterface
}
