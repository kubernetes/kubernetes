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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// StorageCompressionConfiguration chooses which resources are compressed as
// they're written to etcd, ahead of any encryption at rest. Entries are checked
// in order and the first match wins, so a selector already covered by an earlier
// one is rejected. Wildcards never match secrets; name secrets to compress them.
type StorageCompressionConfiguration struct {
	metav1.TypeMeta `json:""`

	// resources is the ordered list of resources and how to store each. An empty
	// or missing list compresses nothing.
	// +optional
	Resources []CompressionResourceConfiguration `json:"resources,omitempty"`
}

// CompressionResourceConfiguration applies one algorithm and threshold to a list
// of resources.
type CompressionResourceConfiguration struct {
	// resources lists resources as <resource> or <resource>.<group>. '*.<group>'
	// selects a group, '*.' the core group and '*.*' every group. No wildcard
	// selects secrets.
	Resources []string `json:"resources"`

	// algorithm is Deflate or None. None keeps resources out of a later wildcard.
	Algorithm CompressionAlgorithm `json:"algorithm"`

	// minSize is the smallest serialized value that is compressed, as a quantity
	// of bytes such as 512 or 4Ki. It defaults to 1Ki. It must be a whole number
	// of bytes, at least 64 and less than 1472Ki, the largest value that's ever
	// compressed.
	// +optional
	MinSize *resource.Quantity `json:"minSize,omitempty"`
}

// CompressionAlgorithm names how a resource's values are written.
type CompressionAlgorithm string

const (
	CompressionAlgorithmDeflate CompressionAlgorithm = "Deflate"
	CompressionAlgorithmNone    CompressionAlgorithm = "None"
)
