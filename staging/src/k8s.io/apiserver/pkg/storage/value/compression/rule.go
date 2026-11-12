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

package compression

import (
	"strconv"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Rule tells WithCompression how to write one resource's values; any reader
// accepts what any Rule writes. The zero Rule compresses nothing, and a new
// field mustn't change existing Rules. WithCompression uses a Rule as given.
type Rule struct {
	// Algorithm is the compression new writes use. AlgorithmNone, or any value
	// this package doesn't define, writes them uncompressed.
	Algorithm Algorithm
	// MinSize is the smallest value, in bytes, that the writer compresses. Zero or
	// less sets no floor; a configuration file that omits minSize gets 1Ki.
	MinSize int
}

// ResourceRules gives each resource the Rule its store writes with.
// *compressionconfig.CompressionConfiguration implements it.
type ResourceRules interface {
	// RuleFor returns gr's Rule, or the zero Rule if gr isn't compressed.
	RuleFor(gr schema.GroupResource) Rule
}

// Algorithm says how a Rule compresses new writes.
type Algorithm int

const (
	// AlgorithmNone writes values uncompressed.
	AlgorithmNone Algorithm = iota
	// AlgorithmDeflate writes values in DEFLATE frames.
	AlgorithmDeflate
)

// String returns the name a configuration file uses, so a failed test
// comparison prints Deflate rather than 1.
func (a Algorithm) String() string {
	switch a {
	case AlgorithmNone:
		return "None"
	case AlgorithmDeflate:
		return "Deflate"
	default:
		return "Algorithm(" + strconv.Itoa(int(a)) + ")"
	}
}
