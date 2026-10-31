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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

// SecretsGroupResource is never selected by a wildcard. Compressing a Secret
// exposes the compressibility of its plaintext to anyone who can read etcd, so
// it has to be named explicitly.
var SecretsGroupResource = schema.GroupResource{Group: "", Resource: "secrets"}

var anyGroupAnyResource = schema.GroupResource{Group: "*", Resource: "*"}

// Policy decides, per resource, what the write path does, and so which stored
// values a read reports stale. It never decides whether a read succeeds. A nil
// Policy selects nothing.
type Policy struct {
	rules map[schema.GroupResource]rule
}

type rule struct {
	enabled bool
	minSize int
}

// NewPolicy expects a configuration that passed ValidateStorageCompressionConfiguration,
// which rejects any selector that an earlier one masks. In such a file the most
// specific match for a resource is also the first one listed, so the rules can live
// in a map. A nil configuration gives a nil Policy.
func NewPolicy(c *apiserver.StorageCompressionConfiguration) *Policy {
	if c == nil {
		return nil
	}
	p := &Policy{rules: map[schema.GroupResource]rule{}}
	for _, entry := range c.Resources {
		r := rule{
			enabled: entry.Algorithm == apiserver.CompressionAlgorithmDeflate,
			minSize: DefaultMinSize,
		}
		if entry.MinSize != nil {
			r.minSize = int(entry.MinSize.Value())
		}
		for _, selector := range entry.Resources {
			p.rules[schema.ParseGroupResource(selector)] = r
		}
	}
	return p
}

func (p *Policy) ruleFor(gr schema.GroupResource) (rule, bool) {
	if p == nil {
		return rule{}, false
	}
	for _, selector := range CoveringSelectors(gr) {
		if r, ok := p.rules[selector]; ok {
			return r, true
		}
	}
	return rule{}, false
}

func (p *Policy) EnabledFor(gr schema.GroupResource) bool {
	r, ok := p.ruleFor(gr)
	return ok && r.enabled
}

// MinSizeFor takes minSize from the same entry that sets gr's algorithm, so a
// wildcard can carry one threshold while a named resource listed before it
// carries another.
func (p *Policy) MinSizeFor(gr schema.GroupResource) int {
	if r, ok := p.ruleFor(gr); ok {
		return r.minSize
	}
	return DefaultMinSize
}

// CoveringSelectors returns the selectors that could apply to gr, most specific
// first: gr itself, then '*.<group>', then '*.*'. For secrets it returns only
// secrets, because no wildcard selects them. Validation uses it to find masked
// selectors and Policy uses it to look up resources, so the two always agree on
// which entry decides a resource.
func CoveringSelectors(gr schema.GroupResource) []schema.GroupResource {
	if gr == SecretsGroupResource {
		return []schema.GroupResource{gr}
	}
	return []schema.GroupResource{gr, {Group: gr.Group, Resource: "*"}, anyGroupAnyResource}
}
