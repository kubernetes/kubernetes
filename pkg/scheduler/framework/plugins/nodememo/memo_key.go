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

package nodememo

import (
	"strings"

	"k8s.io/apimachinery/pkg/labels"
)

// WriteSelector fingerprints one labels.Selector into a memo key. It lives here because two plugins
// build keys out of selectors and they have to agree on how, and because getting it wrong is not a
// cache miss: a key that conflates two selectors hands one shape's per node contributions to another,
// and those contributions are what Filter and Score read.
//
// String() on its own is not enough. metav1.LabelSelectorAsSelector turns a *missing* labelSelector
// into labels.Nothing(), which matches no pod at all, and an *empty* one into labels.Everything(),
// which matches every pod - and both render as the empty string. So do these two
//
//	podAntiAffinity:
//	  requiredDuringSchedulingIgnoredDuringExecution:
//	  - topologyKey: zone          # no labelSelector -> Nothing
//	  - topologyKey: zone
//	    labelSelector: {}            # empty -> Everything
//
// and the same pair of YAML shapes in a topologySpreadConstraint. Sharing a key between them means
// one pod filters, scores or spreads on the other's counts: for a required anti-affinity term that is
// admitting a pod onto a node it must not run on, or leaving one Pending behind counts that are all
// zero. MatchesNothing is the upstream way to tell them apart; a real selector is wrapped in brackets
// so that its rendering cannot run into a neighboring field's separator.
func WriteSelector(b *strings.Builder, sel labels.Selector) {
	switch {
	case sel == nil:
		b.WriteString("<nil>")
	case labels.MatchesNothing(sel):
		b.WriteString("<nothing>")
	default:
		b.WriteByte('<')
		b.WriteString(sel.String())
		b.WriteByte('>')
	}
}
