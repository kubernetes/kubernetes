// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgtree

import (
	"sort"
	"strings"

	"github.com/armon/go-radix"
)

// IgnoredRuleset comprises a set of rules for ignoring import paths. It can
// manage both literal and prefix-wildcard matches.
type IgnoredRuleset struct {
	t *radix.Tree
}

// NewIgnoredRuleset processes a set of strings into an IgnoredRuleset. Strings
// that end in "*" are treated as wildcards, where any import path with a
// matching prefix will be ignored. IgnoredRulesets are immutable once created.
//
// Duplicate and redundant (i.e. a literal path that has a prefix of a wildcard
// path) declarations are discarded. Consequently, it is possible that the
// returned IgnoredRuleset may have a smaller Len() than the input slice.
func NewIgnoredRuleset(ig []string) *IgnoredRuleset {
	if len(ig) == 0 {
		return &IgnoredRuleset{}
	}

	ir := &IgnoredRuleset{
		t: radix.New(),
	}

	// Sort the list of all the ignores in order to ensure that wildcard
	// precedence is recorded correctly in the trie.
	sort.Strings(ig)
	for _, i := range ig {
		// Skip global ignore and empty string.
		if i == "*" || i == "" {
			continue
		}

		_, wildi, has := ir.t.LongestPrefix(i)
		// We may not always have a value here, but if we do, then it's a bool.
		wild, _ := wildi.(bool)
		// Check if it's a wildcard ignore.
		if strings.HasSuffix(i, "*") {
			// Check if it is ineffectual.
			if has && wild {
				// Skip ineffectual wildcard ignore.
				continue
			}
			// Create the ignore prefix and insert in the radix tree.
			ir.t.Insert(i[:len(i)-1], true)
		} else if !has || !wild {
			ir.t.Insert(i, false)
		}
	}

	if ir.t.Len() == 0 {
		ir.t = nil
	}

	return ir
}

// IsIgnored indicates whether the provided path should be ignored, according to
// the ruleset.
func (ir *IgnoredRuleset) IsIgnored(path string) bool {
	if path == "" || ir == nil || ir.t == nil {
		return false
	}

	prefix, wildi, has := ir.t.LongestPrefix(path)
	return has && (wildi.(bool) || path == prefix)
}

// Len indicates the number of rules in the ruleset.
func (ir *IgnoredRuleset) Len() int {
	if ir == nil || ir.t == nil {
		return 0
	}

	return ir.t.Len()
}

// ToSlice converts the contents of the IgnoredRuleset to a string slice.
//
// This operation is symmetrically dual to NewIgnoredRuleset.
func (ir *IgnoredRuleset) ToSlice() []string {
	irlen := ir.Len()
	if irlen == 0 {
		return nil
	}

	items := make([]string, 0, irlen)
	ir.t.Walk(func(s string, v interface{}) bool {
		if s != "" {
			if v.(bool) {
				items = append(items, s+"*")
			} else {
				items = append(items, s)
			}
		}
		return false
	})

	return items
}
