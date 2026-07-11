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

package sharding

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
)

// Selector represents a shard selector that can match objects based on
// hash ranges of their metadata fields. It follows the labels.Selector
// pattern from the Kubernetes API.
type Selector interface {
	// Matches returns true if the given object matches the shard selector.
	Matches(obj runtime.Object) (bool, error)

	// Empty returns true if the selector matches everything (no filtering).
	Empty() bool

	// String returns the wire-format string representation that can be
	// round-tripped through Parse.
	String() string

	// Requirements returns the list of shard range requirements.
	Requirements() []ShardRangeRequirement

	// DeepCopySelector returns a deep copy of the selector.
	DeepCopySelector() Selector
}

// Everything returns a selector that matches all objects.
func Everything() Selector {
	return &everythingSelector{}
}

type everythingSelector struct{}

func (s *everythingSelector) Matches(_ runtime.Object) (bool, error) { return true, nil }
func (s *everythingSelector) Empty() bool                            { return true }
func (s *everythingSelector) String() string                         { return "" }
func (s *everythingSelector) Requirements() []ShardRangeRequirement  { return nil }
func (s *everythingSelector) DeepCopySelector() Selector             { return &everythingSelector{} }

// shardSelector implements Selector with one or more shard range requirements.
type shardSelector struct {
	requirements []ShardRangeRequirement
}

func (s *shardSelector) Matches(obj runtime.Object) (bool, error) {
	if len(s.requirements) == 0 {
		return true, nil
	}
	// All requirements must share the same key so we resolve the field value
	// and compute the hash once. The parser enforces this, but we verify here
	// to guard against selectors constructed through other means.
	key := s.requirements[0].Key
	for _, req := range s.requirements[1:] {
		if req.Key != key {
			return false, fmt.Errorf("inconsistent shard keys: %q vs %q", key, req.Key)
		}
	}

	value, err := ResolveFieldValue(obj, key)
	if err != nil {
		return false, err
	}
	hash := "0x" + HashField(value)

	for _, req := range s.requirements {
		if !HexLess(hash, req.Start) && HexLess(hash, req.End) {
			return true, nil
		}
	}
	return false, nil
}

// HexLess compares two 0x-prefixed lowercase hex strings numerically.
// Both values must be normalized to 16 hex digits (e.g. "0x0000000000000000"),
// except for the special upper bound "0x10000000000000000" (2^64) which has 17.
func HexLess(a, b string) bool {
	if len(a) != len(b) {
		return len(a) < len(b)
	}
	return a < b
}

func (s *shardSelector) Empty() bool {
	return len(s.requirements) == 0
}

func (s *shardSelector) String() string {
	parts := make([]string, 0, len(s.requirements))
	for _, req := range s.requirements {
		parts = append(parts, fmt.Sprintf("shardRange(%s, '%s', '%s')", req.Key, req.Start, req.End))
	}
	return strings.Join(parts, " || ")
}

func (s *shardSelector) Requirements() []ShardRangeRequirement {
	result := make([]ShardRangeRequirement, len(s.requirements))
	copy(result, s.requirements)
	return result
}

func (s *shardSelector) DeepCopySelector() Selector {
	reqs := make([]ShardRangeRequirement, len(s.requirements))
	copy(reqs, s.requirements)
	return &shardSelector{requirements: reqs}
}

// NewSelector creates a Selector from the given requirements.
// All requirements must use the same Key; this is validated at match time.
// If no requirements are provided, returns Everything().
func NewSelector(reqs ...ShardRangeRequirement) Selector {
	if len(reqs) == 0 {
		return Everything()
	}
	return &shardSelector{
		requirements: reqs,
	}
}
