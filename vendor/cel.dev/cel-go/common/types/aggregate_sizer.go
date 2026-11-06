// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

// AggregateSizer calculates the recursive element size of values.
type AggregateSizer interface {
	// AggregateSize returns the size of the input value, if known.
	// Otherwise, a unit size of 1 is returned.
	AggregateSize(val any) uint32
}

// AggregateSizeVisitor interface for ref.Val implementations capable of returning
// their total recursive element count.
type AggregateSizeVisitor interface {
	// AggregateSize returns the total count of nested atomic elements (capped at math.MaxUint32).
	AggregateSize(sizer AggregateSizer) uint32
}

// Helper for computing aggregate sizes of traits.Foldable types.
type foldableAggregateSizer struct {
	sizer AggregateSizer
	total uint32
}

// FoldEntry implements the traits.FoldEntry interface method and counts the aggregate size
// keys and values.
func (f *foldableAggregateSizer) FoldEntry(k, v any) bool {
	f.total = safeAddUint32(f.total, f.sizer.AggregateSize(k))
	f.total = safeAddUint32(f.total, f.sizer.AggregateSize(v))
	return true
}
