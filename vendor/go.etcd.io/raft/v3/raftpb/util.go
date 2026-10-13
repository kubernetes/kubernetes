// Copyright 2026 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package raftpb

// EnsureConfState ensures that cs and all of its pointer fields are non-nil.
// If cs is nil, a new ConfState is allocated. Any nil pointer field is set to
// point to its zero value. Returns the resulting cs.
func EnsureConfState(cs *ConfState) *ConfState {
	if cs == nil {
		cs = new(ConfState)
	}
	if cs.AutoLeave == nil {
		cs.AutoLeave = new(false)
	}
	return cs
}

// EnsureSnapshotMetadata ensures that m and all of its pointer fields are
// non-nil. If m is nil, a new SnapshotMetadata is allocated. Any nil pointer
// field is set to point to its zero value. Returns the resulting m.
func EnsureSnapshotMetadata(m *SnapshotMetadata) *SnapshotMetadata {
	if m == nil {
		m = new(SnapshotMetadata)
	}
	newConfState := EnsureConfState(m.ConfState)
	if m.ConfState == nil {
		// ConfState in a SnapshotMetadata is guaranteed to be non-nil, so in most cases
		// we don't need to update m.ConfState, which also avoids potential races.
		m.ConfState = newConfState
	}

	if m.Index == nil {
		m.Index = new(uint64)
	}
	if m.Term == nil {
		m.Term = new(uint64)
	}
	return m
}

// EnsureSnapshot ensures that s and all of its pointer fields are non-nil.
// If s is nil, a new Snapshot is allocated. Any nil pointer field is set to
// point to its zero value. Returns the resulting s.
func EnsureSnapshot(s *Snapshot) *Snapshot {
	if s == nil {
		s = new(Snapshot)
	}
	newMedata := EnsureSnapshotMetadata(s.Metadata)
	if s.Metadata == nil {
		// Metadata in a snapshot is guaranteed to be non-nil, so in most cases
		// we don't need to update s.Metadata, which also avoids potential races.
		s.Metadata = newMedata
	}
	return s
}
