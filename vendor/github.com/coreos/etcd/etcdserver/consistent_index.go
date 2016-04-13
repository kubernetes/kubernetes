// Copyright 2015 CoreOS, Inc.
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

package etcdserver

// consistentIndex represents the offset of an entry in a consistent replica log.
// It implements the storage.ConsistentIndexGetter interface.
// It is always set to the offset of current entry before executing the entry,
// so ConsistentWatchableKV could get the consistent index from it.
type consistentIndex uint64

func (i *consistentIndex) setConsistentIndex(v uint64) { *i = consistentIndex(v) }

func (i *consistentIndex) ConsistentIndex() uint64 { return uint64(*i) }
