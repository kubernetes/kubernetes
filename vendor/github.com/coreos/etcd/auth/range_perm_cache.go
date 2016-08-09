// Copyright 2016 The etcd Authors
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

package auth

import (
	"bytes"
	"sort"

	"github.com/coreos/etcd/auth/authpb"
	"github.com/coreos/etcd/mvcc/backend"
)

// isSubset returns true if a is a subset of b
func isSubset(a, b *rangePerm) bool {
	switch {
	case len(a.end) == 0 && len(b.end) == 0:
		// a, b are both keys
		return bytes.Equal(a.begin, b.begin)
	case len(b.end) == 0:
		// b is a key, a is a range
		return false
	case len(a.end) == 0:
		return 0 <= bytes.Compare(a.begin, b.begin) && bytes.Compare(a.begin, b.end) <= 0
	default:
		return 0 <= bytes.Compare(a.begin, b.begin) && bytes.Compare(a.end, b.end) <= 0
	}
}

func isRangeEqual(a, b *rangePerm) bool {
	return bytes.Equal(a.begin, b.begin) && bytes.Equal(a.end, b.end)
}

// removeSubsetRangePerms removes any rangePerms that are subsets of other rangePerms.
// If there are equal ranges, removeSubsetRangePerms only keeps one of them.
func removeSubsetRangePerms(perms []*rangePerm) []*rangePerm {
	// TODO(mitake): currently it is O(n^2), we need a better algorithm
	newp := make([]*rangePerm, 0)

	for i := range perms {
		skip := false

		for j := range perms {
			if i == j {
				continue
			}

			if isRangeEqual(perms[i], perms[j]) {
				// if ranges are equal, we only keep the first range.
				if i > j {
					skip = true
					break
				}
			} else if isSubset(perms[i], perms[j]) {
				// if a range is a strict subset of the other one, we skip the subset.
				skip = true
				break
			}
		}

		if skip {
			continue
		}

		newp = append(newp, perms[i])
	}

	return newp
}

// mergeRangePerms merges adjacent rangePerms.
func mergeRangePerms(perms []*rangePerm) []*rangePerm {
	merged := make([]*rangePerm, 0)
	perms = removeSubsetRangePerms(perms)
	sort.Sort(RangePermSliceByBegin(perms))

	i := 0
	for i < len(perms) {
		begin, next := i, i
		for next+1 < len(perms) && bytes.Compare(perms[next].end, perms[next+1].begin) != -1 {
			next++
		}

		merged = append(merged, &rangePerm{begin: perms[begin].begin, end: perms[next].end})

		i = next + 1
	}

	return merged
}

func getMergedPerms(tx backend.BatchTx, userName string) *unifiedRangePermissions {
	user := getUser(tx, userName)
	if user == nil {
		plog.Errorf("invalid user name %s", userName)
		return nil
	}

	var readPerms, writePerms []*rangePerm

	for _, roleName := range user.Roles {
		role := getRole(tx, roleName)
		if role == nil {
			continue
		}

		for _, perm := range role.KeyPermission {
			rp := &rangePerm{begin: perm.Key, end: perm.RangeEnd}

			switch perm.PermType {
			case authpb.READWRITE:
				readPerms = append(readPerms, rp)
				writePerms = append(writePerms, rp)

			case authpb.READ:
				readPerms = append(readPerms, rp)

			case authpb.WRITE:
				writePerms = append(writePerms, rp)
			}
		}
	}

	return &unifiedRangePermissions{
		readPerms:  mergeRangePerms(readPerms),
		writePerms: mergeRangePerms(writePerms),
	}
}

func checkKeyPerm(cachedPerms *unifiedRangePermissions, key, rangeEnd []byte, permtyp authpb.Permission_Type) bool {
	var tocheck []*rangePerm

	switch permtyp {
	case authpb.READ:
		tocheck = cachedPerms.readPerms
	case authpb.WRITE:
		tocheck = cachedPerms.writePerms
	default:
		plog.Panicf("unknown auth type: %v", permtyp)
	}

	requiredPerm := &rangePerm{begin: key, end: rangeEnd}

	for _, perm := range tocheck {
		if isSubset(requiredPerm, perm) {
			return true
		}
	}

	return false
}

func (as *authStore) isRangeOpPermitted(tx backend.BatchTx, userName string, key, rangeEnd []byte, permtyp authpb.Permission_Type) bool {
	// assumption: tx is Lock()ed
	_, ok := as.rangePermCache[userName]
	if !ok {
		perms := getMergedPerms(tx, userName)
		if perms == nil {
			plog.Errorf("failed to create a unified permission of user %s", userName)
			return false
		}
		as.rangePermCache[userName] = perms
	}

	return checkKeyPerm(as.rangePermCache[userName], key, rangeEnd, permtyp)
}

func (as *authStore) clearCachedPerm() {
	as.rangePermCache = make(map[string]*unifiedRangePermissions)
}

func (as *authStore) invalidateCachedPerm(userName string) {
	delete(as.rangePermCache, userName)
}

type unifiedRangePermissions struct {
	// readPerms[i] and readPerms[j] (i != j) don't overlap
	readPerms []*rangePerm
	// writePerms[i] and writePerms[j] (i != j) don't overlap, too
	writePerms []*rangePerm
}

type rangePerm struct {
	begin, end []byte
}

type RangePermSliceByBegin []*rangePerm

func (slice RangePermSliceByBegin) Len() int {
	return len(slice)
}

func (slice RangePermSliceByBegin) Less(i, j int) bool {
	switch bytes.Compare(slice[i].begin, slice[j].begin) {
	case 0: // begin(i) == begin(j)
		return bytes.Compare(slice[i].end, slice[j].end) == -1

	case -1: // begin(i) < begin(j)
		return true

	default:
		return false
	}
}

func (slice RangePermSliceByBegin) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}
