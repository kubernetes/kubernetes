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
	"sort"
	"strings"

	"github.com/coreos/etcd/auth/authpb"
	"github.com/coreos/etcd/mvcc/backend"
)

func isSubset(a, b *rangePerm) bool {
	// return true if a is a subset of b
	return 0 <= strings.Compare(a.begin, b.begin) && strings.Compare(a.end, b.end) <= 0
}

// removeSubsetRangePerms removes any rangePerms that are subsets of other rangePerms.
func removeSubsetRangePerms(perms []*rangePerm) []*rangePerm {
	// TODO(mitake): currently it is O(n^2), we need a better algorithm
	newp := make([]*rangePerm, 0)

	for i := range perms {
		subset := false

		for j := range perms {
			if i != j && isSubset(perms[i], perms[j]) {
				subset = true
				break
			}
		}

		if subset {
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
		for next+1 < len(perms) && perms[next].end >= perms[next+1].begin {
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
			if len(perm.RangeEnd) == 0 {
				continue
			}
			rp := &rangePerm{begin: string(perm.Key), end: string(perm.RangeEnd)}

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

func checkCachedPerm(cachedPerms *unifiedRangePermissions, userName string, key, rangeEnd string, write, read bool) bool {
	var perms []*rangePerm

	if write {
		perms = cachedPerms.writePerms
	} else {
		perms = cachedPerms.readPerms
	}

	for _, perm := range perms {
		if strings.Compare(rangeEnd, "") != 0 {
			if strings.Compare(perm.begin, key) <= 0 && strings.Compare(rangeEnd, perm.end) <= 0 {
				return true
			}
		} else {
			if strings.Compare(perm.begin, key) <= 0 && strings.Compare(key, perm.end) <= 0 {
				return true
			}
		}
	}

	return false
}

func (as *authStore) isRangeOpPermitted(tx backend.BatchTx, userName string, key, rangeEnd string, write, read bool) bool {
	// assumption: tx is Lock()ed
	_, ok := as.rangePermCache[userName]
	if ok {
		return checkCachedPerm(as.rangePermCache[userName], userName, key, rangeEnd, write, read)
	}

	perms := getMergedPerms(tx, userName)
	if perms == nil {
		plog.Errorf("failed to create a unified permission of user %s", userName)
		return false
	}
	as.rangePermCache[userName] = perms

	return checkCachedPerm(as.rangePermCache[userName], userName, key, rangeEnd, write, read)

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
	begin, end string
}

type RangePermSliceByBegin []*rangePerm

func (slice RangePermSliceByBegin) Len() int {
	return len(slice)
}

func (slice RangePermSliceByBegin) Less(i, j int) bool {
	if slice[i].begin == slice[j].begin {
		return slice[i].end < slice[j].end
	}
	return slice[i].begin < slice[j].begin
}

func (slice RangePermSliceByBegin) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}
