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
	"go.etcd.io/etcd/api/v3/authpb"
	"go.etcd.io/etcd/pkg/v3/adt"
	"go.etcd.io/etcd/server/v3/mvcc/backend"

	"go.uber.org/zap"
)

func getMergedPerms(lg *zap.Logger, tx backend.ReadTx, userName string) *unifiedRangePermissions {
	user := getUser(lg, tx, userName)
	if user == nil {
		return nil
	}

	readPerms := adt.NewIntervalTree()
	writePerms := adt.NewIntervalTree()

	for _, roleName := range user.Roles {
		role := getRole(lg, tx, roleName)
		if role == nil {
			continue
		}

		for _, perm := range role.KeyPermission {
			var ivl adt.Interval
			var rangeEnd []byte

			if len(perm.RangeEnd) != 1 || perm.RangeEnd[0] != 0 {
				rangeEnd = perm.RangeEnd
			}

			if len(perm.RangeEnd) != 0 {
				ivl = adt.NewBytesAffineInterval(perm.Key, rangeEnd)
			} else {
				ivl = adt.NewBytesAffinePoint(perm.Key)
			}

			switch perm.PermType {
			case authpb.READWRITE:
				readPerms.Insert(ivl, struct{}{})
				writePerms.Insert(ivl, struct{}{})

			case authpb.READ:
				readPerms.Insert(ivl, struct{}{})

			case authpb.WRITE:
				writePerms.Insert(ivl, struct{}{})
			}
		}
	}

	return &unifiedRangePermissions{
		readPerms:  readPerms,
		writePerms: writePerms,
	}
}

func checkKeyInterval(
	lg *zap.Logger,
	cachedPerms *unifiedRangePermissions,
	key, rangeEnd []byte,
	permtyp authpb.Permission_Type) bool {
	if len(rangeEnd) == 1 && rangeEnd[0] == 0 {
		rangeEnd = nil
	}

	ivl := adt.NewBytesAffineInterval(key, rangeEnd)
	switch permtyp {
	case authpb.READ:
		return cachedPerms.readPerms.Contains(ivl)
	case authpb.WRITE:
		return cachedPerms.writePerms.Contains(ivl)
	default:
		lg.Panic("unknown auth type", zap.String("auth-type", permtyp.String()))
	}
	return false
}

func checkKeyPoint(lg *zap.Logger, cachedPerms *unifiedRangePermissions, key []byte, permtyp authpb.Permission_Type) bool {
	pt := adt.NewBytesAffinePoint(key)
	switch permtyp {
	case authpb.READ:
		return cachedPerms.readPerms.Intersects(pt)
	case authpb.WRITE:
		return cachedPerms.writePerms.Intersects(pt)
	default:
		lg.Panic("unknown auth type", zap.String("auth-type", permtyp.String()))
	}
	return false
}

func (as *authStore) isRangeOpPermitted(userName string, key, rangeEnd []byte, permtyp authpb.Permission_Type) bool {
	as.rangePermCacheMu.RLock()
	defer as.rangePermCacheMu.RUnlock()

	rangePerm, ok := as.rangePermCache[userName]
	if !ok {
		as.lg.Error(
			"user doesn't exist",
			zap.String("user-name", userName),
		)
		return false
	}

	if len(rangeEnd) == 0 {
		return checkKeyPoint(as.lg, rangePerm, key, permtyp)
	}

	return checkKeyInterval(as.lg, rangePerm, key, rangeEnd, permtyp)
}

func (as *authStore) refreshRangePermCache(tx backend.ReadTx) {
	// Note that every authentication configuration update calls this method and it invalidates the entire
	// rangePermCache and reconstruct it based on information of users and roles stored in the backend.
	// This can be a costly operation.
	as.rangePermCacheMu.Lock()
	defer as.rangePermCacheMu.Unlock()

	as.rangePermCache = make(map[string]*unifiedRangePermissions)

	users := getAllUsers(as.lg, tx)
	for _, user := range users {
		userName := string(user.Name)
		perms := getMergedPerms(as.lg, tx, userName)
		if perms == nil {
			as.lg.Error(
				"failed to create a merged permission",
				zap.String("user-name", userName),
			)
			continue
		}
		as.rangePermCache[userName] = perms
	}
}

type unifiedRangePermissions struct {
	readPerms  adt.IntervalTree
	writePerms adt.IntervalTree
}
