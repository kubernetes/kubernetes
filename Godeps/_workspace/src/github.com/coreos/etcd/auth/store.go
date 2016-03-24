// Copyright 2016 Nippon Telegraph and Telephone Corporation.
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
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/pkg/capnslog"
)

type backendGetter interface {
	Backend() backend.Backend
}

var (
	enableFlagKey  = []byte("authEnabled")
	authBucketName = []byte("auth")

	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "auth")
)

type AuthStore interface {
	// AuthEnable() turns on the authentication feature
	AuthEnable()
}

type authStore struct {
	bgetter backendGetter
}

func (as *authStore) AuthEnable() {
	value := []byte{1}

	b := as.bgetter.Backend()
	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafePut(authBucketName, enableFlagKey, value)
	tx.Unlock()
	b.ForceCommit()

	plog.Noticef("Authentication enabled")
}

func NewAuthStore(bgetter backendGetter) *authStore {
	b := bgetter.Backend()
	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket(authBucketName)
	tx.Unlock()
	b.ForceCommit()

	return &authStore{
		bgetter: bgetter,
	}
}
