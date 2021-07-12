// Copyright 2021 The etcd Authors
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

package membership

import (
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
)

// IsMetaStoreOnly verifies if the given `store` contains only
// a meta-information (members, version) that can be recovered from the
// backend (storev3) as well as opposed to user-data.
func IsMetaStoreOnly(store v2store.Store) (bool, error) {
	event, err := store.Get("/", true, false)
	if err != nil {
		return false, err
	}
	for _, n := range event.Node.Nodes {
		if n.Key != storePrefix && n.Nodes.Len() > 0 {
			return false, nil
		}
	}

	return true, nil
}
