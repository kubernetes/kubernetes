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

	// storePermsPrefix is the internal prefix of the storage layer dedicated to storing user data.
	// refer to https://github.com/etcd-io/etcd/blob/v3.5.21/server/etcdserver/api/v2auth/auth.go#L40
	storePermsPrefix := "/2"
	for _, n := range event.Node.Nodes {
		if n.Key == storePrefix {
			continue
		}

		// For auth data, even after we remove all users and roles, the node
		// "/2/roles" and "/2/users" are still present in the tree. We need
		// to exclude such case. See an example below,
		// Refer to https://github.com/etcd-io/etcd/discussions/20231#discussioncomment-13791940
		/*
			"2": {
				"Path": "/2",
					"CreatedIndex": 204749,
					"ModifiedIndex": 204749,
					"ExpireTime": "0001-01-01T00:00:00Z",
					"Value": "",
					"Children": {
					"enabled": {
						"Path": "/2/enabled",
							"CreatedIndex": 204752,
							"ModifiedIndex": 16546016,
							"ExpireTime": "0001-01-01T00:00:00Z",
							"Value": "false",
							"Children": null
					},
					"roles": {
						"Path": "/2/roles",
							"CreatedIndex": 204751,
							"ModifiedIndex": 204751,
							"ExpireTime": "0001-01-01T00:00:00Z",
							"Value": "",
							"Children": {}
					},
					"users": {
						"Path": "/2/users",
							"CreatedIndex": 204750,
							"ModifiedIndex": 204750,
							"ExpireTime": "0001-01-01T00:00:00Z",
							"Value": "",
							"Children": {}
					}
				}
			}
		*/
		if n.Key == storePermsPrefix {
			if n.Nodes.Len() > 0 {
				for _, child := range n.Nodes {
					if child.Nodes.Len() > 0 {
						return false, nil
					}
				}
			}
			continue
		}

		if n.Nodes.Len() > 0 {
			return false, nil
		}
	}

	return true, nil
}
