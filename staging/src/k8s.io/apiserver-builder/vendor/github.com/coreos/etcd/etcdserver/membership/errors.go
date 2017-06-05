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

package membership

import (
	"errors"

	etcdErr "github.com/coreos/etcd/error"
)

var (
	ErrIDRemoved     = errors.New("membership: ID removed")
	ErrIDExists      = errors.New("membership: ID exists")
	ErrIDNotFound    = errors.New("membership: ID not found")
	ErrPeerURLexists = errors.New("membership: peerURL exists")
)

func isKeyNotFound(err error) bool {
	e, ok := err.(*etcdErr.Error)
	return ok && e.ErrorCode == etcdErr.EcodeKeyNotFound
}
