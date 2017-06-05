// Copyright 2015 The etcd Authors
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

import (
	"errors"
	"fmt"
)

var (
	ErrUnknownMethod              = errors.New("etcdserver: unknown method")
	ErrStopped                    = errors.New("etcdserver: server stopped")
	ErrCanceled                   = errors.New("etcdserver: request cancelled")
	ErrTimeout                    = errors.New("etcdserver: request timed out")
	ErrTimeoutDueToLeaderFail     = errors.New("etcdserver: request timed out, possibly due to previous leader failure")
	ErrTimeoutDueToConnectionLost = errors.New("etcdserver: request timed out, possibly due to connection lost")
	ErrNotEnoughStartedMembers    = errors.New("etcdserver: re-configuration failed due to not enough started members")
	ErrNoLeader                   = errors.New("etcdserver: no leader")
	ErrRequestTooLarge            = errors.New("etcdserver: request is too large")
	ErrNoSpace                    = errors.New("etcdserver: no space")
	ErrInvalidAuthToken           = errors.New("etcdserver: invalid auth token")
)

type DiscoveryError struct {
	Op  string
	Err error
}

func (e DiscoveryError) Error() string {
	return fmt.Sprintf("failed to %s discovery cluster (%v)", e.Op, e.Err)
}
