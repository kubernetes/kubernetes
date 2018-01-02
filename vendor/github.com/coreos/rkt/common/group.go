// Copyright 2015 The rkt Authors
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

package common

import (
	"github.com/coreos/rkt/pkg/group"
)

const (
	RktGroup      = "rkt"       // owns /var/lib/rkt
	RktAdminGroup = "rkt-admin" // owns /etc/rkt
)

// LookupGid reads the group file and returns the gid of the group
// specified by groupName.
func LookupGid(groupName string) (gid int, err error) {
	return group.LookupGid(groupName)
}
