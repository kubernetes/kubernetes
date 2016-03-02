/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package chown

import (
	"os"
)

// Interface is something that knows how to run the chown system call.
// It is non-recursive.
type Interface interface {
	// Chown changes the owning UID and GID of a file, implementing
	// the exact same semantics as os.Chown.
	Chown(path string, uid, gid int) error
}

func New() Interface {
	return &chownRunner{}
}

type chownRunner struct{}

func (_ *chownRunner) Chown(path string, uid, gid int) error {
	return os.Chown(path, uid, gid)
}
