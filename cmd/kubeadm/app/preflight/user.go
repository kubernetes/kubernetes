/*
Copyright 2016 The Kubernetes Authors.

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

package preflight

import (
	"fmt"
	"os"
)

const (
	sudoUser = 0
)

type NotRootError struct {
	Msg string
	Uid int
}

func (e *NotRootError) Error() string {
	return fmt.Sprintf("%s: Current User Uid: %d", e.Msg, e.Uid)
}

func IsRoot() (int, bool) {
	return os.Getuid(), os.Getuid() == 0
}
