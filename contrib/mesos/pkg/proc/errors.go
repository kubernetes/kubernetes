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

package proc

import (
	"errors"
)

var (
	errProcessTerminated = errors.New("cannot execute action because process has terminated")
	errIllegalState      = errors.New("illegal state, cannot execute action")
)

func IsProcessTerminated(err error) bool {
	return err == errProcessTerminated
}

func IsIllegalState(err error) bool {
	return err == errIllegalState
}
