// Copyright 2015 CoreOS, Inc.
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

// Package journal provides write bindings to the local systemd journal.
// It is implemented in pure Go and connects to the journal directly over its
// unix socket.
//
// To read from the journal, see the "sdjournal" package, which wraps the
// sd-journal a C API.
//
// http://www.freedesktop.org/software/systemd/man/systemd-journald.service.html
package journal

import (
	"errors"
)

func Enabled() bool {
	return false
}

func Send(message string, priority Priority, vars map[string]string) error {
	return errors.New("could not initialize socket to journald")
}
