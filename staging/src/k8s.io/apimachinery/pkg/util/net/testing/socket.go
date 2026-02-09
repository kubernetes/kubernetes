/*
Copyright 2025 The Kubernetes Authors.

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

// Package nettesting contains utilities for testing networking functionality.
// Don't use these utilities in production code. They have not been security
// reviewed.
package nettesting

import (
	"os"
	goruntime "runtime"
	"testing"
)

// MakeSocketNameForTest returns a socket name to use for the duration of a test.
// On Operating systems that support abstract sockets, it the name is prefixed with `@` to make it an abstract socket.
// On Operating systems that do not support abstract sockets, the name is treated as a filename and a cleanup hook is
// registered to delete the socket at the end of the test.
func MakeSocketNameForTest(t testing.TB, name string) string {
	var sockname = name
	switch goruntime.GOOS {
	case "darwin", "windows":
		t.Cleanup(func() { _ = os.Remove(sockname) })
	default:
		sockname = "@" + name
	}
	return sockname
}
