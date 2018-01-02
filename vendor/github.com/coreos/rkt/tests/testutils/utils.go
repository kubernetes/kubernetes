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

package testutils

import (
	"fmt"
	"os"
	"testing"
	"time"
)

const (
	nobodyUid = 65534
	nobodyGid = 65534
)

func GetUnprivilegedUidGid() (int, int) {
	return nobodyUid, nobodyGid
}

func GetValueFromEnvOrPanic(envVar string) string {
	path := os.Getenv(envVar)
	if path == "" {
		panic(fmt.Sprintf("Empty %v environment variable\n", envVar))
	}
	return path
}

func WaitOrTimeout(t *testing.T, timeout time.Duration, notify chan struct{}) {
	select {
	case <-time.After(timeout):
		t.Fatalf("Timeout after %v", timeout)
	case <-notify:
	}
}
