// Copyright 2016 The rkt Authors
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

// +build kvm

package main

import "testing"

var pidFileName = "pid"

func TestPidFileDelayedStart(t *testing.T) {
	NewPidFileDelayedStartTest(pidFileName)
}

func TestPidFileAbortedStart(t *testing.T) {
	// For nspawn, the escape character is ^]^]^]. This process will exit with code 1
	NewPidFileAbortedStartTest(pidFileName, "\001\170", 0)
}
