// Copyright 2016 CoreOS, Inc.
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

package testutil

import (
	"fmt"
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
	isLeaked := CheckLeakedGoroutine()
	if !isLeaked {
		fmt.Fprintln(os.Stderr, "expected leaky goroutines but none is detected")
		os.Exit(1)
	}
	os.Exit(0)
}

func TestSample(t *testing.T) {
	defer AfterTest(t)
	for range make([]struct{}, 100) {
		go func() {
			select {}
		}()
	}
}
