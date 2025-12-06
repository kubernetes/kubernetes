/*
Copyright The Kubernetes Authors.

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

package ktesting

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

// Run executes all given functions as sub-tests.
//
// The name of each sub-test is derived from the function name by
// stripping the package and "test" or "synctest" prefix (case doesn't matter).
// If the prefix is "synctest", then the function runs inside
// a synctest bubble (see TContext.SyncTest).
//
// This can be used in a single top-level test. If the test
// function is just called "Test", then the actual test names
// become "Test/<sub-test name>:
//
//	func Test(t *testing.T) {
//	    ktesting.Run(t, testSomething)
//	}
//
//	func testSomething(tCtx ktesting.TContext) { ... }
func Run(tb TB, funcs ...func(TContext)) {
	tCtx := Init(tb)
	for i, f := range funcs {
		run := tCtx.Run
		name := runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
		if index := strings.LastIndex(name, "."); index > 0 {
			name = name[index+1:]
		}
		switch {
		case strings.HasPrefix(strings.ToLower(name), "test"):
			name = name[4:]
		case strings.HasPrefix(strings.ToLower(name), "synctest"):
			name = name[8:]
			run = tCtx.SyncTest
		case name == "":
			name = fmt.Sprintf("function-%d", i)
		}
		run(name, f)
	}
}
