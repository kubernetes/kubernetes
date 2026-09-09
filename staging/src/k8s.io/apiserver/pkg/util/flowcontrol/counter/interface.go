/*
Copyright 2019 The Kubernetes Authors.

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

package counter

// GoRoutineCounter keeps track of the number of active goroutines
// working on/for something.  This is a utility that makes such code more
// testable.  The code uses this utility to report the number of active
// goroutines to the test code, so that the test code can advance a fake
// clock when and only when the code being tested has finished all
// the work that is ready to do at the present time.
type GoRoutineCounter interface {
	// Add adds the given delta to the count of active goroutines.
	// Call Add(1) before forking a goroutine, Add(-1) at the end of that goroutine.
	// Call Add(-1) just before waiting on something from another goroutine (e.g.,
	// just before a `select`).
	// Call Add(1) just before doing something that unblocks a goroutine that is
	// waiting on that something.
	Add(delta int)
}
