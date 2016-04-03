/*
Copyright 2013 The Camlistore Authors

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

package syncdebug

import "testing"

func TestGoroutineID(t *testing.T) {
	c := make(chan int64, 2)
	c <- GoroutineID()
	go func() {
		c <- GoroutineID()
	}()
	if a, b := <-c, <-c; a == b {
		t.Errorf("both goroutine IDs were %d; expected different", a)
	}
}
