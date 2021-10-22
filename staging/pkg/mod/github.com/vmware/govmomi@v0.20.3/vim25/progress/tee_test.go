/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

import "testing"

func TestTee(t *testing.T) {
	var ok bool

	ch1 := make(chan Report)
	ch2 := make(chan Report)

	s := Tee(&dummySinker{ch: ch1}, &dummySinker{ch: ch2})

	in := s.Sink()
	in <- dummyReport{}
	close(in)

	// Receive dummy on both sinks
	<-ch1
	<-ch2

	_, ok = <-ch1
	if ok {
		t.Errorf("Expected channel to be closed")
	}

	_, ok = <-ch2
	if ok {
		t.Errorf("Expected channel to be closed")
	}
}
