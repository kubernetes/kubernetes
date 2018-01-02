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

func TestScaleMany(t *testing.T) {
	ch := make(chan Report)
	a := NewAggregator(dummySinker{ch})
	defer a.Done()

	s := Scale(a, 5)

	go func() {
		for i := 0; i < 5; i++ {
			go func(ch chan<- Report) {
				ch <- dummyReport{p: 0.0}
				ch <- dummyReport{p: 50.0}
				close(ch)
			}(s.Sink())
		}
	}()

	// Expect percentages to be scaled across sinks
	for p := float32(0.0); p < 100.0; p += 10.0 {
		r := <-ch
		if r.Percentage() != p {
			t.Errorf("Expected percentage to be: %.0f%%", p)
		}
	}
}
