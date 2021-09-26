// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import "testing"

func TestInotifyWatchLen(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	fdinfos, err := p1.FileDescriptorsInfo()
	if err != nil {
		t.Fatal(err)
	}
	l, err := fdinfos.InotifyWatchLen()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := 3, l; want != have {
		t.Errorf("want length %d, have %d", want, have)
	}
}
