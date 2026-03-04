/*
Copyright 2018 The Kubernetes Authors.

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

package logreduction

import (
	"testing"
	"time"
)

var time0 = time.Unix(1000, 0)
var time1 = time.Unix(1001, 0)
var time2 = time.Unix(1012, 0)
var identicalErrorDelay = 10 * time.Second
var testCount = 0

const (
	mesg1 = "This is a message"
	mesg2 = "This is not a message"
	id1   = "Container1"
	id2   = "Container2"
)

func checkThat(t *testing.T, r *LogReduction, m, id string) {
	testCount++
	if !r.ShouldMessageBePrinted(m, id) {
		t.Errorf("Case %d failed (%s/%s should be printed)", testCount, m, id)
	}
}

func checkThatNot(t *testing.T, r *LogReduction, m, id string) {
	testCount++
	if r.ShouldMessageBePrinted(m, id) {
		t.Errorf("Case %d failed (%s/%s should not be printed)", testCount, m, id)
	}
}

func TestLogReduction(t *testing.T) {
	var timeToReturn = time0
	nowfunc = func() time.Time { return timeToReturn }
	r := NewLogReduction(identicalErrorDelay)
	checkThat(t, r, mesg1, id1)    // 1
	checkThatNot(t, r, mesg1, id1) // 2
	checkThat(t, r, mesg1, id2)    // 3
	checkThatNot(t, r, mesg1, id1) // 4
	timeToReturn = time1
	checkThatNot(t, r, mesg1, id1) // 5
	timeToReturn = time2
	checkThat(t, r, mesg1, id1)    // 6
	checkThatNot(t, r, mesg1, id1) // 7
	checkThat(t, r, mesg2, id1)    // 8
	checkThat(t, r, mesg1, id1)    // 9
	checkThat(t, r, mesg1, id2)    // 10
	r.ClearID(id1)
	checkThat(t, r, mesg1, id1)    // 11
	checkThatNot(t, r, mesg1, id2) // 12
}
