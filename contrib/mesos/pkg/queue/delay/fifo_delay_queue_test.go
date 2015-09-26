/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package delay

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDFIFO_sanity_check(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	df := NewFIFOQueue()
	delay := 2 * time.Second
	df.Add(&testjob{d: delay, uid: "a", instance: 1}, ReplaceExisting)
	assert.True(df.ContainedIDs().Has("a"))

	// re-add by ReplaceExisting
	df.Add(&testjob{d: delay, uid: "a", instance: 2}, ReplaceExisting)
	assert.True(df.ContainedIDs().Has("a"))

	a, ok := df.Get("a")
	assert.True(ok)
	assert.Equal(a.(*testjob).instance, 2)

	// re-add by KeepExisting
	df.Add(&testjob{d: delay, uid: "a", instance: 3}, KeepExisting)
	assert.True(df.ContainedIDs().Has("a"))

	a, ok = df.Get("a")
	assert.True(ok)
	assert.Equal(a.(*testjob).instance, 2)

	// pop last
	before := time.Now()
	x := df.Pop()
	assert.Equal(a.(*testjob).instance, 2)

	now := time.Now()
	waitPeriod := now.Sub(before)

	if waitPeriod+tolerance < delay {
		t.Fatalf("delay too short: %v, expected: %v", waitPeriod, delay)
	}
	if x == nil {
		t.Fatalf("x is nil")
	}
	item := x.(*testjob)
	if item.d != delay {
		t.Fatalf("d != delay")
	}
}

func TestDFIFO_Offer(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	dq := NewFIFOQueue()
	delay := time.Second

	added := dq.Offer(&testjob{instance: 1}, ReplaceExisting)
	if added {
		t.Fatalf("DelayFIFO should not add offered job without eventTime")
	}

	eventTime := time.Now().Add(delay)
	added = dq.Offer(&testjob{eventTime: &eventTime, instance: 2}, ReplaceExisting)
	if !added {
		t.Fatalf("DelayFIFO should add offered job with eventTime")
	}

	before := time.Now()
	x := dq.Pop()

	now := time.Now()
	waitPeriod := now.Sub(before)

	if waitPeriod+tolerance < delay {
		t.Fatalf("delay too short: %v, expected: %v", waitPeriod, delay)
	}
	assert.NotNil(x)
	assert.Equal(x.(*testjob).instance, 2)
}
