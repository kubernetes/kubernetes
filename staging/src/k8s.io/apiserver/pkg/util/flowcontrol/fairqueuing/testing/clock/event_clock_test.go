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

package clock

import (
	"math/rand"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

type TestableEventClock interface {
	EventClock
	SetTime(time.Time)
	Run(*time.Time)
}

// settablePassiveClock allows setting current time of a passive clock
type settablePassiveClock interface {
	clock.PassiveClock
	SetTime(time.Time)
}

func exerciseTestableEventClock(t *testing.T, ec TestableEventClock, fuzz time.Duration) {
	exercisePassiveClock(t, ec)
	var numDone int32
	now := ec.Now()
	strictable := true
	const batchSize = 100
	times := make(chan time.Time, batchSize+1)
	try := func(abs, strict bool, d time.Duration) {
		f := func(u time.Time) {
			realD := ec.Since(now)
			atomic.AddInt32(&numDone, 1)
			times <- u
			if realD < d || strict && strictable && realD > d+fuzz {
				t.Errorf("Asked for %v, got %v", d, realD)
			}
		}
		if abs {
			ec.EventAfterTime(f, now.Add(d))
		} else {
			ec.EventAfterDuration(f, d)
		}
	}
	try(true, true, time.Minute)
	for i := 0; i < batchSize; i++ {
		d := time.Duration(rand.Intn(30)-3) * time.Second
		try(i%2 == 0, d >= 0, d)
	}
	ec.Run(nil)
	if numDone != batchSize+1 {
		t.Errorf("Got only %v events", numDone)
	}
	lastTime := now.Add(-3 * time.Second)
	for i := 0; i <= batchSize; i++ {
		nextTime := <-times
		if nextTime.Before(lastTime) {
			t.Errorf("Got %s after %s", nextTime, lastTime)
		}
	}
	endTime := ec.Now()
	dx := endTime.Sub(now)
	if dx > time.Minute+fuzz {
		t.Errorf("Run started at %#+v, ended at %#+v, dx=%d", now, endTime, dx)
	}
	now = endTime
	var shouldRun int32
	strictable = false
	for i := 0; i < batchSize; i++ {
		d := time.Duration(rand.Intn(30)-3) * time.Second
		try(i%2 == 0, d >= 0, d)
		if d <= 12*time.Second {
			shouldRun++
		}
	}
	ec.SetTime(now.Add(13*time.Second - 1))
	if numDone != batchSize+1+shouldRun {
		t.Errorf("Expected %v, but %v ran", shouldRun, numDone-batchSize-1)
	}
	lastTime = now.Add(-3 * time.Second)
	for i := int32(0); i < shouldRun; i++ {
		nextTime := <-times
		if nextTime.Before(lastTime) {
			t.Errorf("Got %s after %s", nextTime, lastTime)
		}
		lastTime = nextTime
	}
}

func exercisePassiveClock(t *testing.T, pc settablePassiveClock) {
	t1 := time.Now()
	t2 := t1.Add(time.Hour)
	pc.SetTime(t1)
	tx := pc.Now()
	if tx != t1 {
		t.Errorf("SetTime(%#+v); Now() => %#+v", t1, tx)
	}
	dx := pc.Since(t1)
	if dx != 0 {
		t.Errorf("Since() => %v", dx)
	}
	pc.SetTime(t2)
	dx = pc.Since(t1)
	if dx != time.Hour {
		t.Errorf("Since() => %v", dx)
	}
	tx = pc.Now()
	if tx != t2 {
		t.Errorf("Now() => %#+v", tx)
	}
}

func TestFakeEventClock(t *testing.T) {
	startTime := time.Now()
	fec, _ := NewFakeEventClock(startTime, 0, nil)
	exerciseTestableEventClock(t, fec, 0)
	fec, _ = NewFakeEventClock(startTime, time.Second, nil)
	exerciseTestableEventClock(t, fec, time.Second)
}

func exerciseEventClock(t *testing.T, ec EventClock, relax func(time.Duration)) {
	var numDone int32
	now := ec.Now()
	const batchSize = 100
	times := make(chan time.Time, batchSize+1)
	try := func(abs bool, d time.Duration) {
		f := func(u time.Time) {
			realD := ec.Since(now)
			atomic.AddInt32(&numDone, 1)
			times <- u
			if realD < d {
				t.Errorf("Asked for %v, got %v", d, realD)
			}
		}
		if abs {
			ec.EventAfterTime(f, now.Add(d))
		} else {
			ec.EventAfterDuration(f, d)
		}
	}
	try(true, time.Millisecond*3300)
	for i := 0; i < batchSize; i++ {
		d := time.Duration(rand.Intn(30)-3) * time.Millisecond * 100
		try(i%2 == 0, d)
	}
	relax(time.Second * 4)
	if atomic.LoadInt32(&numDone) != batchSize+1 {
		t.Errorf("Got only %v events", numDone)
	}
	lastTime := now
	for i := 0; i <= batchSize; i++ {
		nextTime := <-times
		if nextTime.Before(now) {
			continue
		}
		dt := nextTime.Sub(lastTime) / (50 * time.Millisecond)
		if dt < 0 {
			t.Errorf("Got %s after %s", nextTime, lastTime)
		}
		lastTime = nextTime
	}
}

func TestRealEventClock(t *testing.T) {
	exerciseEventClock(t, RealEventClock{}, func(d time.Duration) { time.Sleep(d) })
}
