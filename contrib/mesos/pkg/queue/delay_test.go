/*
Copyright 2015 The Kubernetes Authors.

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

package queue

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

const (
	tolerance = 100 * time.Millisecond // go time delays aren't perfect, this is our tolerance for errors WRT expected timeouts
)

func timedPriority(t time.Time) Priority {
	return Priority{ts: t}
}

func TestPQ(t *testing.T) {
	t.Parallel()

	var pq priorityQueue
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}

	now := timedPriority(time.Now())
	now2 := timedPriority(now.ts.Add(2 * time.Second))
	pq.Push(&qitem{priority: now2})
	if pq.Len() != 1 {
		t.Fatalf("pq.len should be 1")
	}
	x := pq.Pop()
	if x == nil {
		t.Fatalf("x is nil")
	}
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	item := x.(*qitem)
	if !item.priority.Equal(now2) {
		t.Fatalf("item.priority != now2")
	}

	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	now4 := timedPriority(now.ts.Add(4 * time.Second))
	now6 := timedPriority(now.ts.Add(4 * time.Second))
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now4})
	pq.Push(&qitem{priority: now6})
	pq.Swap(0, 2)
	if !pq[0].priority.Equal(now6) || !pq[2].priority.Equal(now2) {
		t.Fatalf("swap failed")
	}
	if pq.Less(1, 2) {
		t.Fatalf("now4 < now2")
	}
}

func TestPopEmptyPQ(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Expected panic from popping an empty PQ")
		}
	}()
	var pq priorityQueue
	pq.Pop()
}

type testjob struct {
	d        time.Duration
	t        time.Time
	deadline *time.Time
	uid      string
	instance int
}

func (j *testjob) GetDelay() time.Duration {
	return j.d
}

func (j testjob) GetUID() string {
	return j.uid
}

func (td *testjob) Deadline() (deadline time.Time, ok bool) {
	if td.deadline != nil {
		return *td.deadline, true
	} else {
		return time.Now(), false
	}
}

func TestDQ_sanity_check(t *testing.T) {
	t.Parallel()

	dq := NewDelayQueue()
	delay := 2 * time.Second
	dq.Add(&testjob{d: delay})

	before := time.Now()
	x := dq.Pop()

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

func TestDQ_Offer(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	dq := NewDelayQueue()
	delay := time.Second

	added := dq.Offer(&testjob{})
	if added {
		t.Fatalf("DelayQueue should not add offered job without deadline")
	}

	deadline := time.Now().Add(delay)
	added = dq.Offer(&testjob{deadline: &deadline})
	if !added {
		t.Fatalf("DelayQueue should add offered job with deadline")
	}

	before := time.Now()
	x := dq.Pop()

	now := time.Now()
	waitPeriod := now.Sub(before)

	if waitPeriod+tolerance < delay {
		t.Fatalf("delay too short: %v, expected: %v", waitPeriod, delay)
	}
	assert.NotNil(x)
	assert.Equal(x.(*testjob).deadline, &deadline)
}

func TestDQ_ordered_add_pop(t *testing.T) {
	t.Parallel()

	dq := NewDelayQueue()
	dq.Add(&testjob{d: 2 * time.Second})
	dq.Add(&testjob{d: 1 * time.Second})
	dq.Add(&testjob{d: 3 * time.Second})

	var finished [3]*testjob
	before := time.Now()
	idx := int32(-1)
	ch := make(chan bool, 3)
	//TODO: replace with `for range finished` once Go 1.3 support is dropped
	for n := 0; n < len(finished); n++ {
		go func() {
			var ok bool
			x := dq.Pop()
			i := atomic.AddInt32(&idx, 1)
			if finished[i], ok = x.(*testjob); !ok {
				t.Fatalf("expected a *testjob, not %v", x)
			}
			finished[i].t = time.Now()
			ch <- true
		}()
	}
	<-ch
	<-ch
	<-ch

	after := time.Now()
	totalDelay := after.Sub(before)
	if totalDelay+tolerance < (3 * time.Second) {
		t.Fatalf("totalDelay < 3s: %v", totalDelay)
	}
	for i, v := range finished {
		if v == nil {
			t.Fatalf("task %d was nil", i)
		}
		expected := time.Duration(i+1) * time.Second
		if v.d != expected {
			t.Fatalf("task %d had delay-priority %v, expected %v", i, v.d, expected)
		}
		actualDelay := v.t.Sub(before)
		if actualDelay+tolerance < v.d {
			t.Fatalf("task %d had actual-delay %v < expected delay %v", i, actualDelay, v.d)
		}
	}
}

func TestDQ_always_pop_earliest_deadline(t *testing.T) {
	t.Skip("disabled due to flakiness; see #11857")
	t.Parallel()

	// add a testjob with delay of 2s
	// spawn a func f1 that attempts to Pop() and wait for f1 to begin
	// add a testjob with a delay of 1s
	// check that the func f1 actually popped the 1s task (not the 2s task)

	dq := NewDelayQueue()
	dq.Add(&testjob{d: 2 * time.Second})
	ch := make(chan *testjob)
	started := make(chan bool)

	go func() {
		started <- true
		x := dq.Pop()
		job := x.(*testjob)
		job.t = time.Now()
		ch <- job
	}()

	<-started
	time.Sleep(500 * time.Millisecond) // give plently of time for Pop() to enter
	expected := 1 * time.Second
	dq.Add(&testjob{d: expected})
	job := <-ch

	if expected != job.d {
		t.Fatalf("Expected delay-prority of %v got instead got %v", expected, job.d)
	}

	job = dq.Pop().(*testjob)
	expected = 2 * time.Second
	if expected != job.d {
		t.Fatalf("Expected delay-prority of %v got instead got %v", expected, job.d)
	}
}

func TestDQ_always_pop_earliest_deadline_multi(t *testing.T) {
	t.Skip("disabled due to flakiness; see #11821")
	t.Parallel()

	dq := NewDelayQueue()
	dq.Add(&testjob{d: 2 * time.Second})

	ch := make(chan *testjob)
	multi := 10
	started := make(chan bool, multi)

	go func() {
		started <- true
		for i := 0; i < multi; i++ {
			x := dq.Pop()
			job := x.(*testjob)
			job.t = time.Now()
			ch <- job
		}
	}()

	<-started
	time.Sleep(500 * time.Millisecond) // give plently of time for Pop() to enter
	expected := 1 * time.Second

	for i := 0; i < multi; i++ {
		dq.Add(&testjob{d: expected})
	}
	for i := 0; i < multi; i++ {
		job := <-ch
		if expected != job.d {
			t.Fatalf("Expected delay-prority of %v got instead got %v", expected, job.d)
		}
	}

	job := dq.Pop().(*testjob)
	expected = 2 * time.Second
	if expected != job.d {
		t.Fatalf("Expected delay-prority of %v got instead got %v", expected, job.d)
	}
}

func TestDQ_negative_delay(t *testing.T) {
	t.Parallel()

	dq := NewDelayQueue()
	delay := -2 * time.Second
	dq.Add(&testjob{d: delay})

	before := time.Now()
	x := dq.Pop()

	now := time.Now()
	waitPeriod := now.Sub(before)

	if waitPeriod > tolerance {
		t.Fatalf("delay too long: %v, expected something less than: %v", waitPeriod, tolerance)
	}
	if x == nil {
		t.Fatalf("x is nil")
	}
	item := x.(*testjob)
	if item.d != delay {
		t.Fatalf("d != delay")
	}
}

func TestDFIFO_sanity_check(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	df := NewDelayFIFO()
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
	x := df.Pop(WithoutCancel())
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

	dq := NewDelayFIFO()
	delay := time.Second

	added := dq.Offer(&testjob{instance: 1}, ReplaceExisting)
	if added {
		t.Fatalf("DelayFIFO should not add offered job without deadline")
	}

	deadline := time.Now().Add(delay)
	added = dq.Offer(&testjob{deadline: &deadline, instance: 2}, ReplaceExisting)
	if !added {
		t.Fatalf("DelayFIFO should add offered job with deadline")
	}

	before := time.Now()
	x := dq.Pop(WithoutCancel())

	now := time.Now()
	waitPeriod := now.Sub(before)

	if waitPeriod+tolerance < delay {
		t.Fatalf("delay too short: %v, expected: %v", waitPeriod, delay)
	}
	assert.NotNil(x)
	assert.Equal(x.(*testjob).instance, 2)
}
