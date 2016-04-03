package hh

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func BenchmarkQueueAppend(b *testing.B) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		b.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	q, err := newQueue(dir, 1024*1024*1024)
	if err != nil {
		b.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		b.Fatalf("failed to open queue: %v", err)
	}

	for i := 0; i < b.N; i++ {
		if err := q.Append([]byte(fmt.Sprintf("%d", i))); err != nil {
			println(q.diskUsage())
			b.Fatalf("Queue.Append failed: %v", err)
		}
	}
}

func TestQueueAppendOne(t *testing.T) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	q, err := newQueue(dir, 1024)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	if err := q.Append([]byte("test")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	exp := filepath.Join(dir, "1")
	stats, err := os.Stat(exp)
	if os.IsNotExist(err) {
		t.Fatalf("Queue.Append file not exists. exp %v to exist", exp)
	}

	// 8 byte header ptr + 8 byte record len + record len
	if exp := int64(8 + 8 + 4); stats.Size() != exp {
		t.Fatalf("Queue.Append file size mismatch. got %v, exp %v", stats.Size(), exp)
	}

	cur, err := q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "test"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}
}

func TestQueueAppendMultiple(t *testing.T) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	q, err := newQueue(dir, 1024)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	if err := q.Append([]byte("one")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	if err := q.Append([]byte("two")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	for _, exp := range []string{"one", "two"} {
		cur, err := q.Current()
		if err != nil {
			t.Fatalf("Queue.Current failed: %v", err)
		}

		if string(cur) != exp {
			t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
		}

		if err := q.Advance(); err != nil {
			t.Fatalf("Queue.Advance failed: %v", err)
		}
	}
}

func TestQueueAdvancePastEnd(t *testing.T) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	// create the queue
	q, err := newQueue(dir, 1024)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	// append one entry, should go to the first segment
	if err := q.Append([]byte("one")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	// set the segment size low to force a new segment to be created
	q.SetMaxSegmentSize(12)

	// Should go into a new segment
	if err := q.Append([]byte("two")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	// should read from first segment
	cur, err := q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "one"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}

	if err := q.Advance(); err != nil {
		t.Fatalf("Queue.Advance failed: %v", err)
	}

	// ensure the first segment file is removed since we've advanced past the end
	_, err = os.Stat(filepath.Join(dir, "1"))
	if !os.IsNotExist(err) {
		t.Fatalf("Queue.Advance should have removed the segment")
	}

	// should read from second segment
	cur, err = q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "two"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}

	_, err = os.Stat(filepath.Join(dir, "2"))
	if os.IsNotExist(err) {
		t.Fatalf("Queue.Advance should have removed the segment")
	}

	if err := q.Advance(); err != nil {
		t.Fatalf("Queue.Advance failed: %v", err)
	}

	cur, err = q.Current()
	if err != io.EOF {
		t.Fatalf("Queue.Current should have returned error")
	}
}

func TestQueueFull(t *testing.T) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	// create the queue
	q, err := newQueue(dir, 10)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	if err := q.Append([]byte("one")); err != ErrQueueFull {
		t.Fatalf("Queue.Append expected to return queue full")
	}
}

func TestQueueReopen(t *testing.T) {
	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	// create the queue
	q, err := newQueue(dir, 1024)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	if err := q.Append([]byte("one")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	cur, err := q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "one"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}

	// close and re-open the queue
	if err := q.Close(); err != nil {
		t.Fatalf("Queue.Close failed: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to re-open queue: %v", err)
	}

	// Make sure we can read back the last current value
	cur, err = q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "one"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}

	if err := q.Append([]byte("two")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	if err := q.Advance(); err != nil {
		t.Fatalf("Queue.Advance failed: %v", err)
	}

	cur, err = q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "two"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}
}

func TestPurgeQueue(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping purge queue")
	}

	dir, err := ioutil.TempDir("", "hh_queue")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	// create the queue
	q, err := newQueue(dir, 1024)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}

	if err := q.Open(); err != nil {
		t.Fatalf("failed to open queue: %v", err)
	}

	if err := q.Append([]byte("one")); err != nil {
		t.Fatalf("Queue.Append failed: %v", err)
	}

	cur, err := q.Current()
	if err != nil {
		t.Fatalf("Queue.Current failed: %v", err)
	}

	if exp := "one"; string(cur) != exp {
		t.Errorf("Queue.Current mismatch: got %v, exp %v", string(cur), exp)
	}

	time.Sleep(time.Second)

	if err := q.PurgeOlderThan(time.Now()); err != nil {
		t.Errorf("Queue.PurgeOlderThan failed: %v", err)
	}

	_, err = q.Current()
	if err != io.EOF {
		t.Fatalf("Queue.Current expected io.EOF, got: %v", err)
	}

}
