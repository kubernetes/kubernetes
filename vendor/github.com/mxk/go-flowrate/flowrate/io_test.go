//
// Written by Maxim Khitrov (November 2012)
//

package flowrate

import (
	"bytes"
	"reflect"
	"testing"
	"time"
)

const (
	_50ms  = 50 * time.Millisecond
	_100ms = 100 * time.Millisecond
	_200ms = 200 * time.Millisecond
	_300ms = 300 * time.Millisecond
	_400ms = 400 * time.Millisecond
	_500ms = 500 * time.Millisecond
)

func nextStatus(m *Monitor) Status {
	samples := m.samples
	for i := 0; i < 30; i++ {
		if s := m.Status(); s.Samples != samples {
			return s
		}
		time.Sleep(5 * time.Millisecond)
	}
	return m.Status()
}

func TestReader(t *testing.T) {
	in := make([]byte, 100)
	for i := range in {
		in[i] = byte(i)
	}
	b := make([]byte, 100)
	r := NewReader(bytes.NewReader(in), 100)
	start := time.Now()

	// Make sure r implements Limiter
	_ = Limiter(r)

	// 1st read of 10 bytes is performed immediately
	if n, err := r.Read(b); n != 10 || err != nil {
		t.Fatalf("r.Read(b) expected 10 (<nil>); got %v (%v)", n, err)
	} else if rt := time.Since(start); rt > _50ms {
		t.Fatalf("r.Read(b) took too long (%v)", rt)
	}

	// No new Reads allowed in the current sample
	r.SetBlocking(false)
	if n, err := r.Read(b); n != 0 || err != nil {
		t.Fatalf("r.Read(b) expected 0 (<nil>); got %v (%v)", n, err)
	} else if rt := time.Since(start); rt > _50ms {
		t.Fatalf("r.Read(b) took too long (%v)", rt)
	}

	status := [6]Status{0: r.Status()} // No samples in the first status

	// 2nd read of 10 bytes blocks until the next sample
	r.SetBlocking(true)
	if n, err := r.Read(b[10:]); n != 10 || err != nil {
		t.Fatalf("r.Read(b[10:]) expected 10 (<nil>); got %v (%v)", n, err)
	} else if rt := time.Since(start); rt < _100ms {
		t.Fatalf("r.Read(b[10:]) returned ahead of time (%v)", rt)
	}

	status[1] = r.Status()            // 1st sample
	status[2] = nextStatus(r.Monitor) // 2nd sample
	status[3] = nextStatus(r.Monitor) // No activity for the 3rd sample

	if n := r.Done(); n != 20 {
		t.Fatalf("r.Done() expected 20; got %v", n)
	}

	status[4] = r.Status()
	status[5] = nextStatus(r.Monitor) // Timeout
	start = status[0].Start

	// Active, Start, Duration, Idle, Bytes, Samples, InstRate, CurRate, AvgRate, PeakRate, BytesRem, TimeRem, Progress
	want := []Status{
		Status{true, start, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		Status{true, start, _100ms, 0, 10, 1, 100, 100, 100, 100, 0, 0, 0},
		Status{true, start, _200ms, _100ms, 20, 2, 100, 100, 100, 100, 0, 0, 0},
		Status{true, start, _300ms, _200ms, 20, 3, 0, 90, 67, 100, 0, 0, 0},
		Status{false, start, _300ms, 0, 20, 3, 0, 0, 67, 100, 0, 0, 0},
		Status{false, start, _300ms, 0, 20, 3, 0, 0, 67, 100, 0, 0, 0},
	}
	for i, s := range status {
		if !reflect.DeepEqual(&s, &want[i]) {
			t.Errorf("r.Status(%v) expected %v; got %v", i, want[i], s)
		}
	}
	if !bytes.Equal(b[:20], in[:20]) {
		t.Errorf("r.Read() input doesn't match output")
	}
}

func TestWriter(t *testing.T) {
	b := make([]byte, 100)
	for i := range b {
		b[i] = byte(i)
	}
	w := NewWriter(&bytes.Buffer{}, 200)
	start := time.Now()

	// Make sure w implements Limiter
	_ = Limiter(w)

	// Non-blocking 20-byte write for the first sample returns ErrLimit
	w.SetBlocking(false)
	if n, err := w.Write(b); n != 20 || err != ErrLimit {
		t.Fatalf("w.Write(b) expected 20 (ErrLimit); got %v (%v)", n, err)
	} else if rt := time.Since(start); rt > _50ms {
		t.Fatalf("w.Write(b) took too long (%v)", rt)
	}

	// Blocking 80-byte write
	w.SetBlocking(true)
	if n, err := w.Write(b[20:]); n != 80 || err != nil {
		t.Fatalf("w.Write(b[20:]) expected 80 (<nil>); got %v (%v)", n, err)
	} else if rt := time.Since(start); rt < _400ms {
		t.Fatalf("w.Write(b[20:]) returned ahead of time (%v)", rt)
	}

	w.SetTransferSize(100)
	status := []Status{w.Status(), nextStatus(w.Monitor)}
	start = status[0].Start

	// Active, Start, Duration, Idle, Bytes, Samples, InstRate, CurRate, AvgRate, PeakRate, BytesRem, TimeRem, Progress
	want := []Status{
		Status{true, start, _400ms, 0, 80, 4, 200, 200, 200, 200, 20, _100ms, 80000},
		Status{true, start, _500ms, _100ms, 100, 5, 200, 200, 200, 200, 0, 0, 100000},
	}
	for i, s := range status {
		if !reflect.DeepEqual(&s, &want[i]) {
			t.Errorf("w.Status(%v) expected %v; got %v", i, want[i], s)
		}
	}
	if !bytes.Equal(b, w.Writer.(*bytes.Buffer).Bytes()) {
		t.Errorf("w.Write() input doesn't match output")
	}
}
