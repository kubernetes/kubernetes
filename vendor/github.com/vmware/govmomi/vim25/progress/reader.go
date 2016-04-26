/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

import (
	"container/list"
	"fmt"
	"io"
	"sync/atomic"
	"time"
)

type readerReport struct {
	t time.Time

	pos  int64
	size int64
	bps  *uint64

	err error
}

func (r readerReport) Percentage() float32 {
	return 100.0 * float32(r.pos) / float32(r.size)
}

func (r readerReport) Detail() string {
	const (
		KiB = 1024
		MiB = 1024 * KiB
		GiB = 1024 * MiB
	)

	// Use the reader's bps field, so this report returns an up-to-date number.
	//
	// For example: if there hasn't been progress for the last 5 seconds, the
	// most recent report should return "0B/s".
	//
	bps := atomic.LoadUint64(r.bps)

	switch {
	case bps >= GiB:
		return fmt.Sprintf("%.1fGiB/s", float32(bps)/float32(GiB))
	case bps >= MiB:
		return fmt.Sprintf("%.1fMiB/s", float32(bps)/float32(MiB))
	case bps >= KiB:
		return fmt.Sprintf("%.1fKiB/s", float32(bps)/float32(KiB))
	default:
		return fmt.Sprintf("%dB/s", bps)
	}
}

func (p readerReport) Error() error {
	return p.err
}

// reader wraps an io.Reader and sends a progress report over a channel for
// every read it handles.
type reader struct {
	r io.Reader

	pos  int64
	size int64

	bps uint64

	ch chan<- Report
}

func NewReader(s Sinker, r io.Reader, size int64) *reader {
	pr := reader{
		r: r,

		size: size,
	}

	// Reports must be sent downstream and to the bps computation loop.
	pr.ch = Tee(s, newBpsLoop(&pr.bps)).Sink()

	return &pr
}

// Read calls the Read function on the underlying io.Reader. Additionally,
// every read causes a progress report to be sent to the progress reader's
// underlying channel.
func (r *reader) Read(b []byte) (int, error) {
	n, err := r.r.Read(b)
	if err != nil {
		return n, err
	}

	r.pos += int64(n)
	q := readerReport{
		t:    time.Now(),
		pos:  r.pos,
		size: r.size,
		bps:  &r.bps,
	}

	r.ch <- q

	return n, err
}

// Done marks the progress reader as done, optionally including an error in the
// progress report. After sending it, the underlying channel is closed.
func (r *reader) Done(err error) {
	q := readerReport{
		t:    time.Now(),
		pos:  r.pos,
		size: r.size,
		bps:  &r.bps,
		err:  err,
	}

	r.ch <- q
	close(r.ch)
}

// newBpsLoop returns a sink that monitors and stores throughput.
func newBpsLoop(dst *uint64) SinkFunc {
	fn := func() chan<- Report {
		sink := make(chan Report)
		go bpsLoop(sink, dst)
		return sink
	}

	return fn
}

func bpsLoop(ch <-chan Report, dst *uint64) {
	l := list.New()

	for {
		var tch <-chan time.Time

		// Setup timer for front of list to become stale.
		if e := l.Front(); e != nil {
			dt := time.Second - time.Now().Sub(e.Value.(readerReport).t)
			tch = time.After(dt)
		}

		select {
		case q, ok := <-ch:
			if !ok {
				return
			}

			l.PushBack(q)
		case <-tch:
			l.Remove(l.Front())
		}

		// Compute new bps
		if l.Len() == 0 {
			atomic.StoreUint64(dst, 0)
		} else {
			f := l.Front().Value.(readerReport)
			b := l.Back().Value.(readerReport)
			atomic.StoreUint64(dst, uint64(b.pos-f.pos))
		}
	}
}
