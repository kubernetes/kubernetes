package progress

import (
	"io"
	"time"

	"golang.org/x/time/rate"
)

// Reader is a Reader with progress bar.
type Reader struct {
	in          io.ReadCloser // Stream to read from
	out         Output        // Where to send progress bar to
	size        int64
	current     int64
	lastUpdate  int64
	id          string
	action      string
	rateLimiter *rate.Limiter
}

// NewProgressReader creates a new ProgressReader.
func NewProgressReader(in io.ReadCloser, out Output, size int64, id, action string) *Reader {
	return &Reader{
		in:          in,
		out:         out,
		size:        size,
		id:          id,
		action:      action,
		rateLimiter: rate.NewLimiter(rate.Every(100*time.Millisecond), 1),
	}
}

func (p *Reader) Read(buf []byte) (n int, err error) {
	read, err := p.in.Read(buf)
	p.current += int64(read)
	updateEvery := int64(1024 * 512) //512kB
	if p.size > 0 {
		// Update progress for every 1% read if 1% < 512kB
		if increment := int64(0.01 * float64(p.size)); increment < updateEvery {
			updateEvery = increment
		}
	}
	if p.current-p.lastUpdate > updateEvery || err != nil {
		p.updateProgress(err != nil && read == 0)
		p.lastUpdate = p.current
	}

	return read, err
}

// Close closes the progress reader and its underlying reader.
func (p *Reader) Close() error {
	if p.current < p.size {
		// print a full progress bar when closing prematurely
		p.current = p.size
		p.updateProgress(false)
	}
	return p.in.Close()
}

func (p *Reader) updateProgress(last bool) {
	if last || p.current == p.size || p.rateLimiter.Allow() {
		p.out.WriteProgress(Progress{ID: p.id, Action: p.action, Current: p.current, Total: p.size, LastUpdate: last})
	}
}
