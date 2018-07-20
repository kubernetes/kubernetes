package logger

import (
	"bytes"
	"io"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	// readSize is the maximum bytes read during a single read
	// operation.
	readSize = 2 * 1024

	// defaultBufSize provides a reasonable default for loggers that do
	// not have an external limit to impose on log line size.
	defaultBufSize = 16 * 1024
)

// Copier can copy logs from specified sources to Logger and attach Timestamp.
// Writes are concurrent, so you need implement some sync in your logger.
type Copier struct {
	// srcs is map of name -> reader pairs, for example "stdout", "stderr"
	srcs      map[string]io.Reader
	dst       Logger
	copyJobs  sync.WaitGroup
	closeOnce sync.Once
	closed    chan struct{}
}

// NewCopier creates a new Copier
func NewCopier(srcs map[string]io.Reader, dst Logger) *Copier {
	return &Copier{
		srcs:   srcs,
		dst:    dst,
		closed: make(chan struct{}),
	}
}

// Run starts logs copying
func (c *Copier) Run() {
	for src, w := range c.srcs {
		c.copyJobs.Add(1)
		go c.copySrc(src, w)
	}
}

func (c *Copier) copySrc(name string, src io.Reader) {
	defer c.copyJobs.Done()

	bufSize := defaultBufSize
	if sizedLogger, ok := c.dst.(SizedLogger); ok {
		bufSize = sizedLogger.BufSize()
	}
	buf := make([]byte, bufSize)

	n := 0
	eof := false

	for {
		select {
		case <-c.closed:
			return
		default:
			// Work out how much more data we are okay with reading this time.
			upto := n + readSize
			if upto > cap(buf) {
				upto = cap(buf)
			}
			// Try to read that data.
			if upto > n {
				read, err := src.Read(buf[n:upto])
				if err != nil {
					if err != io.EOF {
						logrus.Errorf("Error scanning log stream: %s", err)
						return
					}
					eof = true
				}
				n += read
			}
			// If we have no data to log, and there's no more coming, we're done.
			if n == 0 && eof {
				return
			}
			// Break up the data that we've buffered up into lines, and log each in turn.
			p := 0
			for q := bytes.IndexByte(buf[p:n], '\n'); q >= 0; q = bytes.IndexByte(buf[p:n], '\n') {
				select {
				case <-c.closed:
					return
				default:
					msg := NewMessage()
					msg.Source = name
					msg.Timestamp = time.Now().UTC()
					msg.Line = append(msg.Line, buf[p:p+q]...)

					if logErr := c.dst.Log(msg); logErr != nil {
						logrus.Errorf("Failed to log msg %q for logger %s: %s", msg.Line, c.dst.Name(), logErr)
					}
				}
				p += q + 1
			}
			// If there's no more coming, or the buffer is full but
			// has no newlines, log whatever we haven't logged yet,
			// noting that it's a partial log line.
			if eof || (p == 0 && n == len(buf)) {
				if p < n {
					msg := NewMessage()
					msg.Source = name
					msg.Timestamp = time.Now().UTC()
					msg.Line = append(msg.Line, buf[p:n]...)
					msg.Partial = true

					if logErr := c.dst.Log(msg); logErr != nil {
						logrus.Errorf("Failed to log msg %q for logger %s: %s", msg.Line, c.dst.Name(), logErr)
					}
					p = 0
					n = 0
				}
				if eof {
					return
				}
			}
			// Move any unlogged data to the front of the buffer in preparation for another read.
			if p > 0 {
				copy(buf[0:], buf[p:n])
				n -= p
			}
		}
	}
}

// Wait waits until all copying is done
func (c *Copier) Wait() {
	c.copyJobs.Wait()
}

// Close closes the copier
func (c *Copier) Close() {
	c.closeOnce.Do(func() {
		close(c.closed)
	})
}
