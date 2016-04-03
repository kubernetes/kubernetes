package broadcastwriter

import (
	"bytes"
	"io"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/timeutils"
)

// BroadcastWriter accumulate multiple io.WriteCloser by stream.
type BroadcastWriter struct {
	sync.Mutex
	buf      *bytes.Buffer
	jsLogBuf *bytes.Buffer
	streams  map[string](map[io.WriteCloser]struct{})
}

// AddWriter adds new io.WriteCloser for stream.
// If stream is "", then all writes proceed as is. Otherwise every line from
// input will be packed to serialized jsonlog.JSONLog.
func (w *BroadcastWriter) AddWriter(writer io.WriteCloser, stream string) {
	w.Lock()
	if _, ok := w.streams[stream]; !ok {
		w.streams[stream] = make(map[io.WriteCloser]struct{})
	}
	w.streams[stream][writer] = struct{}{}
	w.Unlock()
}

// Write writes bytes to all writers. Failed writers will be evicted during
// this call.
func (w *BroadcastWriter) Write(p []byte) (n int, err error) {
	w.Lock()
	if writers, ok := w.streams[""]; ok {
		for sw := range writers {
			if n, err := sw.Write(p); err != nil || n != len(p) {
				// On error, evict the writer
				delete(writers, sw)
			}
		}
		if len(w.streams) == 1 {
			if w.buf.Len() >= 4096 {
				w.buf.Reset()
			} else {
				w.buf.Write(p)
			}
			w.Unlock()
			return len(p), nil
		}
	}
	if w.jsLogBuf == nil {
		w.jsLogBuf = new(bytes.Buffer)
		w.jsLogBuf.Grow(1024)
	}
	var timestamp string
	created := time.Now().UTC()
	w.buf.Write(p)
	for {
		if n := w.buf.Len(); n == 0 {
			break
		}
		i := bytes.IndexByte(w.buf.Bytes(), '\n')
		if i < 0 {
			break
		}
		lineBytes := w.buf.Next(i + 1)
		if timestamp == "" {
			timestamp, err = timeutils.FastMarshalJSON(created)
			if err != nil {
				continue
			}
		}

		for stream, writers := range w.streams {
			if stream == "" {
				continue
			}
			jsonLog := jsonlog.JSONLogBytes{Log: lineBytes, Stream: stream, Created: timestamp}
			err = jsonLog.MarshalJSONBuf(w.jsLogBuf)
			if err != nil {
				logrus.Errorf("Error making JSON log line: %s", err)
				continue
			}
			w.jsLogBuf.WriteByte('\n')
			b := w.jsLogBuf.Bytes()
			for sw := range writers {
				if _, err := sw.Write(b); err != nil {
					delete(writers, sw)
				}
			}
		}
		w.jsLogBuf.Reset()
	}
	w.jsLogBuf.Reset()
	w.Unlock()
	return len(p), nil
}

// Clean closes and removes all writers. Last non-eol-terminated part of data
// will be saved.
func (w *BroadcastWriter) Clean() error {
	w.Lock()
	for _, writers := range w.streams {
		for w := range writers {
			w.Close()
		}
	}
	w.streams = make(map[string](map[io.WriteCloser]struct{}))
	w.Unlock()
	return nil
}

func New() *BroadcastWriter {
	return &BroadcastWriter{
		streams: make(map[string](map[io.WriteCloser]struct{})),
		buf:     bytes.NewBuffer(nil),
	}
}
