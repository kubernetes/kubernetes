package jsonfilelog

import (
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/jsonfilelog/jsonlog"
)

const maxJSONDecodeRetry = 20000

// ReadLogs implements the logger's LogReader interface for the logs
// created by this driver.
func (l *JSONFileLogger) ReadLogs(config logger.ReadConfig) *logger.LogWatcher {
	logWatcher := logger.NewLogWatcher()

	go l.readLogs(logWatcher, config)
	return logWatcher
}

func (l *JSONFileLogger) readLogs(watcher *logger.LogWatcher, config logger.ReadConfig) {
	defer close(watcher.Msg)

	l.mu.Lock()
	l.readers[watcher] = struct{}{}
	l.mu.Unlock()

	l.writer.ReadLogs(config, watcher)

	l.mu.Lock()
	delete(l.readers, watcher)
	l.mu.Unlock()
}

func decodeLogLine(dec *json.Decoder, l *jsonlog.JSONLog) (*logger.Message, error) {
	l.Reset()
	if err := dec.Decode(l); err != nil {
		return nil, err
	}

	var attrs []backend.LogAttr
	if len(l.Attrs) != 0 {
		attrs = make([]backend.LogAttr, 0, len(l.Attrs))
		for k, v := range l.Attrs {
			attrs = append(attrs, backend.LogAttr{Key: k, Value: v})
		}
	}
	msg := &logger.Message{
		Source:    l.Stream,
		Timestamp: l.Created,
		Line:      []byte(l.Log),
		Attrs:     attrs,
	}
	return msg, nil
}

// decodeFunc is used to create a decoder for the log file reader
func decodeFunc(rdr io.Reader) func() (*logger.Message, error) {
	l := &jsonlog.JSONLog{}
	dec := json.NewDecoder(rdr)
	return func() (msg *logger.Message, err error) {
		for retries := 0; retries < maxJSONDecodeRetry; retries++ {
			msg, err = decodeLogLine(dec, l)
			if err == nil {
				break
			}

			// try again, could be due to a an incomplete json object as we read
			if _, ok := err.(*json.SyntaxError); ok {
				dec = json.NewDecoder(rdr)
				retries++
				continue
			}

			// io.ErrUnexpectedEOF is returned from json.Decoder when there is
			// remaining data in the parser's buffer while an io.EOF occurs.
			// If the json logger writes a partial json log entry to the disk
			// while at the same time the decoder tries to decode it, the race condition happens.
			if err == io.ErrUnexpectedEOF {
				reader := io.MultiReader(dec.Buffered(), rdr)
				dec = json.NewDecoder(reader)
				retries++
			}
		}
		return msg, err
	}
}
