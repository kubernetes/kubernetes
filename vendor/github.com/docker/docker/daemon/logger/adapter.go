package logger

import (
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/api/types/plugins/logdriver"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// pluginAdapter takes a plugin and implements the Logger interface for logger
// instances
type pluginAdapter struct {
	driverName   string
	id           string
	plugin       logPlugin
	basePath     string
	fifoPath     string
	capabilities Capability
	logInfo      Info

	// synchronize access to the log stream and shared buffer
	mu     sync.Mutex
	enc    logdriver.LogEntryEncoder
	stream io.WriteCloser
	// buf is shared for each `Log()` call to reduce allocations.
	// buf must be protected by mutex
	buf logdriver.LogEntry
}

func (a *pluginAdapter) Log(msg *Message) error {
	a.mu.Lock()

	a.buf.Line = msg.Line
	a.buf.TimeNano = msg.Timestamp.UnixNano()
	a.buf.Partial = msg.Partial
	a.buf.Source = msg.Source

	err := a.enc.Encode(&a.buf)
	a.buf.Reset()

	a.mu.Unlock()

	PutMessage(msg)
	return err
}

func (a *pluginAdapter) Name() string {
	return a.driverName
}

func (a *pluginAdapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := a.plugin.StopLogging(strings.TrimPrefix(a.fifoPath, a.basePath)); err != nil {
		return err
	}

	if err := a.stream.Close(); err != nil {
		logrus.WithError(err).Error("error closing plugin fifo")
	}
	if err := os.Remove(a.fifoPath); err != nil && !os.IsNotExist(err) {
		logrus.WithError(err).Error("error cleaning up plugin fifo")
	}

	// may be nil, especially for unit tests
	if pluginGetter != nil {
		pluginGetter.Get(a.Name(), extName, plugingetter.Release)
	}
	return nil
}

type pluginAdapterWithRead struct {
	*pluginAdapter
}

func (a *pluginAdapterWithRead) ReadLogs(config ReadConfig) *LogWatcher {
	watcher := NewLogWatcher()

	go func() {
		defer close(watcher.Msg)
		stream, err := a.plugin.ReadLogs(a.logInfo, config)
		if err != nil {
			watcher.Err <- errors.Wrap(err, "error getting log reader")
			return
		}
		defer stream.Close()

		dec := logdriver.NewLogEntryDecoder(stream)
		for {
			select {
			case <-watcher.WatchClose():
				return
			default:
			}

			var buf logdriver.LogEntry
			if err := dec.Decode(&buf); err != nil {
				if err == io.EOF {
					return
				}
				select {
				case watcher.Err <- errors.Wrap(err, "error decoding log message"):
				case <-watcher.WatchClose():
				}
				return
			}

			msg := &Message{
				Timestamp: time.Unix(0, buf.TimeNano),
				Line:      buf.Line,
				Source:    buf.Source,
			}

			// plugin should handle this, but check just in case
			if !config.Since.IsZero() && msg.Timestamp.Before(config.Since) {
				continue
			}

			select {
			case watcher.Msg <- msg:
			case <-watcher.WatchClose():
				// make sure the message we consumed is sent
				watcher.Msg <- msg
				return
			}
		}
	}()

	return watcher
}
