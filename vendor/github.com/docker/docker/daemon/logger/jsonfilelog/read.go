package jsonfilelog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/fsnotify/fsnotify"
	"golang.org/x/net/context"

	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/jsonfilelog/multireader"
	"github.com/docker/docker/pkg/filenotify"
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/tailfile"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

const maxJSONDecodeRetry = 20000

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

// ReadLogs implements the logger's LogReader interface for the logs
// created by this driver.
func (l *JSONFileLogger) ReadLogs(config logger.ReadConfig) *logger.LogWatcher {
	logWatcher := logger.NewLogWatcher()

	go l.readLogs(logWatcher, config)
	return logWatcher
}

func (l *JSONFileLogger) readLogs(logWatcher *logger.LogWatcher, config logger.ReadConfig) {
	defer close(logWatcher.Msg)

	// lock so the read stream doesn't get corrupted due to rotations or other log data written while we open these files
	// This will block writes!!!
	l.mu.RLock()

	// TODO it would be nice to move a lot of this reader implementation to the rotate logger object
	pth := l.writer.LogPath()
	var files []io.ReadSeeker
	for i := l.writer.MaxFiles(); i > 1; i-- {
		f, err := os.Open(fmt.Sprintf("%s.%d", pth, i-1))
		if err != nil {
			if !os.IsNotExist(err) {
				logWatcher.Err <- err
				l.mu.RUnlock()
				return
			}
			continue
		}
		defer f.Close()
		files = append(files, f)
	}

	latestFile, err := os.Open(pth)
	if err != nil {
		logWatcher.Err <- errors.Wrap(err, "error opening latest log file")
		l.mu.RUnlock()
		return
	}
	defer latestFile.Close()

	latestChunk, err := newSectionReader(latestFile)

	// Now we have the reader sectioned, all fd's opened, we can unlock.
	// New writes/rotates will not affect seeking through these files
	l.mu.RUnlock()

	if err != nil {
		logWatcher.Err <- err
		return
	}

	if config.Tail != 0 {
		tailer := multireader.MultiReadSeeker(append(files, latestChunk)...)
		tailFile(tailer, logWatcher, config.Tail, config.Since)
	}

	// close all the rotated files
	for _, f := range files {
		if err := f.(io.Closer).Close(); err != nil {
			logrus.WithField("logger", "json-file").Warnf("error closing tailed log file: %v", err)
		}
	}

	if !config.Follow || l.closed {
		return
	}

	notifyRotate := l.writer.NotifyRotate()
	defer l.writer.NotifyRotateEvict(notifyRotate)

	l.mu.Lock()
	l.readers[logWatcher] = struct{}{}
	l.mu.Unlock()

	followLogs(latestFile, logWatcher, notifyRotate, config.Since)

	l.mu.Lock()
	delete(l.readers, logWatcher)
	l.mu.Unlock()
}

func newSectionReader(f *os.File) (*io.SectionReader, error) {
	// seek to the end to get the size
	// we'll leave this at the end of the file since section reader does not advance the reader
	size, err := f.Seek(0, os.SEEK_END)
	if err != nil {
		return nil, errors.Wrap(err, "error getting current file size")
	}
	return io.NewSectionReader(f, 0, size), nil
}

func tailFile(f io.ReadSeeker, logWatcher *logger.LogWatcher, tail int, since time.Time) {
	var rdr io.Reader
	rdr = f
	if tail > 0 {
		ls, err := tailfile.TailFile(f, tail)
		if err != nil {
			logWatcher.Err <- err
			return
		}
		rdr = bytes.NewBuffer(bytes.Join(ls, []byte("\n")))
	}
	dec := json.NewDecoder(rdr)
	l := &jsonlog.JSONLog{}
	for {
		msg, err := decodeLogLine(dec, l)
		if err != nil {
			if err != io.EOF {
				logWatcher.Err <- err
			}
			return
		}
		if !since.IsZero() && msg.Timestamp.Before(since) {
			continue
		}
		select {
		case <-logWatcher.WatchClose():
			return
		case logWatcher.Msg <- msg:
		}
	}
}

func watchFile(name string) (filenotify.FileWatcher, error) {
	fileWatcher, err := filenotify.New()
	if err != nil {
		return nil, err
	}

	if err := fileWatcher.Add(name); err != nil {
		logrus.WithField("logger", "json-file").Warnf("falling back to file poller due to error: %v", err)
		fileWatcher.Close()
		fileWatcher = filenotify.NewPollingWatcher()

		if err := fileWatcher.Add(name); err != nil {
			fileWatcher.Close()
			logrus.Debugf("error watching log file for modifications: %v", err)
			return nil, err
		}
	}
	return fileWatcher, nil
}

func followLogs(f *os.File, logWatcher *logger.LogWatcher, notifyRotate chan interface{}, since time.Time) {
	dec := json.NewDecoder(f)
	l := &jsonlog.JSONLog{}

	name := f.Name()
	fileWatcher, err := watchFile(name)
	if err != nil {
		logWatcher.Err <- err
		return
	}
	defer func() {
		f.Close()
		fileWatcher.Remove(name)
		fileWatcher.Close()
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		select {
		case <-logWatcher.WatchClose():
			fileWatcher.Remove(name)
			cancel()
		case <-ctx.Done():
			return
		}
	}()

	var retries int
	handleRotate := func() error {
		f.Close()
		fileWatcher.Remove(name)

		// retry when the file doesn't exist
		for retries := 0; retries <= 5; retries++ {
			f, err = os.Open(name)
			if err == nil || !os.IsNotExist(err) {
				break
			}
		}
		if err != nil {
			return err
		}
		if err := fileWatcher.Add(name); err != nil {
			return err
		}
		dec = json.NewDecoder(f)
		return nil
	}

	errRetry := errors.New("retry")
	errDone := errors.New("done")
	waitRead := func() error {
		select {
		case e := <-fileWatcher.Events():
			switch e.Op {
			case fsnotify.Write:
				dec = json.NewDecoder(f)
				return nil
			case fsnotify.Rename, fsnotify.Remove:
				select {
				case <-notifyRotate:
				case <-ctx.Done():
					return errDone
				}
				if err := handleRotate(); err != nil {
					return err
				}
				return nil
			}
			return errRetry
		case err := <-fileWatcher.Errors():
			logrus.Debug("logger got error watching file: %v", err)
			// Something happened, let's try and stay alive and create a new watcher
			if retries <= 5 {
				fileWatcher.Close()
				fileWatcher, err = watchFile(name)
				if err != nil {
					return err
				}
				retries++
				return errRetry
			}
			return err
		case <-ctx.Done():
			return errDone
		}
	}

	handleDecodeErr := func(err error) error {
		if err == io.EOF {
			for {
				err := waitRead()
				if err == nil {
					break
				}
				if err == errRetry {
					continue
				}
				return err
			}
			return nil
		}
		// try again because this shouldn't happen
		if _, ok := err.(*json.SyntaxError); ok && retries <= maxJSONDecodeRetry {
			dec = json.NewDecoder(f)
			retries++
			return nil
		}
		// io.ErrUnexpectedEOF is returned from json.Decoder when there is
		// remaining data in the parser's buffer while an io.EOF occurs.
		// If the json logger writes a partial json log entry to the disk
		// while at the same time the decoder tries to decode it, the race condition happens.
		if err == io.ErrUnexpectedEOF && retries <= maxJSONDecodeRetry {
			reader := io.MultiReader(dec.Buffered(), f)
			dec = json.NewDecoder(reader)
			retries++
			return nil
		}
		return err
	}

	// main loop
	for {
		msg, err := decodeLogLine(dec, l)
		if err != nil {
			if err := handleDecodeErr(err); err != nil {
				if err == errDone {
					return
				}
				// we got an unrecoverable error, so return
				logWatcher.Err <- err
				return
			}
			// ready to try again
			continue
		}

		retries = 0 // reset retries since we've succeeded
		if !since.IsZero() && msg.Timestamp.Before(since) {
			continue
		}
		select {
		case logWatcher.Msg <- msg:
		case <-ctx.Done():
			logWatcher.Msg <- msg
			for {
				msg, err := decodeLogLine(dec, l)
				if err != nil {
					return
				}
				if !since.IsZero() && msg.Timestamp.Before(since) {
					continue
				}
				logWatcher.Msg <- msg
			}
		}
	}
}
