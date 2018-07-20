package loggerutils

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/loggerutils/multireader"
	"github.com/docker/docker/pkg/filenotify"
	"github.com/docker/docker/pkg/pubsub"
	"github.com/docker/docker/pkg/tailfile"
	"github.com/fsnotify/fsnotify"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// LogFile is Logger implementation for default Docker logging.
type LogFile struct {
	f             *os.File // store for closing
	closed        bool
	mu            sync.RWMutex
	capacity      int64 //maximum size of each file
	currentSize   int64 // current size of the latest file
	maxFiles      int   //maximum number of files
	notifyRotate  *pubsub.Publisher
	marshal       logger.MarshalFunc
	createDecoder makeDecoderFunc
}

type makeDecoderFunc func(rdr io.Reader) func() (*logger.Message, error)

//NewLogFile creates new LogFile
func NewLogFile(logPath string, capacity int64, maxFiles int, marshaller logger.MarshalFunc, decodeFunc makeDecoderFunc) (*LogFile, error) {
	log, err := os.OpenFile(logPath, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0640)
	if err != nil {
		return nil, err
	}

	size, err := log.Seek(0, os.SEEK_END)
	if err != nil {
		return nil, err
	}

	return &LogFile{
		f:             log,
		capacity:      capacity,
		currentSize:   size,
		maxFiles:      maxFiles,
		notifyRotate:  pubsub.NewPublisher(0, 1),
		marshal:       marshaller,
		createDecoder: decodeFunc,
	}, nil
}

// WriteLogEntry writes the provided log message to the current log file.
// This may trigger a rotation event if the max file/capacity limits are hit.
func (w *LogFile) WriteLogEntry(msg *logger.Message) error {
	b, err := w.marshal(msg)
	if err != nil {
		return errors.Wrap(err, "error marshalling log message")
	}

	logger.PutMessage(msg)

	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return errors.New("cannot write because the output file was closed")
	}

	if err := w.checkCapacityAndRotate(); err != nil {
		w.mu.Unlock()
		return err
	}

	n, err := w.f.Write(b)
	if err == nil {
		w.currentSize += int64(n)
	}
	w.mu.Unlock()
	return err
}

func (w *LogFile) checkCapacityAndRotate() error {
	if w.capacity == -1 {
		return nil
	}

	if w.currentSize >= w.capacity {
		name := w.f.Name()
		if err := w.f.Close(); err != nil {
			return errors.Wrap(err, "error closing file")
		}
		if err := rotate(name, w.maxFiles); err != nil {
			return err
		}
		file, err := os.OpenFile(name, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0640)
		if err != nil {
			return err
		}
		w.f = file
		w.currentSize = 0
		w.notifyRotate.Publish(struct{}{})
	}

	return nil
}

func rotate(name string, maxFiles int) error {
	if maxFiles < 2 {
		return nil
	}
	for i := maxFiles - 1; i > 1; i-- {
		toPath := name + "." + strconv.Itoa(i)
		fromPath := name + "." + strconv.Itoa(i-1)
		if err := os.Rename(fromPath, toPath); err != nil && !os.IsNotExist(err) {
			return errors.Wrap(err, "error rotating old log entries")
		}
	}

	if err := os.Rename(name, name+".1"); err != nil && !os.IsNotExist(err) {
		return errors.Wrap(err, "error rotating current log")
	}
	return nil
}

// LogPath returns the location the given writer logs to.
func (w *LogFile) LogPath() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.f.Name()
}

// MaxFiles return maximum number of files
func (w *LogFile) MaxFiles() int {
	return w.maxFiles
}

// Close closes underlying file and signals all readers to stop.
func (w *LogFile) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	if err := w.f.Close(); err != nil {
		return err
	}
	w.closed = true
	return nil
}

// ReadLogs decodes entries from log files and sends them the passed in watcher
func (w *LogFile) ReadLogs(config logger.ReadConfig, watcher *logger.LogWatcher) {
	w.mu.RLock()
	files, err := w.openRotatedFiles()
	if err != nil {
		w.mu.RUnlock()
		watcher.Err <- err
		return
	}
	defer func() {
		for _, f := range files {
			f.Close()
		}
	}()

	currentFile, err := os.Open(w.f.Name())
	if err != nil {
		w.mu.RUnlock()
		watcher.Err <- err
		return
	}
	defer currentFile.Close()

	currentChunk, err := newSectionReader(currentFile)
	w.mu.RUnlock()

	if err != nil {
		watcher.Err <- err
		return
	}

	if config.Tail != 0 {
		seekers := make([]io.ReadSeeker, 0, len(files)+1)
		for _, f := range files {
			seekers = append(seekers, f)
		}
		seekers = append(seekers, currentChunk)
		tailFile(multireader.MultiReadSeeker(seekers...), watcher, w.createDecoder, config)
	}

	w.mu.RLock()
	if !config.Follow || w.closed {
		w.mu.RUnlock()
		return
	}
	w.mu.RUnlock()

	notifyRotate := w.notifyRotate.Subscribe()
	defer w.notifyRotate.Evict(notifyRotate)
	followLogs(currentFile, watcher, notifyRotate, w.createDecoder, config.Since, config.Until)
}

func (w *LogFile) openRotatedFiles() (files []*os.File, err error) {
	defer func() {
		if err == nil {
			return
		}
		for _, f := range files {
			f.Close()
		}
	}()

	for i := w.maxFiles; i > 1; i-- {
		f, err := os.Open(fmt.Sprintf("%s.%d", w.f.Name(), i-1))
		if err != nil {
			if !os.IsNotExist(err) {
				return nil, err
			}
			continue
		}
		files = append(files, f)
	}

	return files, nil
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

type decodeFunc func() (*logger.Message, error)

func tailFile(f io.ReadSeeker, watcher *logger.LogWatcher, createDecoder makeDecoderFunc, config logger.ReadConfig) {
	var rdr io.Reader = f
	if config.Tail > 0 {
		ls, err := tailfile.TailFile(f, config.Tail)
		if err != nil {
			watcher.Err <- err
			return
		}
		rdr = bytes.NewBuffer(bytes.Join(ls, []byte("\n")))
	}

	decodeLogLine := createDecoder(rdr)
	for {
		msg, err := decodeLogLine()
		if err != nil {
			if err != io.EOF {
				watcher.Err <- err
			}
			return
		}
		if !config.Since.IsZero() && msg.Timestamp.Before(config.Since) {
			continue
		}
		if !config.Until.IsZero() && msg.Timestamp.After(config.Until) {
			return
		}
		select {
		case <-watcher.WatchClose():
			return
		case watcher.Msg <- msg:
		}
	}
}

func followLogs(f *os.File, logWatcher *logger.LogWatcher, notifyRotate chan interface{}, createDecoder makeDecoderFunc, since, until time.Time) {
	decodeLogLine := createDecoder(f)

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
		decodeLogLine = createDecoder(f)
		return nil
	}

	errRetry := errors.New("retry")
	errDone := errors.New("done")
	waitRead := func() error {
		select {
		case e := <-fileWatcher.Events():
			switch e.Op {
			case fsnotify.Write:
				decodeLogLine = createDecoder(f)
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
		if err != io.EOF {
			return err
		}

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

	// main loop
	for {
		msg, err := decodeLogLine()
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
		if !until.IsZero() && msg.Timestamp.After(until) {
			return
		}
		select {
		case logWatcher.Msg <- msg:
		case <-ctx.Done():
			logWatcher.Msg <- msg
			for {
				msg, err := decodeLogLine()
				if err != nil {
					return
				}
				if !since.IsZero() && msg.Timestamp.Before(since) {
					continue
				}
				if !until.IsZero() && msg.Timestamp.After(until) {
					return
				}
				logWatcher.Msg <- msg
			}
		}
	}
}

func watchFile(name string) (filenotify.FileWatcher, error) {
	fileWatcher, err := filenotify.New()
	if err != nil {
		return nil, err
	}

	logger := logrus.WithFields(logrus.Fields{
		"module": "logger",
		"fille":  name,
	})

	if err := fileWatcher.Add(name); err != nil {
		logger.WithError(err).Warnf("falling back to file poller")
		fileWatcher.Close()
		fileWatcher = filenotify.NewPollingWatcher()

		if err := fileWatcher.Add(name); err != nil {
			fileWatcher.Close()
			logger.WithError(err).Debugf("error watching log file for modifications")
			return nil, err
		}
	}
	return fileWatcher, nil
}
