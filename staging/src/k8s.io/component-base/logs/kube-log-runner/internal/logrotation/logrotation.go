/*
Copyright 2025 The Kubernetes Authors.

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

package logrotation

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const timeLayout string = "20060102-150405"

type rotationFile struct {
	// required, the max size of the log file in bytes, 0 means no rotation
	maxSize int64
	// required, the max age of the log file, 0 means no cleanup
	maxAge        time.Duration
	filePath      string
	mut           sync.Mutex
	file          *os.File
	currentSize   int64
	lasSyncTime   time.Time
	flushInterval time.Duration
	// set after a failed rotation so that retries on every write log only once
	rotationFailing bool
}

func Open(filePath string, flushInterval time.Duration, maxSize int64, maxAge time.Duration) (io.WriteCloser, error) {
	w := &rotationFile{
		filePath:      filePath,
		maxSize:       maxSize,
		maxAge:        maxAge,
		flushInterval: flushInterval,
	}

	logFile, err := os.OpenFile(w.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}

	w.file = logFile

	w.lasSyncTime = time.Now()

	if w.maxSize > 0 {
		info, err := os.Stat(w.filePath)
		if err != nil {
			return nil, err
		}
		w.currentSize = info.Size()
	}

	return w, nil
}

// Write implements the io.Writer interface.
func (w *rotationFile) Write(p []byte) (n int, err error) {
	w.mut.Lock()
	defer w.mut.Unlock()

	n, err = w.file.Write(p)
	if err != nil {
		return 0, err
	}

	if w.flushInterval > 0 && time.Since(w.lasSyncTime) >= w.flushInterval {
		err = w.file.Sync()
		if err != nil {
			return 0, err
		}
		w.lasSyncTime = time.Now()
	}

	if w.maxSize > 0 {
		w.currentSize += int64(len(p))

		// if file size over maxsize rotate the log file
		if w.currentSize >= w.maxSize {
			// p is already written, so a failed rotation must not fail the
			// write: os/exec stops copying on the first error, which breaks
			// the pipe and kills the wrapped process with SIGPIPE. Keep
			// appending to the current file and retry on the next write.
			if err := w.rotate(); err != nil {
				if !w.rotationFailing {
					log.Printf("Failed to rotate log file %q, will retry on next write: %v", w.filePath, err)
				}
				w.rotationFailing = true
			} else {
				w.rotationFailing = false
			}
		}
	}

	return n, nil
}

func (w *rotationFile) rotate() error {
	// Get the file extension
	ext := filepath.Ext(w.filePath)

	// Remove the extension from the filename
	pathWithoutExt := strings.TrimSuffix(w.filePath, ext)

	timestamp := time.Now().Format(timeLayout)
	rotateFilePath := pathWithoutExt + "-" + timestamp + ext

	if w.filePath == rotateFilePath {
		return nil
	}

	// Rotating more than once per second would otherwise rename over, and
	// silently discard, the previous backup. clean() still matches the ".N"
	// suffix because time.Parse accepts it as fractional seconds.
	for i := 1; ; i++ {
		if _, err := os.Lstat(rotateFilePath); err != nil {
			break
		}
		rotateFilePath = fmt.Sprintf("%s-%s.%d%s", pathWithoutExt, timestamp, i, ext)
	}

	closeErr := w.file.Close()
	renameErr := os.Rename(w.filePath, rotateFilePath)

	// Reopen even if the rename failed, otherwise w.file stays closed and
	// every later write fails with "file already closed".
	var err error
	w.file, err = os.OpenFile(w.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	if renameErr != nil {
		return renameErr
	}
	if closeErr != nil {
		log.Printf("Failed to close log file %q before rotation: %v", w.filePath, closeErr)
	}

	w.currentSize = 0

	if w.maxAge > 0 {
		go func() {
			err = w.clean(pathWithoutExt, ext)
		}()
	}

	return nil
}

// Clean up the old log files in the format of
// <basename>-<timestamp><ext>.
// This should be safe enough to avoid false deletion.
// This will work for multiple restarts of the same program.
func (w *rotationFile) clean(pathWithoutExt string, ext string) error {
	ageTime := time.Now().Add(-w.maxAge)

	directory := filepath.Dir(pathWithoutExt)
	basename := filepath.Base(pathWithoutExt) + "-"

	dir, err := os.ReadDir(directory)
	if err != nil {
		return err
	}

	err = nil
	for _, v := range dir {
		if strings.HasPrefix(v.Name(), basename) && strings.HasSuffix(v.Name(), ext) {
			// Remove the prefix and suffix
			trimmed := strings.TrimPrefix(v.Name(), basename)
			trimmed = strings.TrimSuffix(trimmed, ext)

			_, err = time.Parse(timeLayout, trimmed)
			if err == nil {
				info, errInfo := v.Info()
				if errInfo != nil {
					err = errInfo
					// Ignore the error while continue with the next clenup
					continue
				}

				if ageTime.After(info.ModTime()) {
					err = os.Remove(filepath.Join(directory, v.Name()))
					if err != nil {
						// Ignore the error while continue with the next clenup
						continue
					}
				}
			}

		}
	}

	return err
}

func (w *rotationFile) Close() error {
	w.mut.Lock()
	defer w.mut.Unlock()

	// Explicitly call file.Sync() to ensure data is written to disk
	err := w.file.Sync()
	if err != nil {
		return err
	}

	err = w.file.Close()
	if err != nil {
		return err
	}

	return nil
}
