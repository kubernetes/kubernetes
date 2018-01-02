// Copyright 2016 CoreOS Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package progressutil

import (
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

var (
	ErrAlreadyStarted = errors.New("cannot add copies after PrintAndWait has been called")
)

type copyReader struct {
	reader  io.Reader
	current int64
	total   int64
	pb      *ProgressBar
}

func (cr *copyReader) Read(p []byte) (int, error) {
	n, err := cr.reader.Read(p)
	cr.current += int64(n)
	err1 := cr.updateProgressBar()
	if err == nil {
		err = err1
	}
	return n, err
}

func (cr *copyReader) updateProgressBar() error {
	cr.pb.SetPrintAfter(cr.formattedProgress())

	progress := float64(cr.current) / float64(cr.total)
	if progress > 1 {
		progress = 1
	}
	return cr.pb.SetCurrentProgress(progress)
}

// NewCopyProgressPrinter returns a new CopyProgressPrinter
func NewCopyProgressPrinter() *CopyProgressPrinter {
	return &CopyProgressPrinter{results: make(chan error), cancel: make(chan struct{})}
}

// CopyProgressPrinter will perform an arbitrary number of io.Copy calls, while
// continually printing the progress of each copy.
type CopyProgressPrinter struct {
	results chan error
	cancel  chan struct{}

	// `lock` mutex protects all fields below it in CopyProgressPrinter struct
	lock    sync.Mutex
	readers []*copyReader
	started bool
	pbp     *ProgressBarPrinter
}

// AddCopy adds a copy for this CopyProgressPrinter to perform. An io.Copy call
// will be made to copy bytes from reader to dest, and name and size will be
// used to label the progress bar and display how much progress has been made.
// If size is 0, the total size of the reader is assumed to be unknown.
// AddCopy can only be called before PrintAndWait; otherwise, ErrAlreadyStarted
// will be returned.
func (cpp *CopyProgressPrinter) AddCopy(reader io.Reader, name string, size int64, dest io.Writer) error {
	cpp.lock.Lock()
	defer cpp.lock.Unlock()

	if cpp.started {
		return ErrAlreadyStarted
	}
	if cpp.pbp == nil {
		cpp.pbp = &ProgressBarPrinter{}
		cpp.pbp.PadToBeEven = true
	}

	cr := &copyReader{
		reader:  reader,
		current: 0,
		total:   size,
		pb:      cpp.pbp.AddProgressBar(),
	}
	cr.pb.SetPrintBefore(name)
	cr.pb.SetPrintAfter(cr.formattedProgress())

	cpp.readers = append(cpp.readers, cr)

	go func() {
		_, err := io.Copy(dest, cr)
		select {
		case <-cpp.cancel:
			return
		case cpp.results <- err:
			return
		}
	}()
	return nil
}

// PrintAndWait will print the progress for each copy operation added with
// AddCopy to printTo every printInterval. This will continue until every added
// copy is finished, or until cancel is written to.
// PrintAndWait may only be called once; any subsequent calls will immediately
// return ErrAlreadyStarted.  After PrintAndWait has been called, no more
// copies may be added to the CopyProgressPrinter.
func (cpp *CopyProgressPrinter) PrintAndWait(printTo io.Writer, printInterval time.Duration, cancel chan struct{}) error {
	cpp.lock.Lock()
	if cpp.started {
		cpp.lock.Unlock()
		return ErrAlreadyStarted
	}
	cpp.started = true
	cpp.lock.Unlock()

	n := len(cpp.readers)
	if n == 0 {
		// Nothing to do.
		return nil
	}

	defer close(cpp.cancel)
	t := time.NewTicker(printInterval)
	allDone := false
	for i := 0; i < n; {
		select {
		case <-cancel:
			return nil
		case <-t.C:
			_, err := cpp.pbp.Print(printTo)
			if err != nil {
				return err
			}
		case err := <-cpp.results:
			i++
			// Once completion is signaled, further on this just drains
			// (unlikely) errors from the channel.
			if err == nil && !allDone {
				allDone, err = cpp.pbp.Print(printTo)
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (cr *copyReader) formattedProgress() string {
	var totalStr string
	if cr.total == 0 {
		totalStr = "?"
	} else {
		totalStr = ByteUnitStr(cr.total)
	}
	return fmt.Sprintf("%s / %s", ByteUnitStr(cr.current), totalStr)
}

var byteUnits = []string{"B", "KB", "MB", "GB", "TB", "PB"}

// ByteUnitStr pretty prints a number of bytes.
func ByteUnitStr(n int64) string {
	var unit string
	size := float64(n)
	for i := 1; i < len(byteUnits); i++ {
		if size < 1000 {
			unit = byteUnits[i-1]
			break
		}

		size = size / 1000
	}

	return fmt.Sprintf("%.3g %s", size, unit)
}
