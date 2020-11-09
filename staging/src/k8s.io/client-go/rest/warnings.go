/*
Copyright 2020 The Kubernetes Authors.

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

package rest

import (
	"fmt"
	"io"
	"net/http"
	"sync"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/net"
)

// WarningHandler is an interface for handling warning headers
type WarningHandler interface {
	// HandleWarningHeader is called with the warn code, agent, and text when a warning header is countered.
	HandleWarningHeader(code int, agent string, text string)
}

var (
	defaultWarningHandler     WarningHandler = WarningLogger{}
	defaultWarningHandlerLock sync.RWMutex
)

// SetDefaultWarningHandler sets the default handler clients use when warning headers are encountered.
// By default, warnings are logged. Several built-in implementations are provided:
//  - NoWarnings suppresses warnings.
//  - WarningLogger logs warnings.
//  - NewWarningWriter() outputs warnings to the provided writer.
func SetDefaultWarningHandler(l WarningHandler) {
	defaultWarningHandlerLock.Lock()
	defer defaultWarningHandlerLock.Unlock()
	defaultWarningHandler = l
}
func getDefaultWarningHandler() WarningHandler {
	defaultWarningHandlerLock.RLock()
	defer defaultWarningHandlerLock.RUnlock()
	l := defaultWarningHandler
	return l
}

// NoWarnings is an implementation of WarningHandler that suppresses warnings.
type NoWarnings struct{}

func (NoWarnings) HandleWarningHeader(code int, agent string, message string) {}

// WarningLogger is an implementation of WarningHandler that logs code 299 warnings
type WarningLogger struct{}

func (WarningLogger) HandleWarningHeader(code int, agent string, message string) {
	if code != 299 || len(message) == 0 {
		return
	}
	klog.Warning(message)
}

type warningWriter struct {
	// out is the writer to output warnings to
	out io.Writer
	// opts contains options controlling warning output
	opts WarningWriterOptions
	// writtenLock guards written and writtenCount
	writtenLock  sync.Mutex
	writtenCount int
	written      map[string]struct{}
}

// WarningWriterOptions controls the behavior of a WarningHandler constructed using NewWarningWriter()
type WarningWriterOptions struct {
	// Deduplicate indicates a given warning message should only be written once.
	// Setting this to true in a long-running process handling many warnings can result in increased memory use.
	Deduplicate bool
	// Color indicates that warning output can include ANSI color codes
	Color bool
}

// NewWarningWriter returns an implementation of WarningHandler that outputs code 299 warnings to the specified writer.
func NewWarningWriter(out io.Writer, opts WarningWriterOptions) *warningWriter {
	h := &warningWriter{out: out, opts: opts}
	if opts.Deduplicate {
		h.written = map[string]struct{}{}
	}
	return h
}

const (
	yellowColor = "\u001b[33;1m"
	resetColor  = "\u001b[0m"
)

// HandleWarningHeader prints warnings with code=299 to the configured writer.
func (w *warningWriter) HandleWarningHeader(code int, agent string, message string) {
	if code != 299 || len(message) == 0 {
		return
	}

	w.writtenLock.Lock()
	defer w.writtenLock.Unlock()

	if w.opts.Deduplicate {
		if _, alreadyWritten := w.written[message]; alreadyWritten {
			return
		}
		w.written[message] = struct{}{}
	}
	w.writtenCount++

	if w.opts.Color {
		fmt.Fprintf(w.out, "%sWarning:%s %s\n", yellowColor, resetColor, message)
	} else {
		fmt.Fprintf(w.out, "Warning: %s\n", message)
	}
}

func (w *warningWriter) WarningCount() int {
	w.writtenLock.Lock()
	defer w.writtenLock.Unlock()
	return w.writtenCount
}

func handleWarnings(headers http.Header, handler WarningHandler) []net.WarningHeader {
	if handler == nil {
		handler = getDefaultWarningHandler()
	}

	warnings, _ := net.ParseWarningHeaders(headers["Warning"])
	for _, warning := range warnings {
		handler.HandleWarningHeader(warning.Code, warning.Agent, warning.Text)
	}
	return warnings
}
