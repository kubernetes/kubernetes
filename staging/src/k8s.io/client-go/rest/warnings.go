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
	"context"
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

// WarningHandlerWithContext is an interface for handling warning headers with
// support for contextual logging.
type WarningHandlerWithContext interface {
	// HandleWarningHeaderWithContext is called with the warn code, agent, and text when a warning header is countered.
	HandleWarningHeaderWithContext(ctx context.Context, code int, agent string, text string)
}

var (
	defaultWarningHandler     WarningHandlerWithContext = WarningLogger{}
	defaultWarningHandlerLock sync.RWMutex
)

// SetDefaultWarningHandler sets the default handler clients use when warning headers are encountered.
// By default, warnings are logged. Several built-in implementations are provided:
//   - NoWarnings suppresses warnings.
//   - WarningLogger logs warnings.
//   - NewWarningWriter() outputs warnings to the provided writer.
//
// logcheck:context // SetDefaultWarningHandlerWithContext should be used instead of SetDefaultWarningHandler in code which supports contextual logging.
func SetDefaultWarningHandler(l WarningHandler) {
	if l == nil {
		SetDefaultWarningHandlerWithContext(nil)
		return
	}
	SetDefaultWarningHandlerWithContext(warningLoggerNopContext{l: l})
}

// SetDefaultWarningHandlerWithContext is a variant of [SetDefaultWarningHandler] which supports contextual logging.
func SetDefaultWarningHandlerWithContext(l WarningHandlerWithContext) {
	defaultWarningHandlerLock.Lock()
	defer defaultWarningHandlerLock.Unlock()
	defaultWarningHandler = l
}

func getDefaultWarningHandler() WarningHandlerWithContext {
	defaultWarningHandlerLock.RLock()
	defer defaultWarningHandlerLock.RUnlock()
	l := defaultWarningHandler
	return l
}

type warningLoggerNopContext struct {
	l WarningHandler
}

func (w warningLoggerNopContext) HandleWarningHeaderWithContext(_ context.Context, code int, agent string, message string) {
	w.l.HandleWarningHeader(code, agent, message)
}

// NoWarnings is an implementation of [WarningHandler] and [WarningHandlerWithContext] that suppresses warnings.
type NoWarnings struct{}

func (NoWarnings) HandleWarningHeader(code int, agent string, message string) {}
func (NoWarnings) HandleWarningHeaderWithContext(ctx context.Context, code int, agent string, message string) {
}

var _ WarningHandler = NoWarnings{}
var _ WarningHandlerWithContext = NoWarnings{}

// WarningLogger is an implementation of [WarningHandler] and [WarningHandlerWithContext] that logs code 299 warnings
type WarningLogger struct{}

func (WarningLogger) HandleWarningHeader(code int, agent string, message string) {
	if code != 299 || len(message) == 0 {
		return
	}
	klog.Background().Info("Warning: " + message)
}

func (WarningLogger) HandleWarningHeaderWithContext(ctx context.Context, code int, agent string, message string) {
	if code != 299 || len(message) == 0 {
		return
	}
	klog.FromContext(ctx).Info("Warning: " + message)
}

var _ WarningHandler = WarningLogger{}
var _ WarningHandlerWithContext = WarningLogger{}

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

func handleWarnings(ctx context.Context, headers http.Header, handler WarningHandlerWithContext) []net.WarningHeader {
	if handler == nil {
		handler = getDefaultWarningHandler()
	}

	warnings, _ := net.ParseWarningHeaders(headers["Warning"])
	for _, warning := range warnings {
		handler.HandleWarningHeaderWithContext(ctx, warning.Code, warning.Agent, warning.Text)
	}
	return warnings
}
