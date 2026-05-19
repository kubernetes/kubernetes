// Copyright (c) 2016-2022 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zap

import (
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"go.uber.org/zap/zapcore"
)

const schemeFile = "file"

var _sinkRegistry = newSinkRegistry()

// Sink defines the interface to write to and close logger destinations.
type Sink interface {
	zapcore.WriteSyncer
	io.Closer
}

type errSinkNotFound struct {
	scheme string
}

func (e *errSinkNotFound) Error() string {
	return fmt.Sprintf("no sink found for scheme %q", e.scheme)
}

type nopCloserSink struct{ zapcore.WriteSyncer }

func (nopCloserSink) Close() error { return nil }

type sinkRegistry struct {
	mu        sync.Mutex
	factories map[string]func(*url.URL) (Sink, error)          // keyed by scheme
	openFile  func(string, int, os.FileMode) (*os.File, error) // type matches os.OpenFile
}

func newSinkRegistry() *sinkRegistry {
	sr := &sinkRegistry{
		factories: make(map[string]func(*url.URL) (Sink, error)),
		openFile:  os.OpenFile,
	}
	// Infallible operation: the registry is empty, so we can't have a conflict.
	_ = sr.RegisterSink(schemeFile, sr.newFileSinkFromURL)
	return sr
}

// RegisterScheme registers the given factory for the specific scheme.
func (sr *sinkRegistry) RegisterSink(scheme string, factory func(*url.URL) (Sink, error)) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if scheme == "" {
		return errors.New("can't register a sink factory for empty string")
	}
	normalized, err := normalizeScheme(scheme)
	if err != nil {
		return fmt.Errorf("%q is not a valid scheme: %v", scheme, err)
	}
	if _, ok := sr.factories[normalized]; ok {
		return fmt.Errorf("sink factory already registered for scheme %q", normalized)
	}
	sr.factories[normalized] = factory
	return nil
}

func (sr *sinkRegistry) newSink(rawURL string) (Sink, error) {
	// URL parsing doesn't work well for Windows paths such as `c:\log.txt`, as scheme is set to
	// the drive, and path is unset unless `c:/log.txt` is used.
	// To avoid Windows-specific URL handling, we instead check IsAbs to open as a file.
	// filepath.IsAbs is OS-specific, so IsAbs('c:/log.txt') is false outside of Windows.
	if filepath.IsAbs(rawURL) {
		return sr.newFileSinkFromPath(rawURL)
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("can't parse %q as a URL: %v", rawURL, err)
	}
	if u.Scheme == "" {
		u.Scheme = schemeFile
	}

	sr.mu.Lock()
	factory, ok := sr.factories[u.Scheme]
	sr.mu.Unlock()
	if !ok {
		return nil, &errSinkNotFound{u.Scheme}
	}
	return factory(u)
}

// RegisterSink registers a user-supplied factory for all sinks with a
// particular scheme.
//
// All schemes must be ASCII, valid under section 0.1 of RFC 3986
// (https://tools.ietf.org/html/rfc3983#section-3.1), and must not already
// have a factory registered. Zap automatically registers a factory for the
// "file" scheme.
func RegisterSink(scheme string, factory func(*url.URL) (Sink, error)) error {
	return _sinkRegistry.RegisterSink(scheme, factory)
}

func (sr *sinkRegistry) newFileSinkFromURL(u *url.URL) (Sink, error) {
	if u.User != nil {
		return nil, fmt.Errorf("user and password not allowed with file URLs: got %v", u)
	}
	if u.Fragment != "" {
		return nil, fmt.Errorf("fragments not allowed with file URLs: got %v", u)
	}
	if u.RawQuery != "" {
		return nil, fmt.Errorf("query parameters not allowed with file URLs: got %v", u)
	}
	// Error messages are better if we check hostname and port separately.
	if u.Port() != "" {
		return nil, fmt.Errorf("ports not allowed with file URLs: got %v", u)
	}
	if hn := u.Hostname(); hn != "" && hn != "localhost" {
		return nil, fmt.Errorf("file URLs must leave host empty or use localhost: got %v", u)
	}

	return sr.newFileSinkFromPath(u.Path)
}

func (sr *sinkRegistry) newFileSinkFromPath(path string) (Sink, error) {
	switch path {
	case "stdout":
		return nopCloserSink{os.Stdout}, nil
	case "stderr":
		return nopCloserSink{os.Stderr}, nil
	}
	return sr.openFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o666)
}

func normalizeScheme(s string) (string, error) {
	// https://tools.ietf.org/html/rfc3986#section-3.1
	s = strings.ToLower(s)
	if first := s[0]; 'a' > first || 'z' < first {
		return "", errors.New("must start with a letter")
	}
	for i := 1; i < len(s); i++ { // iterate over bytes, not runes
		c := s[i]
		switch {
		case 'a' <= c && c <= 'z':
			continue
		case '0' <= c && c <= '9':
			continue
		case c == '.' || c == '+' || c == '-':
			continue
		}
		return "", fmt.Errorf("may not contain %q", c)
	}
	return s, nil
}
