//go:build go1.21
// +build go1.21

/*
Copyright 2023 The Kubernetes Authors.

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

package klog

import (
	"context"
	"log/slog"
	"strconv"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/buffer"
	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/severity"
	"k8s.io/klog/v2/internal/sloghandler"
)

func (l *klogger) Handle(ctx context.Context, record slog.Record) error {
	if logging.logger != nil {
		if slogSink, ok := logging.logger.GetSink().(logr.SlogSink); ok {
			// Let that logger do the work.
			return slogSink.Handle(ctx, record)
		}
	}

	return sloghandler.Handle(ctx, record, l.groups, slogOutput)
}

// slogOutput corresponds to several different functions in klog.go.
// It goes through some of the same checks and formatting steps before
// it ultimately converges by calling logging.printWithInfos.
func slogOutput(file string, line int, now time.Time, err error, s severity.Severity, msg string, kvList []interface{}) {
	// See infoS.
	if logging.logger != nil {
		// Taking this path happens when klog has a logger installed
		// as backend which doesn't support slog. Not good, we have to
		// guess about the call depth and drop the actual location.
		logger := logging.logger.WithCallDepth(2)
		if s > severity.ErrorLog {
			logger.Error(err, msg, kvList...)
		} else {
			logger.Info(msg, kvList...)
		}
		return
	}

	// See printS.
	b := buffer.GetBuffer()
	b.WriteString(strconv.Quote(msg))
	if err != nil {
		serialize.KVListFormat(&b.Buffer, "err", err)
	}
	serialize.KVListFormat(&b.Buffer, kvList...)

	// See print + header.
	buf := logging.formatHeader(s, file, line, now)
	logging.printWithInfos(buf, file, line, s, nil, nil, 0, &b.Buffer)

	buffer.PutBuffer(b)
}

func (l *klogger) WithAttrs(attrs []slog.Attr) logr.SlogSink {
	clone := *l
	clone.values = serialize.WithValues(l.values, sloghandler.Attrs2KVList(l.groups, attrs))
	return &clone
}

func (l *klogger) WithGroup(name string) logr.SlogSink {
	clone := *l
	if clone.groups != "" {
		clone.groups += "." + name
	} else {
		clone.groups = name
	}
	return &clone
}

var _ logr.SlogSink = &klogger{}
