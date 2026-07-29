//go:build go1.21
// +build go1.21

/*
Copyright 2023 The logr Authors.

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

package funcr

import (
	"context"
	"log/slog"

	"github.com/go-logr/logr"
)

var _ logr.SlogSink = &fnlogger{}

const extraSlogSinkDepth = 3 // 2 for slog, 1 for SlogSink

func (l fnlogger) Handle(_ context.Context, record slog.Record) error {
	kvList := make([]any, 0, 2*record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		kvList = attrToKVs(attr, kvList)
		return true
	})

	if record.Level >= slog.LevelError {
		l.WithCallDepth(extraSlogSinkDepth).Error(nil, record.Message, kvList...)
	} else {
		level := l.levelFromSlog(record.Level)
		l.WithCallDepth(extraSlogSinkDepth).Info(level, record.Message, kvList...)
	}
	return nil
}

func (l fnlogger) WithAttrs(attrs []slog.Attr) logr.SlogSink {
	kvList := make([]any, 0, 2*len(attrs))
	for _, attr := range attrs {
		kvList = attrToKVs(attr, kvList)
	}
	l.AddValues(kvList)
	return &l
}

func (l fnlogger) WithGroup(name string) logr.SlogSink {
	l.startGroup(name)
	return &l
}

// attrToKVs appends a slog.Attr to a logr-style kvList.  It handle slog Groups
// and other details of slog.
func attrToKVs(attr slog.Attr, kvList []any) []any {
	attrVal := attr.Value.Resolve()
	if attrVal.Kind() == slog.KindGroup {
		groupVal := attrVal.Group()
		grpKVs := make([]any, 0, 2*len(groupVal))
		for _, attr := range groupVal {
			grpKVs = attrToKVs(attr, grpKVs)
		}
		if attr.Key == "" {
			// slog says we have to inline these
			kvList = append(kvList, grpKVs...)
		} else {
			kvList = append(kvList, attr.Key, PseudoStruct(grpKVs))
		}
	} else if attr.Key != "" {
		kvList = append(kvList, attr.Key, attrVal.Any())
	}

	return kvList
}

// levelFromSlog adjusts the level by the logger's verbosity and negates it.
// It ensures that the result is >= 0. This is necessary because the result is
// passed to a LogSink and that API did not historically document whether
// levels could be negative or what that meant.
//
// Some example usage:
//
//	logrV0 := getMyLogger()
//	logrV2 := logrV0.V(2)
//	slogV2 := slog.New(logr.ToSlogHandler(logrV2))
//	slogV2.Debug("msg") // =~ logrV2.V(4) =~ logrV0.V(6)
//	slogV2.Info("msg")  // =~  logrV2.V(0) =~ logrV0.V(2)
//	slogv2.Warn("msg")  // =~ logrV2.V(-4) =~ logrV0.V(0)
func (l fnlogger) levelFromSlog(level slog.Level) int {
	result := -level
	if result < 0 {
		result = 0 // because LogSink doesn't expect negative V levels
	}
	return int(result)
}
