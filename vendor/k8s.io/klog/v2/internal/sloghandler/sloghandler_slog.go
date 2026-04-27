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

package sloghandler

import (
	"context"
	"log/slog"
	"runtime"
	"strings"
	"time"

	"k8s.io/klog/v2/internal/severity"
)

func Handle(_ context.Context, record slog.Record, groups string, printWithInfos func(file string, line int, now time.Time, err error, s severity.Severity, msg string, kvList []interface{})) error {
	now := record.Time
	if now.IsZero() {
		// This format doesn't support printing entries without a time.
		now = time.Now()
	}

	// slog has numeric severity levels, with 0 as default "info", negative for debugging, and
	// positive with some pre-defined levels for more important. Those ranges get mapped to
	// the corresponding klog levels where possible, with "info" the default that is used
	// also for negative debug levels.
	level := record.Level
	s := severity.InfoLog
	switch {
	case level >= slog.LevelError:
		s = severity.ErrorLog
	case level >= slog.LevelWarn:
		s = severity.WarningLog
	}

	var file string
	var line int
	if record.PC != 0 {
		// Same as https://cs.opensource.google/go/x/exp/+/642cacee:slog/record.go;drc=642cacee5cc05231f45555a333d07f1005ffc287;l=70
		fs := runtime.CallersFrames([]uintptr{record.PC})
		f, _ := fs.Next()
		if f.File != "" {
			file = f.File
			if slash := strings.LastIndex(file, "/"); slash >= 0 {
				file = file[slash+1:]
			}
			line = f.Line
		}
	} else {
		file = "???"
		line = 1
	}

	kvList := make([]interface{}, 0, 2*record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		kvList = appendAttr(groups, kvList, attr)
		return true
	})

	printWithInfos(file, line, now, nil, s, record.Message, kvList)
	return nil
}

func Attrs2KVList(groups string, attrs []slog.Attr) []interface{} {
	kvList := make([]interface{}, 0, 2*len(attrs))
	for _, attr := range attrs {
		kvList = appendAttr(groups, kvList, attr)
	}
	return kvList
}

func appendAttr(groups string, kvList []interface{}, attr slog.Attr) []interface{} {
	var key string
	if groups != "" {
		key = groups + "." + attr.Key
	} else {
		key = attr.Key
	}
	return append(kvList, key, attr.Value)
}
