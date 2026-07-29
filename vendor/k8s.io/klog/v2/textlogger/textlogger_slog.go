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

package textlogger

import (
	"context"
	"log/slog"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/sloghandler"
)

func (l *tlogger) Handle(ctx context.Context, record slog.Record) error {
	return sloghandler.Handle(ctx, record, l.groups, l.printWithInfos)
}

func (l *tlogger) WithAttrs(attrs []slog.Attr) logr.SlogSink {
	clone := *l
	clone.values = serialize.WithValues(l.values, sloghandler.Attrs2KVList(l.groups, attrs))
	return &clone
}

func (l *tlogger) WithGroup(name string) logr.SlogSink {
	clone := *l
	if clone.groups != "" {
		clone.groups += "." + name
	} else {
		clone.groups = name
	}
	return &clone
}

var _ logr.SlogSink = &tlogger{}
