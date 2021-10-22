/*
Copyright 2019 The logr Authors.

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

package main

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/go-logr/logr"
)

// TabLogger is a sample logr.Logger that logs to stderr.
// It's terribly inefficient, and is *only* a basic example.
type TabLogger struct {
	name      string
	keyValues map[string]interface{}

	writer *tabwriter.Writer
}

var _ logr.Logger = &TabLogger{}

func (l *TabLogger) Info(msg string, kvs ...interface{}) {
	fmt.Fprintf(l.writer, "%s\t%s\t", l.name, msg)
	for k, v := range l.keyValues {
		fmt.Fprintf(l.writer, "%s: %+v  ", k, v)
	}
	for i := 0; i < len(kvs); i += 2 {
		fmt.Fprintf(l.writer, "%s: %+v  ", kvs[i], kvs[i+1])
	}
	fmt.Fprintf(l.writer, "\n")
	l.writer.Flush()
}

func (_ *TabLogger) Enabled() bool {
	return true
}

func (l *TabLogger) Error(err error, msg string, kvs ...interface{}) {
	kvs = append(kvs, "error", err)
	l.Info(msg, kvs...)
}

func (l *TabLogger) V(_ int) logr.Logger {
	return l
}

func (l *TabLogger) WithName(name string) logr.Logger {
	return &TabLogger{
		name:      l.name + "." + name,
		keyValues: l.keyValues,
		writer:    l.writer,
	}
}

func (l *TabLogger) WithValues(kvs ...interface{}) logr.Logger {
	newMap := make(map[string]interface{}, len(l.keyValues)+len(kvs)/2)
	for k, v := range l.keyValues {
		newMap[k] = v
	}
	for i := 0; i < len(kvs); i += 2 {
		newMap[kvs[i].(string)] = kvs[i+1]
	}
	return &TabLogger{
		name:      l.name,
		keyValues: newMap,
		writer:    l.writer,
	}
}

func NewTabLogger() logr.Logger {
	return &TabLogger{
		writer: tabwriter.NewWriter(os.Stderr, 40, 8, 2, '\t', 0),
	}
}
