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

package printers

import (
	"encoding/csv"
	"strings"

	"github.com/liggitt/tabwriter"
)

// writer could be a tabwriter or csvwriter.
type writer interface {
	Write([]string) error
	Flush() error
}

var _ writer = (*tabWriter)(nil)
var _ writer = (*csvWriter)(nil)

type tabWriter struct {
	inner *tabwriter.Writer
}

func (w *tabWriter) Write(records []string) error {
	_, err := w.inner.Write([]byte(strings.Join(records, "\t") + "\n"))
	return err
}

func (w *tabWriter) Flush() error {
	return w.inner.Flush()
}

type csvWriter struct {
	inner *csv.Writer
}

func (w *csvWriter) Write(records []string) error {
	return w.inner.Write(records)
}

func (w *csvWriter) Flush() error {
	w.inner.Flush()
	return nil
}
