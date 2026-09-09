/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"io"
)

const (
	yellowColor = "\u001b[33;1m"
	resetColor  = "\u001b[0m"
)

type WarningPrinter struct {
	// out is the writer to output warnings to
	out io.Writer
	// opts contains options controlling warning output
	opts WarningPrinterOptions
}

// WarningPrinterOptions controls the behavior of a WarningPrinter constructed using NewWarningPrinter()
type WarningPrinterOptions struct {
	// Color indicates that warning output can include ANSI color codes
	Color bool
}

// NewWarningPrinter returns an implementation of warningPrinter that outputs warnings to the specified writer.
func NewWarningPrinter(out io.Writer, opts WarningPrinterOptions) *WarningPrinter {
	h := &WarningPrinter{out: out, opts: opts}
	return h
}

// Print prints warnings to the configured writer.
func (w *WarningPrinter) Print(message string) {
	if w.opts.Color {
		fmt.Fprintf(w.out, "%sWarning:%s %s\n", yellowColor, resetColor, message)
	} else {
		fmt.Fprintf(w.out, "Warning: %s\n", message)
	}
}
