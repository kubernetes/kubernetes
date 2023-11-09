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

package coloredwriter

import (
	"github.com/fatih/color"
	"io"
	"k8s.io/cli-runtime/pkg/printers"
	"os"
)

type ColoredWriter struct {
	Enabled      bool
	OutWriter    io.Writer
	ActiveWriter io.Writer
}

func NewColoredWriter(w io.Writer) *ColoredWriter {
	_, found := os.LookupEnv("KUBECTL_THEME")
	if !found || color.NoColor {
		return &ColoredWriter{
			Enabled:      false,
			OutWriter:    w,
			ActiveWriter: w,
		}
	}

	return &ColoredWriter{
		Enabled:      true,
		OutWriter:    w,
		ActiveWriter: w,
	}
}

func (c *ColoredWriter) Write(p []byte) (n int, err error) {
	return c.ActiveWriter.Write(p)
}

func (c *ColoredWriter) UpdatePrinterType(p printers.ResourcePrinter) {
	if !c.Enabled {
		return
	}

	pp := p
	for pp != nil {
		switch v := pp.(type) {
		case *printers.OmitManagedFieldsPrinter:
			pp = v.Delegate
			continue
		case *printers.JSONPrinter:
			c.ActiveWriter = ColoredJsonWriter{Delegate: c.OutWriter}
		case *printers.YAMLPrinter:
			c.ActiveWriter = ColoredYamlWriter{Delegate: c.OutWriter}
		case *printers.HumanReadablePrinter:
			c.ActiveWriter = ColoredTabWriter{Delegate: c.OutWriter}
		default:
			//fmt.Println("Unknown type... %T", pp)
			c.ActiveWriter = c.OutWriter
		}
		break
	}

	return
}
