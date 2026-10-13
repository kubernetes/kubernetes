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

package v1

import (
	"bufio"
	"fmt"
	"io"
	"sync"

	"github.com/go-logr/logr"

	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2/textlogger"
)

// textFactory produces klog text logger instances.
type textFactory struct{}

var _ LogFormatFactory = textFactory{}

func (f textFactory) Feature() featuregate.Feature {
	return LoggingStableOptions
}

func (f textFactory) Create(c LoggingConfiguration, o LoggingOptions) (logr.Logger, RuntimeControl) {
	output := o.ErrorStream
	var flush func()
	if c.Options.Text.SplitStream {
		r := &klogMsgRouter{
			info:  o.InfoStream,
			error: o.ErrorStream,
		}
		size := c.Options.Text.InfoBufferSize.Value()
		if size > 0 {
			// Prevent integer overflow.
			if size > 2*1024*1024*1024 {
				size = 2 * 1024 * 1024 * 1024
			}
			info := newBufferedWriter(r.info, int(size))
			flush = info.Flush
			r.info = info
		}
		output = r
	}

	options := []textlogger.ConfigOption{
		textlogger.Verbosity(int(c.Verbosity)),
		textlogger.Output(output),
	}
	loggerConfig := textlogger.NewConfig(options...)

	// This should never fail, we produce a valid string here.
	_ = loggerConfig.VModule().Set(VModuleConfigurationPflag(&c.VModule).String())

	return textlogger.NewLogger(loggerConfig),
		RuntimeControl{
			SetVerbosityLevel: func(v uint32) error {
				return loggerConfig.Verbosity().Set(fmt.Sprintf("%d", v))
			},
			Flush: flush,
		}
}

type klogMsgRouter struct {
	info, error io.Writer
}

var _ io.Writer = &klogMsgRouter{}

// Write redirects the message into either the info or error
// stream, depending on its type as indicated in text format
// by the first byte.
func (r *klogMsgRouter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	if p[0] == 'I' {
		return r.info.Write(p)
	}
	return r.error.Write(p)
}

// bufferedWriter is an io.Writer that buffers writes in-memory before
// flushing them to a wrapped io.Writer after reaching some limit
// or getting flushed.
type bufferedWriter struct {
	mu     sync.Mutex
	writer *bufio.Writer
	out    io.Writer
}

func newBufferedWriter(out io.Writer, size int) *bufferedWriter {
	return &bufferedWriter{
		writer: bufio.NewWriterSize(out, size),
		out:    out,
	}
}

func (b *bufferedWriter) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// To avoid partial writes into the underlying writer, we ensure that
	// the entire new data fits into the buffer or flush first.
	if len(p) > b.writer.Available() && b.writer.Buffered() > 0 {
		if err := b.writer.Flush(); err != nil {
			return 0, err
		}
	}

	// If it still doesn't fit, then we bypass the now empty buffer
	// and write directly.
	if len(p) > b.writer.Available() {
		return b.out.Write(p)
	}

	// This goes into the buffer.
	return b.writer.Write(p)
}

func (b *bufferedWriter) Flush() {
	b.mu.Lock()
	defer b.mu.Unlock()

	_ = b.writer.Flush()
}
