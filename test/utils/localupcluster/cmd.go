/*
Copyright 2025 The Kubernetes Authors.

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

// Package localupcluster contains wrapper code around invoking hack/local-up-cluster.sh
// and managing the resulting cluster.
package localupcluster

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

type Cmd struct {
	// Name is a short, descriptive name.
	Name string

	// CommandLine is the complete command line, including the command itself.
	CommandLine []string

	// AdditionalEnv gets added to the current environment.
	AdditionalEnv map[string]string

	// Log output as it gets printed.
	LogOutput bool

	// ProcessOutput, it non-nil, gets called for each line printed by the
	// command on stderr or stdout. The line does not include the trailing
	// newline.
	//
	// Called with EOF true when output processing stops. This implies
	// that the command has stopped or output processing failed. A non-empty
	// line in this case is the output processing error.
	ProcessOutput func(output Output)

	// Gather output in a string buffer. That collected output is returned by Wait and Stop.
	// If disabled, those methods return the empty string.
	GatherOutput bool

	// LogFile specifies a file to write the output to.
	// Can be combined with other options for output handling.
	// If it's the only one, then the command writes directly
	// into the file.
	LogFile string

	// KeepRunning ensures that the command is kept running beyond the end of its context,
	// i.e. context cancellation is ignored. Such commands have to be stopped explicitly.
	KeepRunning bool

	cancel    func(string)
	cmd       *exec.Cmd
	wg        sync.WaitGroup
	running   atomic.Pointer[bool]
	result    error
	gathering bool

	mutex  sync.RWMutex
	output strings.Builder
}

type Output struct {
	Line string
	EOF  bool
}

func (c *Cmd) Start(tCtx ktesting.TContext) {
	tCtx.Helper()
	tCtx.Logf("running command %s: %s", c.Name, strings.Join(c.CommandLine, " "))
	if c.KeepRunning {
		tCtx = ktesting.WithoutCancel(tCtx)
	}
	tCtx = ktesting.WithCancel(tCtx)
	c.cancel = tCtx.Cancel
	c.cmd = exec.CommandContext(tCtx, c.CommandLine[0], c.CommandLine[1:]...)
	c.gathering = false

	c.cmd.Env = os.Environ()
	for k, v := range c.AdditionalEnv {
		c.cmd.Env = append(c.cmd.Env, k+"="+v)
	}

	var reader io.Reader
	var writer io.WriteCloser

	c.gathering = false
	switch {
	case c.LogOutput || c.ProcessOutput != nil || c.GatherOutput:
		// Process each line through an in-memory pipe.
		reader, writer = io.Pipe()
		c.gathering = true
	case c.LogFile != "":
		// Let command write directly.
		f, err := os.Create(c.LogFile)
		tCtx.ExpectNoError(err, "create log file")
		writer = f
	}
	c.cmd.Stdout = writer
	c.cmd.Stderr = writer

	tCtx.ExpectNoError(c.cmd.Start(), "start %s command", c.Name)
	c.running.Store(ptr.To(true))

	if reader != nil {
		scanner := bufio.NewScanner(reader)
		c.wg.Add(1)
		go func() {
			defer c.wg.Done()
			for scanner.Scan() {
				line := scanner.Text()
				line = strings.TrimSuffix(line, "\n")
				if c.LogOutput {
					tCtx.Logf("%s: %s", c.Name, line)
				}
				if c.ProcessOutput != nil {
					c.ProcessOutput(Output{Line: line})
				}
				if c.GatherOutput {
					c.mutex.Lock()
					c.output.WriteString(line)
					c.output.WriteByte('\n')
					c.mutex.Unlock()
				}
			}
			if c.ProcessOutput != nil {
				if err := scanner.Err(); err != nil {
					c.ProcessOutput(Output{Line: err.Error(), EOF: true})
				} else {
					c.ProcessOutput(Output{EOF: true})
				}
			}
		}()
	}

	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.result = c.cmd.Wait()
		now := time.Now()
		if reader != nil {
			// Has to be closed to stop output processing, otherwise the scanner
			// keeps reading because someone might still write something.
			_ = writer.Close()
		}
		if c.LogFile != "" {
			f, err := os.OpenFile(c.LogFile, os.O_WRONLY|os.O_APPEND, 0666)
			if err == nil {
				defer func() {
					_ = f.Close()
				}()
				_, _ = fmt.Fprintf(f, "%s: terminated, result: %v\n", now, c.result)
				if err := context.Cause(tCtx); err != nil {
					_, _ = fmt.Fprintf(f, "%s: killed because command context was canceled: %v\n", now, err)
				}
			}
		}
		c.running.Store(ptr.To(false))
	}()
}

func (c *Cmd) Wait(tCtx ktesting.TContext) string {
	return c.wait(tCtx, false)
}

func (c *Cmd) Stop(tCtx ktesting.TContext, reason string) string {
	tCtx.Helper()
	if c.cancel == nil {
		// Not started...
		return ""
	}
	c.cancel(reason)
	return c.wait(tCtx, true)
}

func (c *Cmd) wait(tCtx ktesting.TContext, killed bool) string {
	tCtx.Helper()
	c.wg.Wait()
	if !killed {
		tCtx.ExpectNoError(c.result, fmt.Sprintf("%s command failed, output:\n%s", c.Name, c.output.String()))
	}
	return c.output.String()
}

func (c *Cmd) Running() bool {
	return ptr.Deref(c.running.Load(), false)
}

func (c *Cmd) Output(tCtx ktesting.TContext) string {
	if c.gathering {
		c.mutex.Lock()
		defer c.mutex.Unlock()
		return c.output.String()
	}

	if c.LogFile != "" {
		f, err := os.Open(c.LogFile)
		tCtx.ExpectNoError(err, "open log file")
		content, err := io.ReadAll(f)
		tCtx.ExpectNoError(err, "read log file")
		return string(content)
	}

	return ""
}
