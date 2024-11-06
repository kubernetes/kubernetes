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

package ktesting

import (
	"context"
	"errors"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"testing"
)

var (
	// defaultProgressReporter is inactive until init is called.
	defaultProgressReporter = &progressReporter{}
)

const ginkgoSpecContextKey = "GINKGO_SPEC_CONTEXT"

type ginkgoReporter interface {
	AttachProgressReporter(reporter func() string) func()
}

type progressReporter struct {
	// initMutex protects initialization and finalization of the reporter.
	initMutex sync.Mutex

	usageCount              int64
	wg                      sync.WaitGroup
	signalCtx, interruptCtx context.Context
	signalCancel            func()
	progressChannel         chan os.Signal

	// reportMutex protects report creation and settings.
	reportMutex     sync.Mutex
	reporterCounter int64
	reporters       map[int64]func() string
	out             io.Writer
	closeOut        func() error
}

var _ ginkgoReporter = &progressReporter{}

// init is invoked by Init. It returns the context to be used for the
// new TContext.
//
// By default, that is just context.Background. In a Go unit test, it
// is a context connected to os.Interrupt.
//
// Once activated like that in a Go unit test, the progressReporter implements
// support for triggering a progress report in a running test when sending it a
// USR1 signal, similar to the corresponding Ginkgo feature.
//
// This support is active until the last test terminates.
func (p *progressReporter) init(tb TB) context.Context {
	if _, ok := tb.(testing.TB); !ok {
		// Not in a Go unit test.
		return context.Background()
	}

	p.initMutex.Lock()
	defer p.initMutex.Unlock()

	p.usageCount++
	tb.Cleanup(p.finalize)
	if p.usageCount > 1 {
		// Was already initialized.
		return p.interruptCtx
	}

	// Might have been set for testing purposes.
	if p.out == nil {
		// os.Stderr gets redirected by "go test". "go test -v" has to be
		// used to see that output while a test runs.
		//
		// Opening /dev/tty during init avoids the redirection.
		// May fail, depending on the OS, in which case
		// os.Stderr is used.
		if console, err := os.OpenFile("/dev/tty", os.O_RDWR|os.O_APPEND, 0); err == nil {
			p.out = console
			p.closeOut = console.Close

		} else {
			p.out = os.Stdout
			p.closeOut = nil
		}
	}

	p.signalCtx, p.signalCancel = signal.NotifyContext(context.Background(), os.Interrupt)
	cancelCtx, cancel := context.WithCancelCause(context.Background())
	p.wg.Go(func() {
		<-p.signalCtx.Done()
		cancel(errors.New("received interrupt signal"))
	})

	// This reimplements the contract between Ginkgo and Gomega for progress reporting.
	// When using Ginkgo contexts, Ginkgo will implement it. This here is for "go test".
	//
	// nolint:staticcheck // It complains about using a plain string. This can only be fixed
	// by Ginkgo and Gomega formalizing this interface and define a type (somewhere...
	// probably cannot be in either Ginkgo or Gomega).
	p.interruptCtx = context.WithValue(cancelCtx, ginkgoSpecContextKey, defaultProgressReporter)

	p.progressChannel = make(chan os.Signal, 1)
	// progressSignals will be empty on Windows.
	if len(progressSignals) > 0 {
		signal.Notify(p.progressChannel, progressSignals...)
	}

	p.wg.Go(p.run)

	return p.interruptCtx
}

func (p *progressReporter) finalize() {
	p.initMutex.Lock()
	defer p.initMutex.Unlock()

	p.usageCount--
	if p.usageCount > 0 {
		// Still in use.
		return
	}

	p.signalCancel()
	p.wg.Wait()

	// Now that all goroutines are stopped, we can clean up some more.
	if p.closeOut != nil {
		_ = p.closeOut()
		p.out = nil
	}
}

// AttachProgressReporter implements Gomega's contextWithAttachProgressReporter.
func (p *progressReporter) AttachProgressReporter(reporter func() string) func() {
	p.reportMutex.Lock()
	defer p.reportMutex.Unlock()

	// TODO (?): identify the caller and record that for dumpProgress.
	p.reporterCounter++
	id := p.reporterCounter
	if p.reporters == nil {
		p.reporters = make(map[int64]func() string)
	}
	p.reporters[id] = reporter
	return func() {
		p.detachProgressReporter(id)
	}
}

func (p *progressReporter) detachProgressReporter(id int64) {
	p.reportMutex.Lock()
	defer p.reportMutex.Unlock()

	delete(p.reporters, id)
}

func (p *progressReporter) run() {
	for {
		select {
		case <-p.interruptCtx.Done():
			// Maybe do one last progress report?
			//
			// This is primarily for unit testing of ktesting itself,
			// in a normal test we don't care anymore.
			select {
			case <-p.progressChannel:
				p.dumpProgress()
			default:
			}
			return
		case <-p.progressChannel:
			p.dumpProgress()
		}
	}
}

// dumpProgress is less useful than the Ginkgo progress report. We can't fix
// that we don't know which tests are currently running and instead have to
// rely on "go test -v" for that.
//
// But perhaps dumping goroutines and their callstacks is useful anyway?  TODO:
// look at how Ginkgo does it and replicate some of it.
func (p *progressReporter) dumpProgress() {
	p.reportMutex.Lock()
	defer p.reportMutex.Unlock()

	var buffer strings.Builder
	buffer.WriteString("You requested a progress report.\n")
	if len(p.reporters) == 0 {
		buffer.WriteString("Currently there is no information about test progress available.\n")
	}
	for _, reporter := range p.reporters {
		report := reporter()
		buffer.WriteRune('\n')
		buffer.WriteString(report)
		if !strings.HasSuffix(report, "\n") {
			buffer.WriteRune('\n')
		}
	}

	_, _ = p.out.Write([]byte(buffer.String()))
}
