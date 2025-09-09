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

package invariants

import (
	"context"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
)

// The code in this file grabs logs of system containers and checks them for certain problems:
// - DATA RACE (depends on enabling race detection in the binary, see https://github.com/kubernetes/kubernetes/pull/133834).
// - potential future extension: unhandled and potentially unexpected errors (https://github.com/kubernetes/kubernetes/issues/122005)
//
// Please speak to SIG-Testing leads before adding anything to this file.

const (
	logInvariantsSIG              = "testing"
	logInvariantsContextText      = "Invariant Logs"
	logInvariantsDataRaceLeafText = "should enable data race checking"

	dataRaceStart = "WARNING: DATA RACE"
	dataRaceEnd   = "=================="
)

var (
	enabledLogChecks logCheck
	lc               *logChecker
)

// logCheck determines what gets checked in log output.
type logCheck struct {
	// dataRaces enables checking for "DATA RACE" reports.
	dataRaces bool
	// More checks may get added in the future.
}

// any returns true if any log output check is enabled.
func (c logCheck) any() bool {
	var empty logCheck
	return c != empty
}

var _ = framework.SIGDescribe(logInvariantsSIG)(logInvariantsContextText, func() {
	// This test is a sentinel for selecting the reporting of data races
	// in system components.
	//
	// This allows us to run it by default in most jobs, but it can be opted-out,
	// does not run when selecting Conformance, and it can be tagged Flaky
	// if we encounter issues with it.
	ginkgo.It(logInvariantsDataRaceLeafText /* , feature.DataRace TODO: add this once the pull-kubernetes-e2e-kind-alpha-beta-features-race job also sets it */, func() {})
})

var _ = ginkgo.ReportAfterSuite(fmt.Sprintf("[sig-%s] %s", logInvariantsSIG, logInvariantsContextText), func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	if report.SuiteConfig.DryRun {
		// This is reached after Ginkgo has determined which tests it is going to run
		// and before it actually runs anything. We can determine here what we are
		// meant to check, but actually kicking of background goroutines has to
		// wait until after suite initialization is done and tests start to run.
		// We use ginkgo.ReportBeforeEach for that.
		enabledLogChecks = logCheck{
			dataRaces: invariantsSelected(report, logInvariantsSIG, logInvariantsContextText, logInvariantsDataRaceLeafText),
		}
	} else {
		// This is reached after the test run has completed.
		finalizeLogs()
	}
})

var _ = ginkgo.ReportBeforeEach(func(report ginkgo.SpecReport) {
	initializeLogs()
})

// initializeLogs gets called before tests start to run. It sets up monitoring of system component log output,
// if requested through the sentinel tests and not started yet.
func initializeLogs() {
	if lc != nil || !enabledLogChecks.any() {
		return
	}

	// Failed assertions are okay here.
	// They get included in the final Ginkgo report because that is generated later.
	config, err := framework.LoadConfig()
	framework.ExpectNoError(err, "loading client config")
	client, err := clientset.NewForConfig(config)
	framework.ExpectNoError(err, "creating client")

	ctx, cancel := context.WithCancel(context.Background())
	lc = newLogChecker(client, cancel, enabledLogChecks)
	go lc.run(ctx, podlogs.CopyAllLogs)
}

func finalizeLogs() {
	if lc == nil {
		return
	}
	failed, report := lc.stop()

	if failed {
		// Reports as post-suite failure.
		ginkgo.Fail(report)
	} else {
		// Merely log the result.
		fmt.Fprint(ginkgo.GinkgoWriter, report)
	}
}

func newLogChecker(client kubernetes.Interface, cancel func(), check logCheck) *logChecker {
	return &logChecker{
		client:    client,
		cancel:    cancel,
		check:     check,
		dataRaces: make(map[string][][]string),
	}
}

type logChecker struct {
	client kubernetes.Interface
	cancel func()
	check  logCheck

	// wg counts the number of active pod output streams. Add and Wait must be protected by wgMutex.
	wg      sync.WaitGroup
	wgMutex sync.Mutex

	// mutex protects all following fields.
	mutex sync.Mutex

	// All data races detected so far, indexed by "<namespace>/<pod name>/<container name>".
	// The last entry is initially empty while waiting for the next data race.
	dataRaces map[string][][]string

	// errors collects all runtime errors.
	errors strings.Builder
}

// stop cancels pod monitoring, waits until that is shut down, and then produces a failure message if
// any issues where encountered. It may use ginkgo.Fail.
func (l *logChecker) stop() (failed bool, report string) {
	l.cancel()

	// While we wait for completion, log checking must be able to lock l.mutex.
	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()
	l.wg.Wait()

	l.mutex.Lock()
	defer l.mutex.Unlock()

	var buffer strings.Builder

	if l.errors.Len() > 0 {
		failed = true
		buffer.WriteString("#### Errors\n\n")
		buffer.WriteString(l.errors.String())
	}

	keys := slices.AppendSeq([]string(nil), maps.Keys(l.dataRaces))
	slices.Sort(keys)
	for i, k := range keys {
		if i > 0 || l.errors.Len() > 0 {
			buffer.WriteString("\n")
		}
		buffer.WriteString("#### " + k + "\n")
		races := l.dataRaces[k]
		if len(races) == 0 {
			buffer.WriteString("\nOkay.\n")
			continue
		}
		for _, race := range races {
			failed = true
			indent := "    "
			buffer.WriteString("\n")
			if len(races) > 1 {
				// Format as bullet-point list.
				buffer.WriteString("- DATA RACE:\n  \n")
				// This also shifts the text block to the right.
				indent += "  "
			} else {
				// Single line of intro text, then the text block.
				buffer.WriteString("DATA RACE:\n\n")
			}
			for _, line := range race {
				buffer.WriteString(indent)
				buffer.WriteString(line)
				buffer.WriteString("\n")
			}
		}
	}

	return failed, buffer.String()
}

func (l *logChecker) recordError(err error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.errors.WriteString(fmt.Sprintf("ERROR: %v\n", err))
}

func (l *logChecker) run(ctx context.Context, copyAllLogs func(ctx context.Context, cs clientset.Interface, ns string, to podlogs.LogOutput) error) {
	// Figure out which namespace(s) to watch. If we had an API for
	// identifying "system" namespaces, we could use this here, but we
	// don't have that yet (https://github.com/kubernetes/enhancements/issues/5708).
	//
	// Instead we monitor every namespace which has "system" in the name.
	namespaces, err := l.client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		l.recordError(fmt.Errorf("list namespaces: %w", err))
		return
	}

	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()

	for _, namespace := range namespaces.Items {
		if !strings.Contains(namespace.Name, "system") {
			continue
		}
		to := podlogs.LogOutput{
			StatusWriter: &statusWriter{l: l},
			LogOpen: func(podName, containerName string) io.Writer {
				return l.logOpen(namespace.Name, podName, containerName)
			},
		}

		l.wg.Go(func() {
			if err := copyAllLogs(ctx, l.client, namespace.Name, to); err != nil {
				l.recordError(fmt.Errorf("log output collection failed for %s: %w", namespace.Name, err))
			}
		})
	}
}

func (l *logChecker) logOpen(names ...string) io.Writer {
	k := strings.Join(names, "/")

	if !l.check.dataRaces {
		return nil
	}

	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()
	l.wg.Add(1)

	l.mutex.Lock()
	defer l.mutex.Unlock()
	// No problems so far...
	l.dataRaces[k] = nil

	return &podOutputWriter{k: k, l: l}
}

type statusWriter struct {
	l *logChecker
}

// Write gets called with text that describes problems encountered while monitoring pods and their output.
func (s *statusWriter) Write(msg []byte) (int, error) {
	s.l.mutex.Lock()
	defer s.l.mutex.Unlock()
	s.l.errors.WriteString(string(msg))
	return len(msg), nil
}

type podOutputWriter struct {
	k          string
	l          *logChecker
	inDataRace bool
}

var (
	_ io.Writer = &podOutputWriter{}
	_ io.Closer = &podOutputWriter{}
)

// Write gets called for each line of output received from a container.
func (p *podOutputWriter) Write(l []byte) (int, error) {
	line := string(l)

	p.l.mutex.Lock()
	defer p.l.mutex.Unlock()

	races := p.l.dataRaces[p.k]
	switch {
	case p.inDataRace && line != dataRaceEnd:
		races[len(races)-1] = append(races[len(races)-1], line)
	case p.inDataRace && line == dataRaceEnd:
		// Stop collecting data race lines.
		p.inDataRace = false
	case !p.inDataRace && line == dataRaceStart:
		// Start collecting data race lines.
		p.inDataRace = true
		races = append(races, nil)
	default:
		// Some other log output.
	}
	p.l.dataRaces[p.k] = races

	return len(l), nil
}

// Close gets called once all output is processed.
func (p *podOutputWriter) Close() error {
	p.l.wg.Done()
	return nil
}
