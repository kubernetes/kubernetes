/*
Copyright The Kubernetes Authors.

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

// Package logcheck extends a Ginkgo test suite with optional monitoring
// of system log files. Right now it can detect DATA RACE problems.
//
// This may get expanded in the future, for example to check also
// for unhandled and potentially unexpected errors (https://github.com/kubernetes/kubernetes/issues/122005)
// or log spam (https://github.com/kubernetes/kubernetes/issues/109297).
//
// Because it grabs log output as it is produced, it is possible to dump *all*
// data into $ARTIFACTS/system-logs, in contrast to other approaches which
// potentially only capture the tail of the output (log rotation).
//
// The additional check(s) must be opt-in because they are not portable.
// The package registers additional command line flags for that.
// Dedicated jobs are necessary which enable them.
//
// For DATA RACE, the binaries in the cluster must be compiled with race
// detection (https://github.com/kubernetes/kubernetes/pull/133834/files).
// It's also necessary to specify which containers to check. Other output
// (e.g. kubelet in a kind cluster) currently cannot be checked.
package logcheck

import (
	"context"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
)

const (
	// defaultNamespacesRE is the default regular expression for which namespaces need to be checked,
	// if checking is enabled: anything which contains the word "system".
	//
	// If we had an API for identifying "system" namespaces, we could use that, but we
	// don't have that yet (https://github.com/kubernetes/enhancements/issues/5708).
	defaultNamespacesRE = `system`

	dataRaceStart = "WARNING: DATA RACE\n"
	dataRaceEnd   = "==================\n"
)

var (
	enabledLogChecks = logCheck{
		namespaces: regexpValue{
			re: regexp.MustCompile(defaultNamespacesRE),
		},
	}
	mainProcess bool
	lc          *logChecker
)

// logCheck determines what gets checked in log output.
type logCheck struct {
	// namespaces contains a regular expression which determines which namespaces need to be checked.
	// All containers in those namespaces are checked.
	namespaces regexpValue

	// dataRaces enables checking for "DATA RACE" reports.
	dataRaces bool
	// More checks may get added in the future.
}

// any returns true if any log output check is enabled.
func (c logCheck) any() bool {
	var empty logCheck
	return c != empty
}

// regexpValue implements flag.Value for a regular expression.
type regexpValue struct {
	re *regexp.Regexp
}

var _ flag.Value = &regexpValue{}

func (r *regexpValue) String() string { return r.re.String() }
func (r *regexpValue) Set(expr string) error {
	re, err := regexp.Compile(expr)
	if err != nil {
		// This already starts with "error parsing regexp" and
		// the caller adds the expression string,
		// so no need to wrap the error here.
		return err
	}
	r.re = re
	return nil
}

func init() {
	flag.BoolVar(&enabledLogChecks.dataRaces, "logcheck-data-races", false, "enables checking logs for DATA RACE warnings")
	flag.Var(&enabledLogChecks.namespaces, "logcheck-namespaces-regexp", "all namespaces matching this regular expressions get checked")
}

var _ = ginkgo.ReportBeforeSuite(func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	if report.SuiteConfig.DryRun {
		return
	}

	// This is only reached in the main process.
	initialize()
})

// SIG Testing runs this as a service for the SIGs which own the code.
var _ = ginkgo.ReportAfterSuite("[sig-testing] Log Check", func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	if report.SuiteConfig.DryRun {
		return
	}

	// This is reached only in the main process after the test run has completed.
	finalize()
})

// initialize gets called once before tests start to run.
// It sets up monitoring of system component log output, if requested.
func initialize() {
	if !enabledLogChecks.any() {
		return
	}

	config, err := framework.LoadConfig()
	framework.ExpectNoError(err, "loading client config")
	client, err := clientset.NewForConfig(config)
	framework.ExpectNoError(err, "creating client")

	ctx, cancel := context.WithCancel(context.Background())
	l, err := newLogChecker(client, cancel, enabledLogChecks, framework.TestContext.ReportDir)
	framework.ExpectNoError(err, "set up log checker")
	lc = l
	lc.start(ctx, podlogs.CopyAllLogs)
}

// finalize gets called once after tests have run, in the process where initialize was also called.
func finalize() {
	if lc == nil {
		return
	}

	failure, stdout := lc.stop()
	_, _ = ginkgo.GinkgoWriter.Write([]byte(stdout))
	if failure != "" {
		// Reports as post-suite failure.
		ginkgo.Fail(failure)
	}
}

func newLogChecker(client kubernetes.Interface, cancel func(), check logCheck, logDir string) (*logChecker, error) {
	var logger klog.Logger // No log output unless we have a logDir.
	var logFile string
	if logDir != "" {
		logFile = path.Join(logDir, "system-logs.log")
		output, err := os.Create(logFile)
		if err != nil {
			return nil, fmt.Errorf("create log file: %w", err)
		}
		logger = textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(output)))
	}

	return &logChecker{
		logger:    logger,
		client:    client,
		cancel:    cancel,
		check:     check,
		logDir:    logDir,
		logFile:   logFile,
		dataRaces: make(map[string][][]string),
	}, nil
}

type logChecker struct {
	logger  klog.Logger
	client  kubernetes.Interface
	cancel  func()
	check   logCheck
	logDir  string
	logFile string

	// wg counts the number of active pod output streams. Add and Wait must be protected by wgMutex.
	// Wait then only waits reliably for goroutines which were added *before* it gets called.
	wg      sync.WaitGroup
	wgMutex sync.Mutex

	// mutex protects all following fields.
	mutex sync.Mutex

	// All data races detected so far, indexed by "<namespace>/<pod name>/<container name>".
	// The last entry is initially empty while waiting for the next data race.
	dataRaces map[string][][]string
}

// stop cancels pod monitoring, waits until that is shut down, and then produces text for a failure message (ideally empty) and stdout.
func (l *logChecker) stop() (failure, stdout string) {
	l.cancel()

	// While we wait for completion, log checking must be able to lock l.mutex.
	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()
	l.wg.Wait()

	// Now we can proceed and produce the report.
	l.mutex.Lock()
	defer l.mutex.Unlock()

	var failureBuffer strings.Builder
	var stdoutBuffer strings.Builder
	if l.logFile != "" {
		logData, err := os.ReadFile(l.logFile)
		if err != nil {
			stdoutBuffer.WriteString(fmt.Sprintf("Reading %s failed: %v", l.logFile, err))
		} else {
			stdoutBuffer.Write(logData)
		}
	}

	keys := slices.AppendSeq([]string(nil), maps.Keys(l.dataRaces))
	slices.Sort(keys)
	for _, k := range keys {
		races := l.dataRaces[k]
		buffer := &failureBuffer
		if len(races) == 0 {
			buffer = &stdoutBuffer
		}
		if buffer.Len() > 0 {
			buffer.WriteString("\n")
		}
		buffer.WriteString("#### " + k + "\n")
		if len(races) == 0 {
			buffer.WriteString("\nOkay.\n")
			continue
		}
		for _, race := range races {
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
			}
		}
	}

	return failureBuffer.String(), stdoutBuffer.String()
}

func (l *logChecker) start(ctx context.Context, startCopyAllLogs func(ctx context.Context, cs clientset.Interface, ns string, to podlogs.LogOutput) error) {
	if !l.check.any() {
		return
	}

	// Figure out which namespace(s) to watch.
	namespaces, err := l.client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		l.logger.Error(err, "Failed to list namespaces")
		return
	}

	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()

	for _, namespace := range namespaces.Items {
		if !l.check.namespaces.re.MatchString(namespace.Name) {
			continue
		}
		to := podlogs.LogOutput{
			StatusWriter:  &statusWriter{l: l, namespace: &namespace},
			LogPathPrefix: path.Join(l.logDir, namespace.Name) + "/",
			LogOpen: func(podName, containerName string) io.Writer {
				return l.logOpen(namespace.Name, podName, containerName)
			},
		}

		l.logger.Info("Watching", "namespace", klog.KObj(&namespace))
		if err := startCopyAllLogs(ctx, l.client, namespace.Name, to); err != nil {
			l.logger.Error(err, "Log output collection failed", "namespace", klog.KObj(&namespace))
		}
	}
}

func (l *logChecker) logOpen(names ...string) io.Writer {
	if !l.check.dataRaces {
		return nil
	}

	l.wgMutex.Lock()
	defer l.wgMutex.Unlock()
	l.wg.Add(1)

	k := strings.Join(names, "/")
	l.logger.Info("Starting to check for data races", "container", k)
	return &podOutputWriter{k: k, l: l}
}

type statusWriter struct {
	l         *logChecker
	namespace *v1.Namespace
}

// Write gets called with text that describes problems encountered while monitoring pods and their output.
func (s *statusWriter) Write(msg []byte) (int, error) {
	s.l.logger.Info("PodLogs status", "namespace", klog.KObj(s.namespace), "msg", msg)
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
// The line ends with a newline.
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
		p.l.logger.Info("Completed data race", "container", p.k, "count", len(races), "dataRace", strings.Join(races[len(races)-1], ""))
	case !p.inDataRace && line == dataRaceStart:
		// Start collecting data race lines.
		p.inDataRace = true
		races = append(races, nil)
		p.l.logger.Info("Started new data race", "container", p.k, "count", len(races))
	default:
		// Some other log output.
	}
	p.l.dataRaces[p.k] = races

	return len(l), nil
}

// Close gets called once all output is processed.
func (p *podOutputWriter) Close() error {
	p.l.logger.Info("Done checking for data races", "container", p.k)
	p.l.wg.Done()
	return nil
}
