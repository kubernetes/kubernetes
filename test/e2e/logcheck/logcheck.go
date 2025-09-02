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
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
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

	// defaultNodesRE is the default regular expression for which nodes
	// should have their kubelet log output checked via the log query
	// feature (https://kubernetes.io/docs/concepts/cluster-administration/system-logs/#log-query).
	defaultNodesRE = `.*`

	dataRaceStart     = "WARNING: DATA RACE\n"
	dataRaceEnd       = "==================\n"
	maxBacktraceLines = 20
)

var (
	enabledLogChecks = logCheck{
		namespaces: regexpValue{
			re: regexp.MustCompile(defaultNamespacesRE),
		},
		nodes: regexpValue{
			re: regexp.MustCompile(defaultNodesRE),
		},
	}

	lc       *logChecker
	lcLogger klog.Logger
)

// logCheck determines what gets checked in log output.
type logCheck struct {
	// namespaces contains a regular expression which determines which namespaces need to be checked.
	// All containers in those namespaces are checked.
	// If the regular expression is nil, the default is used.
	namespaces regexpValue

	// namespaces contains a regular expression which determines which nodes need to be checked
	// by retrieving the kubelet log via the log query feature (https://kubernetes.io/docs/concepts/cluster-administration/system-logs/#log-query).
	nodes regexpValue

	// dataRaces enables checking for "DATA RACE" reports.
	dataRaces bool
	// More checks may get added in the future.
}

// any returns true if any log output check is enabled.
func (c logCheck) any() bool {
	// Needs to be extended when adding new checks.
	return c.dataRaces
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

// RegisterFlags adds command line flags for configuring the package to the given flag set.
// They have "logcheck" as prefix.
func RegisterFlags(fs *flag.FlagSet) {
	fs.BoolVar(&enabledLogChecks.dataRaces, "logcheck-data-races", false, "enables checking logs for DATA RACE warnings")
	fs.Var(&enabledLogChecks.namespaces, "logcheck-namespaces-regexp", "all namespaces matching this regular expressions get checked")
	fs.Var(&enabledLogChecks.nodes, "logcheck-nodes-regexp", "all kubelets on nodes matching this regular expressions get checked")
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
	client, err := kubernetes.NewForConfig(config)
	framework.ExpectNoError(err, "creating client")

	ctx, l, err := newLogChecker(context.Background(), client, enabledLogChecks, framework.TestContext.ReportDir)
	framework.ExpectNoError(err, "set up log checker")

	lc = l
	lcLogger = klog.FromContext(ctx)
	lc.start(ctx, podlogs.CopyAllLogs, kubeletLogQuery)
}

// finalize gets called once after tests have run, in the process where initialize was also called.
func finalize() {
	if lc == nil {
		return
	}

	failure, stdout := lc.stop(lcLogger)
	_, _ = ginkgo.GinkgoWriter.Write([]byte(stdout))
	if failure != "" {
		// Reports as post-suite failure.
		ginkgo.Fail(failure)
	}
}

func newLogChecker(ctx context.Context, client kubernetes.Interface, check logCheck, logDir string) (context.Context, *logChecker, error) {
	var logFile string

	if logDir != "" {
		// Redirect log output. Also, check for errors when we stop, those mustn't go unnoticed in a CI job.
		logFile = path.Join(logDir, "system-logs.log")
		output, err := os.Create(logFile)
		if err != nil {
			return ctx, nil, fmt.Errorf("create log file: %w", err)
		}
		// Allow increasing verbosity via the command line, but always use at
		// least 4 when writing into a file - we can afford that.
		vflag := flag.CommandLine.Lookup("v")
		v, _ := strconv.Atoi(vflag.Value.String())
		if v < 5 {
			v = 5
		}

		logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(output), textlogger.Verbosity((int(v)))))
		ctx = klog.NewContext(ctx, logger)
	}
	if check.namespaces.re == nil {
		check.namespaces.re = regexp.MustCompile(defaultNamespacesRE)
	}
	if check.nodes.re == nil {
		check.nodes.re = regexp.MustCompile(defaultNodesRE)
	}

	ctx, cancel := context.WithCancelCause(ctx)
	return ctx, &logChecker{
		wg:        newWaitGroup(),
		client:    client,
		cancel:    cancel,
		check:     check,
		logDir:    logDir,
		logFile:   logFile,
		dataRaces: make(map[string][][]string),
	}, nil
}

type logChecker struct {
	client  kubernetes.Interface
	cancel  func(err error)
	check   logCheck
	logDir  string
	logFile string

	// wg tracks background activity.
	wg *waitGroup

	// mutex protects all following fields.
	mutex sync.Mutex

	// All data races detected so far, indexed by "<namespace>/<pod name>/<container name>"
	// or "kubelet/<node name>".
	//
	// The last entry is initially empty while waiting for the next data race.
	//
	// Only entities for which at least some output was received get added here,
	// therefore also the "<entity>: okay" part of the report only appears for those.
	dataRaces map[string][][]string
}

// stop cancels pod monitoring, waits until that is shut down, and then produces text for a failure message (ideally empty) and stdout.
func (l *logChecker) stop(logger klog.Logger) (failure, stdout string) {
	logger.V(4).Info("Asking log monitors to stop")
	l.cancel(errors.New("asked to stop"))

	// Wait for completion.
	l.wg.wait()

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

		// Find all error log lines. A bit crude (doesn't handle multi-line outout), but good enough
		// because the full log is available.
		errorLogs := regexp.MustCompile(`(?m)^E.*\n`).FindAllString(string(logData), -1)
		if len(errorLogs) > 0 {
			failureBuffer.WriteString("Unexpected errors during log data collection (see stdout for full log):\n\n")
			for _, errorLog := range errorLogs {
				// Indented to make it a verbatim block in Markdown.
				failureBuffer.WriteString("    ")
				failureBuffer.WriteString(errorLog)
			}
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
			// Indented lines are stack backtraces. Those can be long,
			// so collect them as we iterate over lines and truncate in the middle.
			var backtrace []string
			dumpBacktrace := func() {
				if len(backtrace) == 0 {
					return
				}

				if len(backtrace) > maxBacktraceLines {
					head := backtrace[0 : maxBacktraceLines/2]
					tail := backtrace[len(backtrace)-maxBacktraceLines/2:]

					backtrace = slices.Clone(head)
					backtrace = append(backtrace, "  ...\n")
					backtrace = append(backtrace, tail...)
				}

				for _, line := range backtrace {
					buffer.WriteString(indent)
					buffer.WriteString(line)
				}

				backtrace = nil
			}
			for _, line := range race {
				if !strings.HasPrefix(line, " ") {
					// Non-backtrace line => flush and write the line.
					dumpBacktrace()
					buffer.WriteString(indent)
					buffer.WriteString(line)
					continue
				}
				backtrace = append(backtrace, line)
			}
			dumpBacktrace()
		}
	}

	return failureBuffer.String(), stdoutBuffer.String()
}

func (l *logChecker) start(ctx context.Context, startCopyAllLogs func(ctx context.Context, cs kubernetes.Interface, ns string, to podlogs.LogOutput) error, startNodeLog func(ctx context.Context, cs kubernetes.Interface, wg *waitGroup, nodeName string) io.Reader) {
	if !l.check.any() {
		return
	}

	logger := klog.FromContext(ctx)

	// Figure out which namespace(s) to watch.
	namespaces, err := l.client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		logger.Error(err, "Failed to list namespaces")
		return
	}

	// Same for nodes.
	nodes, err := l.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		logger.Error(err, "Failed to list nodes")
		return
	}

	for _, namespace := range namespaces.Items {
		if !l.check.namespaces.re.MatchString(namespace.Name) {
			continue
		}
		to := podlogs.LogOutput{
			StatusWriter:  &statusWriter{logger: logger, namespace: &namespace},
			LogPathPrefix: path.Join(l.logDir, namespace.Name) + "/",
			LogOpen: func(podName, containerName string) io.Writer {
				return l.logOpen(logger, namespace.Name, podName, containerName)
			},
		}

		logger.Info("Watching", "namespace", klog.KObj(&namespace))
		if err := startCopyAllLogs(ctx, l.client, namespace.Name, to); err != nil {
			logger.Error(err, "Log output collection failed", "namespace", klog.KObj(&namespace))
		}
	}

	for _, node := range nodes.Items {
		if !l.check.nodes.re.MatchString(node.Name) {
			continue
		}

		logger.Info("Watching", "node", klog.KObj(&node))
		kubeletLog := startNodeLog(ctx, l.client, l.wg, node.Name)
		l.wg.goIfNotShuttingDown(nil, func() {
			scanner := bufio.NewScanner(kubeletLog)
			writer := logOutputChecker{
				logger: logger,
				k:      "kubelet/" + node.Name,
				l:      l,
			}
			for scanner.Scan() {
				// We need to strip whatever headers might have been added by the log storage
				// that was queried the log query feature. We don't exactly know what that might be,
				// so let's use a regexp that matches all known line headers.
				//
				// Unknown lines are passed through, which is okayish (they get treated like
				// unknown, raw output from the kubelet).
				line := scanner.Text()
				line = logQueryLineHeaderRE.ReplaceAllString(line, "")
				_, _ = writer.Write([]byte(line + "\n"))
			}
			if err := scanner.Err(); err != nil {
				logger.Error(err, "Reading kubelet log failed", "node", klog.KObj(&node))
			}
		})
	}
}

// journaldHeader matches journald lines containing output from the kubelet:
//
//	Jan 06 15:20:26.641748 kind-worker2 kubelet[311]: I0106 15:20:26.641743     311 labels.go:289] ...
//
// We also get messages from systemd itself. Those are not matched:
//
//	Jan 07 08:35:52.139136 kind-worker3 systemd[1]: Started kubelet.service - kubelet: The Kubernetes Node Agent.
const journaldHeader = `... .. ..:..:......... \S+ kubelet\[\d+\]: `

// logQueryLineHeaderRE combines all supported log query data formats.
// Currently only journald is supported.
var logQueryLineHeaderRE = regexp.MustCompile(`^(?:` + journaldHeader + `)`)

func (l *logChecker) logOpen(logger klog.Logger, names ...string) io.Writer {
	if !l.check.dataRaces {
		return nil
	}

	if !l.wg.add(1) {
		return io.Discard
	}

	k := strings.Join(names, "/")
	logger.Info("Starting to check for data races", "container", k)
	return &logOutputChecker{logger: logger, k: k, l: l}
}

type statusWriter struct {
	logger    klog.Logger
	namespace *v1.Namespace
}

// Write gets called with text that describes problems encountered while monitoring pods and their output.
func (s *statusWriter) Write(msg []byte) (int, error) {
	s.logger.Error(nil, "PodLogs status", "namespace", klog.KObj(s.namespace), "msg", msg)
	return len(msg), nil
}

// logOutputChecker receives log output for one component and detects embedded DATA RACE reports.
type logOutputChecker struct {
	logger     klog.Logger
	k          string
	l          *logChecker
	inDataRace bool
}

var (
	_ io.Writer = &logOutputChecker{}
	_ io.Closer = &logOutputChecker{}
)

// Write gets called for each line of output received from a container or kubelet instance.
// The line ends with a newline.
func (p *logOutputChecker) Write(l []byte) (int, error) {
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
		p.logger.Info("Completed data race", "container", p.k, "count", len(races), "dataRace", strings.Join(races[len(races)-1], ""))
	case !p.inDataRace && line == dataRaceStart:
		// Start collecting data race lines.
		p.inDataRace = true
		races = append(races, nil)
		p.logger.Info("Started new data race", "container", p.k, "count", len(races))
	default:
		// Some other log output.
	}
	p.l.dataRaces[p.k] = races

	return len(l), nil
}

// Close gets called once all output is processed.
func (p *logOutputChecker) Close() error {
	p.logger.Info("Done checking for data races", "container", p.k)
	p.l.wg.done()
	return nil
}

// kubeletLogQuery sets up repeatedly querying the log of kubelet on the given node
// via the log query feature.
//
// All background goroutines stop when the context gets canceled and are added
// to the given WaitGroup before kubeletLogQuery returns. The reader will
// stop providing data when that happens.
//
// The lines in the data written to the reader may contain journald headers or other,
// platform specific headers (this is not specified for the log query feature).
func kubeletLogQuery(ctx context.Context, cs kubernetes.Interface, wg *waitGroup, nodeName string) io.Reader {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithName(logger, "KubeletLogQuery")
	logger = klog.LoggerWithValues(logger, "node", klog.KRef("", nodeName))
	reader, writer := io.Pipe()

	wg.goIfNotShuttingDown(func() {
		_ = writer.Close()
	}, func() {
		logger.V(4).Info("Started")
		defer func() {
			logger.V(4).Info("Stopped", "reason", context.Cause(ctx))
		}()

		// How long to wait between queries is a compromise between "too long" (= too much data)
		// and "too short" (= too much overhead because of frequent queries). It's not clear
		// where the sweet spot is. With the current default, there were 12 calls for one worker node
		// during a ~1h pull-kubernetes-e2e-kind-alpha-beta-features run, with an average result
		// size of ~16MB.
		//
		// All log checking activities share the same client instance. It's created specifically
		// for that purpose, so client-side throttling does not affect other tests.
		ticker := time.NewTicker(300 * time.Second)
		defer ticker.Stop()

		var since time.Time
		for {
			select {
			case <-ctx.Done():
				logger.V(4).Info("Asked to stop, will query log one last time", "reason", context.Cause(ctx))
			case <-ticker.C:
				logger.V(6).Info("Starting periodic log query")
			}

			// Query once also when asked to stop via cancelation, to get the tail of the output.
			// We use a timeout to prevent blocking forever here.
			err := func() error {
				ctx := context.WithoutCancel(ctx)
				ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
				defer cancel()

				now := time.Now()
				untilTime := now.Format(time.RFC3339)
				sinceTime := ""
				if !since.IsZero() {
					sinceTime = since.Format(time.RFC3339)
				}
				// The next query will use the current end time as start time. Both start and end
				// are inclusive, so unfortunately there is a risk that we get the same log
				// output for that time stamp twice. There's no good way to avoid that.
				// Let's hope it's rare and doesn't matter when it occurs. Missing
				// output would be worse because then the end marker of a DATA RACE
				// report could be missed.
				since = now
				req := cs.CoreV1().RESTClient().Post().
					Resource("nodes").
					Name(nodeName).
					SubResource("proxy").
					Suffix("logs").
					Param("query", "kubelet").
					Param("untilTime", untilTime)
				if sinceTime != "" {
					req = req.Param("sinceTime", sinceTime)
				}
				data, err := req.DoRaw(ctx)
				if loggerV := logger.V(4); loggerV.Enabled() {
					head := string(data)
					if len(head) > 30 {
						head = head[:30]
						head += "..."
					}
					loggerV.Info("Queried log", "sinceTime", sinceTime, "endTime", now, "len", len(data), "data", head, "err", err)
				}

				// Let's process whatever data we have. The exact result of the query is a bit
				// underspecified. This is based on observed behavior in Kubernetes 1.35.

				// HTML seems to be what the kubelet responds when the feature is disabled?!
				// May or may not be intentional according to a Slack discussion,
				// to be clarified in https://github.com/kubernetes/kubernetes/issues/136275.
				if strings.HasPrefix(string(data), "<!doctype html>") {
					return fmt.Errorf("unexpected result of log query (feature disabled?): %q", string(data))
				}

				// Skip the special string that is used for "no data available".
				if string(data) != "-- No entries --\n" {
					_, _ = writer.Write(data)
				}
				return err
			}()

			if err != nil {
				logger.Error(err, "Log query failed")
				return
			}

			if ctx.Err() != nil {
				return
			}
		}
	})
	return reader
}
