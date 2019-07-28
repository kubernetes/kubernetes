package remote

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/reporters/stenographer"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
)

//An interface to net/http's client to allow the injection of fakes under test
type Poster interface {
	Post(url string, bodyType string, body io.Reader) (resp *http.Response, err error)
}

/*
The ForwardingReporter is a Ginkgo reporter that forwards information to
a Ginkgo remote server.

When streaming parallel test output, this repoter is automatically installed by Ginkgo.

This is accomplished by passing in the GINKGO_REMOTE_REPORTING_SERVER environment variable to `go test`, the Ginkgo test runner
detects this environment variable (which should contain the host of the server) and automatically installs a ForwardingReporter
in place of Ginkgo's DefaultReporter.
*/

type ForwardingReporter struct {
	serverHost        string
	poster            Poster
	outputInterceptor OutputInterceptor
	debugMode         bool
	debugFile         *os.File
	nestedReporter    *reporters.DefaultReporter
}

func NewForwardingReporter(config config.DefaultReporterConfigType, serverHost string, poster Poster, outputInterceptor OutputInterceptor, ginkgoWriter *writer.Writer, debugFile string) *ForwardingReporter {
	reporter := &ForwardingReporter{
		serverHost:        serverHost,
		poster:            poster,
		outputInterceptor: outputInterceptor,
	}

	if debugFile != "" {
		var err error
		reporter.debugMode = true
		reporter.debugFile, err = os.Create(debugFile)
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}

		if !config.Verbose {
			//if verbose is true then the GinkgoWriter emits to stdout.  Don't _also_ redirect GinkgoWriter output as that will result in duplication.
			ginkgoWriter.AndRedirectTo(reporter.debugFile)
		}
		outputInterceptor.StreamTo(reporter.debugFile) //This is not working

		stenographer := stenographer.New(false, true, reporter.debugFile)
		config.Succinct = false
		config.Verbose = true
		config.FullTrace = true
		reporter.nestedReporter = reporters.NewDefaultReporter(config, stenographer)
	}

	return reporter
}

func (reporter *ForwardingReporter) post(path string, data interface{}) {
	encoded, _ := json.Marshal(data)
	buffer := bytes.NewBuffer(encoded)
	reporter.poster.Post(reporter.serverHost+path, "application/json", buffer)
}

func (reporter *ForwardingReporter) SpecSuiteWillBegin(conf config.GinkgoConfigType, summary *types.SuiteSummary) {
	data := struct {
		Config  config.GinkgoConfigType `json:"config"`
		Summary *types.SuiteSummary     `json:"suite-summary"`
	}{
		conf,
		summary,
	}

	reporter.outputInterceptor.StartInterceptingOutput()
	if reporter.debugMode {
		reporter.nestedReporter.SpecSuiteWillBegin(conf, summary)
		reporter.debugFile.Sync()
	}
	reporter.post("/SpecSuiteWillBegin", data)
}

func (reporter *ForwardingReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	setupSummary.CapturedOutput = output
	if reporter.debugMode {
		reporter.nestedReporter.BeforeSuiteDidRun(setupSummary)
		reporter.debugFile.Sync()
	}
	reporter.post("/BeforeSuiteDidRun", setupSummary)
}

func (reporter *ForwardingReporter) SpecWillRun(specSummary *types.SpecSummary) {
	if reporter.debugMode {
		reporter.nestedReporter.SpecWillRun(specSummary)
		reporter.debugFile.Sync()
	}
	reporter.post("/SpecWillRun", specSummary)
}

func (reporter *ForwardingReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	specSummary.CapturedOutput = output
	if reporter.debugMode {
		reporter.nestedReporter.SpecDidComplete(specSummary)
		reporter.debugFile.Sync()
	}
	reporter.post("/SpecDidComplete", specSummary)
}

func (reporter *ForwardingReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	setupSummary.CapturedOutput = output
	if reporter.debugMode {
		reporter.nestedReporter.AfterSuiteDidRun(setupSummary)
		reporter.debugFile.Sync()
	}
	reporter.post("/AfterSuiteDidRun", setupSummary)
}

func (reporter *ForwardingReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	if reporter.debugMode {
		reporter.nestedReporter.SpecSuiteDidEnd(summary)
		reporter.debugFile.Sync()
	}
	reporter.post("/SpecSuiteDidEnd", summary)
}
