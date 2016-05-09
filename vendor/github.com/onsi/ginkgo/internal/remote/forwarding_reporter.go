package remote

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

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
}

func NewForwardingReporter(serverHost string, poster Poster, outputInterceptor OutputInterceptor) *ForwardingReporter {
	return &ForwardingReporter{
		serverHost:        serverHost,
		poster:            poster,
		outputInterceptor: outputInterceptor,
	}
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
	reporter.post("/SpecSuiteWillBegin", data)
}

func (reporter *ForwardingReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	setupSummary.CapturedOutput = output
	reporter.post("/BeforeSuiteDidRun", setupSummary)
}

func (reporter *ForwardingReporter) SpecWillRun(specSummary *types.SpecSummary) {
	reporter.post("/SpecWillRun", specSummary)
}

func (reporter *ForwardingReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	specSummary.CapturedOutput = output
	reporter.post("/SpecDidComplete", specSummary)
}

func (reporter *ForwardingReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	output, _ := reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.outputInterceptor.StartInterceptingOutput()
	setupSummary.CapturedOutput = output
	reporter.post("/AfterSuiteDidRun", setupSummary)
}

func (reporter *ForwardingReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.outputInterceptor.StopInterceptingAndReturnOutput()
	reporter.post("/SpecSuiteDidEnd", summary)
}
