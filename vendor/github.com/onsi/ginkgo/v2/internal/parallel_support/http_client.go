package parallel_support

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/onsi/ginkgo/v2/types"
)

type httpClient struct {
	serverHost string
}

func newHttpClient(serverHost string) *httpClient {
	return &httpClient{
		serverHost: serverHost,
	}
}

func (client *httpClient) Connect() bool {
	resp, err := http.Get(client.serverHost + "/up")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (client *httpClient) Close() error {
	return nil
}

func (client *httpClient) post(path string, data interface{}) error {
	var body io.Reader
	if data != nil {
		encoded, err := json.Marshal(data)
		if err != nil {
			return err
		}
		body = bytes.NewBuffer(encoded)
	}
	resp, err := http.Post(client.serverHost+path, "application/json", body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("received unexpected status code %d", resp.StatusCode)
	}
	return nil
}

func (client *httpClient) poll(path string, data interface{}) error {
	for {
		resp, err := http.Get(client.serverHost + path)
		if err != nil {
			return err
		}
		if resp.StatusCode == http.StatusTooEarly {
			resp.Body.Close()
			time.Sleep(POLLING_INTERVAL)
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusGone {
			return ErrorGone
		}
		if resp.StatusCode == http.StatusFailedDependency {
			return ErrorFailed
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("received unexpected status code %d", resp.StatusCode)
		}
		if data != nil {
			return json.NewDecoder(resp.Body).Decode(data)
		}
		return nil
	}
}

func (client *httpClient) PostSuiteWillBegin(report types.Report) error {
	return client.post("/suite-will-begin", report)
}

func (client *httpClient) PostDidRun(report types.SpecReport) error {
	return client.post("/did-run", report)
}

func (client *httpClient) PostSuiteDidEnd(report types.Report) error {
	return client.post("/suite-did-end", report)
}

func (client *httpClient) PostEmitProgressReport(report types.ProgressReport) error {
	return client.post("/progress-report", report)
}

func (client *httpClient) PostReportBeforeSuiteCompleted(state types.SpecState) error {
	return client.post("/report-before-suite-completed", state)
}

func (client *httpClient) BlockUntilReportBeforeSuiteCompleted() (types.SpecState, error) {
	var state types.SpecState
	err := client.poll("/report-before-suite-state", &state)
	if err == ErrorGone {
		return types.SpecStateFailed, nil
	}
	return state, err
}

func (client *httpClient) PostSynchronizedBeforeSuiteCompleted(state types.SpecState, data []byte) error {
	beforeSuiteState := BeforeSuiteState{
		State: state,
		Data:  data,
	}
	return client.post("/before-suite-completed", beforeSuiteState)
}

func (client *httpClient) BlockUntilSynchronizedBeforeSuiteData() (types.SpecState, []byte, error) {
	var beforeSuiteState BeforeSuiteState
	err := client.poll("/before-suite-state", &beforeSuiteState)
	if err == ErrorGone {
		return types.SpecStateInvalid, nil, types.GinkgoErrors.SynchronizedBeforeSuiteDisappearedOnProc1()
	}
	return beforeSuiteState.State, beforeSuiteState.Data, err
}

func (client *httpClient) BlockUntilNonprimaryProcsHaveFinished() error {
	return client.poll("/have-nonprimary-procs-finished", nil)
}

func (client *httpClient) BlockUntilAggregatedNonprimaryProcsReport() (types.Report, error) {
	var report types.Report
	err := client.poll("/aggregated-nonprimary-procs-report", &report)
	if err == ErrorGone {
		return types.Report{}, types.GinkgoErrors.AggregatedReportUnavailableDueToNodeDisappearing()
	}
	return report, err
}

func (client *httpClient) FetchNextCounter() (int, error) {
	var counter ParallelIndexCounter
	err := client.poll("/counter", &counter)
	return counter.Index, err
}

func (client *httpClient) PostAbort() error {
	return client.post("/abort", nil)
}

func (client *httpClient) ShouldAbort() bool {
	err := client.poll("/abort", nil)
	if err == ErrorGone {
		return true
	}
	return false
}

func (client *httpClient) Write(p []byte) (int, error) {
	resp, err := http.Post(client.serverHost+"/emit-output", "text/plain;charset=UTF-8 ", bytes.NewReader(p))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("failed to emit output")
	}
	return len(p), err
}
