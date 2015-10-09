/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

// Dashboard metrics
type LatencyMetric struct {
	Perc50 time.Duration `json:"Perc50"`
	Perc90 time.Duration `json:"Perc90"`
	Perc99 time.Duration `json:"Perc99"`
}

type PodStartupLatency struct {
	Latency LatencyMetric `json:"latency"`
}

type APICall struct {
	Resource string        `json:"resource"`
	Verb     string        `json:"verb"`
	Latency  LatencyMetric `json:"latency"`
}

type APIResponsiveness struct {
	APICalls []APICall `json:"apicalls"`
}

func (a APIResponsiveness) Len() int      { return len(a.APICalls) }
func (a APIResponsiveness) Swap(i, j int) { a.APICalls[i], a.APICalls[j] = a.APICalls[j], a.APICalls[i] }
func (a APIResponsiveness) Less(i, j int) bool {
	return a.APICalls[i].Latency.Perc99 < a.APICalls[j].Latency.Perc99
}

// 0 <= quantile <=1 (e.g. 0.95 is 95%tile, 0.5 is median)
// Only 0.5, 0.9 and 0.99 quantiles are supported.
func (a *APIResponsiveness) addMetric(resource, verb string, quantile float64, latency time.Duration) {
	for i, apicall := range a.APICalls {
		if apicall.Resource == resource && apicall.Verb == verb {
			a.APICalls[i] = setQuantile(apicall, quantile, latency)
			return
		}
	}
	apicall := setQuantile(APICall{Resource: resource, Verb: verb}, quantile, latency)
	a.APICalls = append(a.APICalls, apicall)
}

// 0 <= quantile <=1 (e.g. 0.95 is 95%tile, 0.5 is median)
// Only 0.5, 0.9 and 0.99 quantiles are supported.
func setQuantile(apicall APICall, quantile float64, latency time.Duration) APICall {
	switch quantile {
	case 0.5:
		apicall.Latency.Perc50 = latency
	case 0.9:
		apicall.Latency.Perc90 = latency
	case 0.99:
		apicall.Latency.Perc99 = latency
	}
	return apicall
}

func readLatencyMetrics(c *client.Client) (APIResponsiveness, error) {
	var a APIResponsiveness

	body, err := getMetrics(c)
	if err != nil {
		return a, err
	}

	samples, err := extractMetricSamples(body)
	if err != nil {
		return a, err
	}

	ignoredResources := sets.NewString("events")
	ignoredVerbs := sets.NewString("WATCHLIST", "PROXY")

	for _, sample := range samples {
		// Example line:
		// apiserver_request_latencies_summary{resource="namespaces",verb="LIST",quantile="0.99"} 908
		if sample.Metric[model.MetricNameLabel] != "apiserver_request_latencies_summary" {
			continue
		}

		resource := string(sample.Metric["resource"])
		verb := string(sample.Metric["verb"])
		if ignoredResources.Has(resource) || ignoredVerbs.Has(verb) {
			continue
		}
		latency := sample.Value
		quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
		if err != nil {
			return a, err
		}
		a.addMetric(resource, verb, quantile, time.Duration(int64(latency))*time.Microsecond)
	}

	return a, err
}

// Prints summary metrics for request types with latency above threshold
// and returns number of such request types.
func HighLatencyRequests(c *client.Client, threshold time.Duration) (int, error) {
	metrics, err := readLatencyMetrics(c)
	if err != nil {
		return 0, err
	}
	sort.Sort(sort.Reverse(metrics))
	badMetrics := 0
	top := 5
	for _, metric := range metrics.APICalls {
		isBad := false
		if metric.Latency.Perc99 > threshold {
			badMetrics++
			isBad = true
		}
		if top > 0 || isBad {
			top--
			prefix := ""
			if isBad {
				prefix = "WARNING "
			}
			Logf("%vTop latency metric: %+v", prefix, metric)
		}
	}

	Logf("API calls latencies: %s", prettyPrintJSON(metrics))

	return badMetrics, nil
}

// Verifies whether 50, 90 and 99th percentiles of PodStartupLatency are smaller
// than the given threshold (returns error in the oposite case).
func VerifyPodStartupLatency(latency PodStartupLatency, podStartupThreshold time.Duration) error {
	Logf("Pod startup latency: %s", prettyPrintJSON(latency))

	if latency.Latency.Perc50 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 50th percentile: %v", latency.Latency.Perc50)
	}
	if latency.Latency.Perc90 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 90th percentile: %v", latency.Latency.Perc90)
	}
	if latency.Latency.Perc99 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 99th percentil: %v", latency.Latency.Perc99)
	}
	return nil
}

// Resets latency metrics in apiserver.
func resetMetrics(c *client.Client) error {
	Logf("Resetting latency metrics in apiserver...")
	body, err := c.Get().AbsPath("/resetMetrics").DoRaw()
	if err != nil {
		return err
	}
	if string(body) != "metrics reset\n" {
		return fmt.Errorf("Unexpected response: %q", string(body))
	}
	return nil
}

// Retrieves metrics information.
func getMetrics(c *client.Client) (string, error) {
	body, err := c.Get().AbsPath("/metrics").DoRaw()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func prettyPrintJSON(metrics interface{}) string {
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(metrics); err != nil {
		return ""
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		return ""
	}
	return string(formatted.Bytes())
}

// Retrieves debug information.
func getDebugInfo(c *client.Client) (map[string]string, error) {
	data := make(map[string]string)
	for _, key := range []string{"block", "goroutine", "heap", "threadcreate"} {
		resp, err := http.Get(c.Get().AbsPath(fmt.Sprintf("debug/pprof/%s", key)).URL().String() + "?debug=2")
		if err != nil {
			Logf("Warning: Error trying to fetch %s debug data: %v", key, err)
			continue
		}
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			Logf("Warning: Error trying to read %s debug data: %v", key, err)
		}
		data[key] = string(body)
	}
	return data, nil
}

func writePerfData(c *client.Client, dirName string, postfix string) error {
	fname := fmt.Sprintf("%s/metrics_%s.txt", dirName, postfix)

	handler, err := os.Create(fname)
	if err != nil {
		return fmt.Errorf("Error creating file '%s': %v", fname, err)
	}

	metrics, err := getMetrics(c)
	if err != nil {
		return fmt.Errorf("Error retrieving metrics: %v", err)
	}

	_, err = handler.WriteString(metrics)
	if err != nil {
		return fmt.Errorf("Error writing metrics: %v", err)
	}

	err = handler.Close()
	if err != nil {
		return fmt.Errorf("Error closing '%s': %v", fname, err)
	}

	debug, err := getDebugInfo(c)
	if err != nil {
		return fmt.Errorf("Error retrieving debug information: %v", err)
	}

	for key, value := range debug {
		fname := fmt.Sprintf("%s/%s_%s.txt", dirName, key, postfix)
		handler, err = os.Create(fname)
		if err != nil {
			return fmt.Errorf("Error creating file '%s': %v", fname, err)
		}
		_, err = handler.WriteString(value)
		if err != nil {
			return fmt.Errorf("Error writing %s: %v", key, err)
		}

		err = handler.Close()
		if err != nil {
			return fmt.Errorf("Error closing '%s': %v", fname, err)
		}
	}
	return nil
}

// extractMetricSamples parses the prometheus metric samples from the input string.
func extractMetricSamples(metricsBlob string) ([]*model.Sample, error) {
	dec, err := expfmt.NewDecoder(strings.NewReader(metricsBlob), expfmt.FmtText)
	if err != nil {
		return nil, err
	}
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	var samples []*model.Sample
	for {
		var v model.Vector
		if err = decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return samples, nil
			}
			return nil, err
		}
		samples = append(samples, v...)
	}
}
