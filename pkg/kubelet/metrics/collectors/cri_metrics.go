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

package collectors

import (
	"bytes"
	"fmt"
	"io"
	"maps"
	"net/http"
	"time"

	"k8s.io/klog/v2"
)

const (
	criStatsPath    = "/metrics/cadvisor"
	criStatsTimeout = 30 * time.Second
)

// CRIStatsHandler returns an http.Handler that serves the output of
// registryHandler (cadvisor machine metrics) followed by metrics fetched from
// the CRI stats endpoint at the given address.
func CRIStatsHandler(logger klog.Logger, registryHandler http.Handler, criStatsAddress string) http.Handler {
	if criStatsAddress == "" {
		logger.Info("No CRI stats address configured, serving only machine metrics on cadvisor endpoint")
		return registryHandler
	}
	criStatsURL := "http://" + criStatsAddress + criStatsPath
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Disable compression for the registry handler since we concatenate
		// its output with the CRI response.
		// TODO(dgrisonnet): revisit this once we have a better way to handle
		// compression.
		r.Header.Del("Accept-Encoding")

		// Capture Accept before spawning the goroutine to avoid a data race
		// with registryHandler.ServeHTTP, which may read or write r.Header.
		accept := r.Header.Get("Accept")

		type criResult struct {
			resp *http.Response
			err  error
		}
		criCh := make(chan criResult, 1)
		go func() {
			criReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet, criStatsURL, nil)
			if err != nil {
				criCh <- criResult{err: err}
				return
			}
			// Forward the original Accept header so the CRI endpoint returns
			// metrics in the same Prometheus exposition format the client requested.
			criReq.Header.Set("Accept", accept)
			criHTTPClient := &http.Client{Timeout: criStatsTimeout}

			resp, err := criHTTPClient.Do(criReq)
			criCh <- criResult{resp: resp, err: err}
		}()

		var buf bytes.Buffer
		rw := &bufferedResponseWriter{header: make(http.Header), body: &buf}
		registryHandler.ServeHTTP(rw, r)

		maps.Copy(w.Header(), rw.header)
		if _, err := buf.WriteTo(w); err != nil {
			logger.Error(err, "Failed to write registry metrics")
		}

		res := <-criCh
		if res.err != nil {
			logger.Error(res.err, "Failed to fetch CRI metrics")
			return
		}

		defer func() {
			if err := res.resp.Body.Close(); err != nil {
				logger.Error(err, "Failed to close CRI response body")
			}
		}()

		if res.resp.StatusCode != http.StatusOK {
			logger.Error(fmt.Errorf("unexpected status %s", res.resp.Status), "Failed to fetch CRI metrics")
			return
		}

		if _, err := io.Copy(w, res.resp.Body); err != nil {
			logger.Error(err, "Failed to write CRI metrics")
		}
	})
}

type bufferedResponseWriter struct {
	header http.Header
	body   *bytes.Buffer
}

func (b *bufferedResponseWriter) Header() http.Header         { return b.header }
func (b *bufferedResponseWriter) WriteHeader(int)             {}
func (b *bufferedResponseWriter) Write(p []byte) (int, error) { return b.body.Write(p) }
