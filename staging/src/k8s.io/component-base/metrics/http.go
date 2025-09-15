/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"io"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	processStartedAt time.Time
)

func init() {
	processStartedAt = time.Now()
}

// These constants cause handlers serving metrics to behave as described if
// errors are encountered.
const (
	// HTTPErrorOnError serve an HTTP status code 500 upon the first error
	// encountered. Report the error message in the body.
	HTTPErrorOnError promhttp.HandlerErrorHandling = iota

	// ContinueOnError ignore errors and try to serve as many metrics as possible.
	// However, if no metrics can be served, serve an HTTP status code 500 and the
	// last error message in the body. Only use this in deliberate "best
	// effort" metrics collection scenarios. In this case, it is highly
	// recommended to provide other means of detecting errors: By setting an
	// ErrorLog in HandlerOpts, the errors are logged. By providing a
	// Registry in HandlerOpts, the exposed metrics include an error counter
	// "promhttp_metric_handler_errors_total", which can be used for
	// alerts.
	ContinueOnError

	// PanicOnError panics upon the first error encountered (useful for "crash only" apps).
	PanicOnError
)

// HandlerOpts specifies options how to serve metrics via an http.Handler. The
// zero value of HandlerOpts is a reasonable default.
type HandlerOpts promhttp.HandlerOpts

func (ho *HandlerOpts) toPromhttpHandlerOpts() promhttp.HandlerOpts {
	ho.ProcessStartTime = processStartedAt
	return promhttp.HandlerOpts(*ho)
}

// HandlerFor returns an uninstrumented http.Handler for the provided
// Gatherer. The behavior of the Handler is defined by the provided
// HandlerOpts. Thus, HandlerFor is useful to create http.Handlers for custom
// Gatherers, with non-default HandlerOpts, and/or with custom (or no)
// instrumentation. Use the InstrumentMetricHandler function to apply the same
// kind of instrumentation as it is used by the Handler function.
func HandlerFor(reg Gatherer, opts HandlerOpts) http.Handler {
	return promhttp.HandlerFor(reg, opts.toPromhttpHandlerOpts())
}

// HandlerWithReset return an http.Handler with Reset
func HandlerWithReset(reg KubeRegistry, opts HandlerOpts) http.Handler {
	defaultHandler := promhttp.HandlerFor(reg, opts.toPromhttpHandlerOpts())
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			reg.Reset()
			io.WriteString(w, "metrics reset\n")
			return
		}
		defaultHandler.ServeHTTP(w, r)
	})
}
