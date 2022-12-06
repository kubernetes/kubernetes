// Copyright 2022 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promhttp

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
)

// Option are used to configure both handler (middleware) or round tripper.
type Option interface {
	apply(*options)
}

// options store options for both a handler or round tripper.
type options struct {
	extraMethods  []string
	getExemplarFn func(requestCtx context.Context) prometheus.Labels
}

func defaultOptions() *options {
	return &options{getExemplarFn: func(ctx context.Context) prometheus.Labels { return nil }}
}

type optionApplyFunc func(*options)

func (o optionApplyFunc) apply(opt *options) { o(opt) }

// WithExtraMethods adds additional HTTP methods to the list of allowed methods.
// See https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods for the default list.
//
// See the example for ExampleInstrumentHandlerWithExtraMethods for example usage.
func WithExtraMethods(methods ...string) Option {
	return optionApplyFunc(func(o *options) {
		o.extraMethods = methods
	})
}

// WithExemplarFromContext adds allows to put a hook to all counter and histogram metrics.
// If the hook function returns non-nil labels, exemplars will be added for that request, otherwise metric
// will get instrumented without exemplar.
func WithExemplarFromContext(getExemplarFn func(requestCtx context.Context) prometheus.Labels) Option {
	return optionApplyFunc(func(o *options) {
		o.getExemplarFn = getExemplarFn
	})
}
