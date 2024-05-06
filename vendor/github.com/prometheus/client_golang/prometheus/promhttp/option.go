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

// LabelValueFromCtx are used to compute the label value from request context.
// Context can be filled with values from request through middleware.
type LabelValueFromCtx func(ctx context.Context) string

// options store options for both a handler or round tripper.
type options struct {
	extraMethods       []string
	getExemplarFn      func(requestCtx context.Context) prometheus.Labels
	extraLabelsFromCtx map[string]LabelValueFromCtx
}

func defaultOptions() *options {
	return &options{
		getExemplarFn:      func(ctx context.Context) prometheus.Labels { return nil },
		extraLabelsFromCtx: map[string]LabelValueFromCtx{},
	}
}

func (o *options) emptyDynamicLabels() prometheus.Labels {
	labels := prometheus.Labels{}

	for label := range o.extraLabelsFromCtx {
		labels[label] = ""
	}

	return labels
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

// WithExemplarFromContext allows to inject function that will get exemplar from context that will be put to counter and histogram metrics.
// If the function returns nil labels or the metric does not support exemplars, no exemplar will be added (noop), but
// metric will continue to observe/increment.
func WithExemplarFromContext(getExemplarFn func(requestCtx context.Context) prometheus.Labels) Option {
	return optionApplyFunc(func(o *options) {
		o.getExemplarFn = getExemplarFn
	})
}

// WithLabelFromCtx registers a label for dynamic resolution with access to context.
// See the example for ExampleInstrumentHandlerWithLabelResolver for example usage
func WithLabelFromCtx(name string, valueFn LabelValueFromCtx) Option {
	return optionApplyFunc(func(o *options) {
		o.extraLabelsFromCtx[name] = valueFn
	})
}
