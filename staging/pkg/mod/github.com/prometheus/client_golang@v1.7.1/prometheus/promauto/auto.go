// Copyright 2018 The Prometheus Authors
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

// Package promauto provides alternative constructors for the fundamental
// Prometheus metric types and their …Vec and …Func variants. The difference to
// their counterparts in the prometheus package is that the promauto
// constructors return Collectors that are already registered with a
// registry. There are two sets of constructors. The constructors in the first
// set are top-level functions, while the constructors in the other set are
// methods of the Factory type. The top-level function return Collectors
// registered with the global registry (prometheus.DefaultRegisterer), while the
// methods return Collectors registered with the registry the Factory was
// constructed with. All constructors panic if the registration fails.
//
// The following example is a complete program to create a histogram of normally
// distributed random numbers from the math/rand package:
//
//      package main
//
//      import (
//              "math/rand"
//              "net/http"
//
//              "github.com/prometheus/client_golang/prometheus"
//              "github.com/prometheus/client_golang/prometheus/promauto"
//              "github.com/prometheus/client_golang/prometheus/promhttp"
//      )
//
//      var histogram = promauto.NewHistogram(prometheus.HistogramOpts{
//              Name:    "random_numbers",
//              Help:    "A histogram of normally distributed random numbers.",
//              Buckets: prometheus.LinearBuckets(-3, .1, 61),
//      })
//
//      func Random() {
//              for {
//                      histogram.Observe(rand.NormFloat64())
//              }
//      }
//
//      func main() {
//              go Random()
//              http.Handle("/metrics", promhttp.Handler())
//              http.ListenAndServe(":1971", nil)
//      }
//
// Prometheus's version of a minimal hello-world program:
//
//      package main
//
//      import (
//      	"fmt"
//      	"net/http"
//
//      	"github.com/prometheus/client_golang/prometheus"
//      	"github.com/prometheus/client_golang/prometheus/promauto"
//      	"github.com/prometheus/client_golang/prometheus/promhttp"
//      )
//
//      func main() {
//      	http.Handle("/", promhttp.InstrumentHandlerCounter(
//      		promauto.NewCounterVec(
//      			prometheus.CounterOpts{
//      				Name: "hello_requests_total",
//      				Help: "Total number of hello-world requests by HTTP code.",
//      			},
//      			[]string{"code"},
//      		),
//      		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
//      			fmt.Fprint(w, "Hello, world!")
//      		}),
//      	))
//      	http.Handle("/metrics", promhttp.Handler())
//      	http.ListenAndServe(":1971", nil)
//      }
//
// A Factory is created with the With(prometheus.Registerer) function, which
// enables two usage pattern. With(prometheus.Registerer) can be called once per
// line:
//
//        var (
//        	reg           = prometheus.NewRegistry()
//        	randomNumbers = promauto.With(reg).NewHistogram(prometheus.HistogramOpts{
//        		Name:    "random_numbers",
//        		Help:    "A histogram of normally distributed random numbers.",
//        		Buckets: prometheus.LinearBuckets(-3, .1, 61),
//        	})
//        	requestCount = promauto.With(reg).NewCounterVec(
//        		prometheus.CounterOpts{
//        			Name: "http_requests_total",
//        			Help: "Total number of HTTP requests by status code and method.",
//        		},
//        		[]string{"code", "method"},
//        	)
//        )
//
// Or it can be used to create a Factory once to be used multiple times:
//
//        var (
//        	reg           = prometheus.NewRegistry()
//        	factory       = promauto.With(reg)
//        	randomNumbers = factory.NewHistogram(prometheus.HistogramOpts{
//        		Name:    "random_numbers",
//        		Help:    "A histogram of normally distributed random numbers.",
//        		Buckets: prometheus.LinearBuckets(-3, .1, 61),
//        	})
//        	requestCount = factory.NewCounterVec(
//        		prometheus.CounterOpts{
//        			Name: "http_requests_total",
//        			Help: "Total number of HTTP requests by status code and method.",
//        		},
//        		[]string{"code", "method"},
//        	)
//        )
//
// This appears very handy. So why are these constructors locked away in a
// separate package?
//
// The main problem is that registration may fail, e.g. if a metric inconsistent
// with or equal to the newly to be registered one is already registered.
// Therefore, the Register method in the prometheus.Registerer interface returns
// an error, and the same is the case for the top-level prometheus.Register
// function that registers with the global registry. The prometheus package also
// provides MustRegister versions for both. They panic if the registration
// fails, and they clearly call this out by using the Must…  idiom. Panicking is
// problematic in this case because it doesn't just happen on input provided by
// the caller that is invalid on its own. Things are a bit more subtle here:
// Metric creation and registration tend to be spread widely over the
// codebase. It can easily happen that an incompatible metric is added to an
// unrelated part of the code, and suddenly code that used to work perfectly
// fine starts to panic (provided that the registration of the newly added
// metric happens before the registration of the previously existing
// metric). This may come as an even bigger surprise with the global registry,
// where simply importing another package can trigger a panic (if the newly
// imported package registers metrics in its init function). At least, in the
// prometheus package, creation of metrics and other collectors is separate from
// registration. You first create the metric, and then you decide explicitly if
// you want to register it with a local or the global registry, and if you want
// to handle the error or risk a panic. With the constructors in the promauto
// package, registration is automatic, and if it fails, it will always
// panic. Furthermore, the constructors will often be called in the var section
// of a file, which means that panicking will happen as a side effect of merely
// importing a package.
//
// A separate package allows conservative users to entirely ignore it. And
// whoever wants to use it, will do so explicitly, with an opportunity to read
// this warning.
//
// Enjoy promauto responsibly!
package promauto

import "github.com/prometheus/client_golang/prometheus"

// NewCounter works like the function of the same name in the prometheus package
// but it automatically registers the Counter with the
// prometheus.DefaultRegisterer. If the registration fails, NewCounter panics.
func NewCounter(opts prometheus.CounterOpts) prometheus.Counter {
	return With(prometheus.DefaultRegisterer).NewCounter(opts)
}

// NewCounterVec works like the function of the same name in the prometheus
// package but it automatically registers the CounterVec with the
// prometheus.DefaultRegisterer. If the registration fails, NewCounterVec
// panics.
func NewCounterVec(opts prometheus.CounterOpts, labelNames []string) *prometheus.CounterVec {
	return With(prometheus.DefaultRegisterer).NewCounterVec(opts, labelNames)
}

// NewCounterFunc works like the function of the same name in the prometheus
// package but it automatically registers the CounterFunc with the
// prometheus.DefaultRegisterer. If the registration fails, NewCounterFunc
// panics.
func NewCounterFunc(opts prometheus.CounterOpts, function func() float64) prometheus.CounterFunc {
	return With(prometheus.DefaultRegisterer).NewCounterFunc(opts, function)
}

// NewGauge works like the function of the same name in the prometheus package
// but it automatically registers the Gauge with the
// prometheus.DefaultRegisterer. If the registration fails, NewGauge panics.
func NewGauge(opts prometheus.GaugeOpts) prometheus.Gauge {
	return With(prometheus.DefaultRegisterer).NewGauge(opts)
}

// NewGaugeVec works like the function of the same name in the prometheus
// package but it automatically registers the GaugeVec with the
// prometheus.DefaultRegisterer. If the registration fails, NewGaugeVec panics.
func NewGaugeVec(opts prometheus.GaugeOpts, labelNames []string) *prometheus.GaugeVec {
	return With(prometheus.DefaultRegisterer).NewGaugeVec(opts, labelNames)
}

// NewGaugeFunc works like the function of the same name in the prometheus
// package but it automatically registers the GaugeFunc with the
// prometheus.DefaultRegisterer. If the registration fails, NewGaugeFunc panics.
func NewGaugeFunc(opts prometheus.GaugeOpts, function func() float64) prometheus.GaugeFunc {
	return With(prometheus.DefaultRegisterer).NewGaugeFunc(opts, function)
}

// NewSummary works like the function of the same name in the prometheus package
// but it automatically registers the Summary with the
// prometheus.DefaultRegisterer. If the registration fails, NewSummary panics.
func NewSummary(opts prometheus.SummaryOpts) prometheus.Summary {
	return With(prometheus.DefaultRegisterer).NewSummary(opts)
}

// NewSummaryVec works like the function of the same name in the prometheus
// package but it automatically registers the SummaryVec with the
// prometheus.DefaultRegisterer. If the registration fails, NewSummaryVec
// panics.
func NewSummaryVec(opts prometheus.SummaryOpts, labelNames []string) *prometheus.SummaryVec {
	return With(prometheus.DefaultRegisterer).NewSummaryVec(opts, labelNames)
}

// NewHistogram works like the function of the same name in the prometheus
// package but it automatically registers the Histogram with the
// prometheus.DefaultRegisterer. If the registration fails, NewHistogram panics.
func NewHistogram(opts prometheus.HistogramOpts) prometheus.Histogram {
	return With(prometheus.DefaultRegisterer).NewHistogram(opts)
}

// NewHistogramVec works like the function of the same name in the prometheus
// package but it automatically registers the HistogramVec with the
// prometheus.DefaultRegisterer. If the registration fails, NewHistogramVec
// panics.
func NewHistogramVec(opts prometheus.HistogramOpts, labelNames []string) *prometheus.HistogramVec {
	return With(prometheus.DefaultRegisterer).NewHistogramVec(opts, labelNames)
}

// NewUntypedFunc works like the function of the same name in the prometheus
// package but it automatically registers the UntypedFunc with the
// prometheus.DefaultRegisterer. If the registration fails, NewUntypedFunc
// panics.
func NewUntypedFunc(opts prometheus.UntypedOpts, function func() float64) prometheus.UntypedFunc {
	return With(prometheus.DefaultRegisterer).NewUntypedFunc(opts, function)
}

// Factory provides factory methods to create Collectors that are automatically
// registered with a Registerer. Create a Factory with the With function,
// providing a Registerer to auto-register created Collectors with. The zero
// value of a Factory creates Collectors that are not registered with any
// Registerer. All methods of the Factory panic if the registration fails.
type Factory struct {
	r prometheus.Registerer
}

// With creates a Factory using the provided Registerer for registration of the
// created Collectors. If the provided Registerer is nil, the returned Factory
// creates Collectors that are not registered with any Registerer.
func With(r prometheus.Registerer) Factory { return Factory{r} }

// NewCounter works like the function of the same name in the prometheus package
// but it automatically registers the Counter with the Factory's Registerer.
func (f Factory) NewCounter(opts prometheus.CounterOpts) prometheus.Counter {
	c := prometheus.NewCounter(opts)
	if f.r != nil {
		f.r.MustRegister(c)
	}
	return c
}

// NewCounterVec works like the function of the same name in the prometheus
// package but it automatically registers the CounterVec with the Factory's
// Registerer.
func (f Factory) NewCounterVec(opts prometheus.CounterOpts, labelNames []string) *prometheus.CounterVec {
	c := prometheus.NewCounterVec(opts, labelNames)
	if f.r != nil {
		f.r.MustRegister(c)
	}
	return c
}

// NewCounterFunc works like the function of the same name in the prometheus
// package but it automatically registers the CounterFunc with the Factory's
// Registerer.
func (f Factory) NewCounterFunc(opts prometheus.CounterOpts, function func() float64) prometheus.CounterFunc {
	c := prometheus.NewCounterFunc(opts, function)
	if f.r != nil {
		f.r.MustRegister(c)
	}
	return c
}

// NewGauge works like the function of the same name in the prometheus package
// but it automatically registers the Gauge with the Factory's Registerer.
func (f Factory) NewGauge(opts prometheus.GaugeOpts) prometheus.Gauge {
	g := prometheus.NewGauge(opts)
	if f.r != nil {
		f.r.MustRegister(g)
	}
	return g
}

// NewGaugeVec works like the function of the same name in the prometheus
// package but it automatically registers the GaugeVec with the Factory's
// Registerer.
func (f Factory) NewGaugeVec(opts prometheus.GaugeOpts, labelNames []string) *prometheus.GaugeVec {
	g := prometheus.NewGaugeVec(opts, labelNames)
	if f.r != nil {
		f.r.MustRegister(g)
	}
	return g
}

// NewGaugeFunc works like the function of the same name in the prometheus
// package but it automatically registers the GaugeFunc with the Factory's
// Registerer.
func (f Factory) NewGaugeFunc(opts prometheus.GaugeOpts, function func() float64) prometheus.GaugeFunc {
	g := prometheus.NewGaugeFunc(opts, function)
	if f.r != nil {
		f.r.MustRegister(g)
	}
	return g
}

// NewSummary works like the function of the same name in the prometheus package
// but it automatically registers the Summary with the Factory's Registerer.
func (f Factory) NewSummary(opts prometheus.SummaryOpts) prometheus.Summary {
	s := prometheus.NewSummary(opts)
	if f.r != nil {
		f.r.MustRegister(s)
	}
	return s
}

// NewSummaryVec works like the function of the same name in the prometheus
// package but it automatically registers the SummaryVec with the Factory's
// Registerer.
func (f Factory) NewSummaryVec(opts prometheus.SummaryOpts, labelNames []string) *prometheus.SummaryVec {
	s := prometheus.NewSummaryVec(opts, labelNames)
	if f.r != nil {
		f.r.MustRegister(s)
	}
	return s
}

// NewHistogram works like the function of the same name in the prometheus
// package but it automatically registers the Histogram with the Factory's
// Registerer.
func (f Factory) NewHistogram(opts prometheus.HistogramOpts) prometheus.Histogram {
	h := prometheus.NewHistogram(opts)
	if f.r != nil {
		f.r.MustRegister(h)
	}
	return h
}

// NewHistogramVec works like the function of the same name in the prometheus
// package but it automatically registers the HistogramVec with the Factory's
// Registerer.
func (f Factory) NewHistogramVec(opts prometheus.HistogramOpts, labelNames []string) *prometheus.HistogramVec {
	h := prometheus.NewHistogramVec(opts, labelNames)
	if f.r != nil {
		f.r.MustRegister(h)
	}
	return h
}

// NewUntypedFunc works like the function of the same name in the prometheus
// package but it automatically registers the UntypedFunc with the Factory's
// Registerer.
func (f Factory) NewUntypedFunc(opts prometheus.UntypedOpts, function func() float64) prometheus.UntypedFunc {
	u := prometheus.NewUntypedFunc(opts, function)
	if f.r != nil {
		f.r.MustRegister(u)
	}
	return u
}
