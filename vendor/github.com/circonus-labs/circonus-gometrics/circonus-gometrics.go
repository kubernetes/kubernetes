// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package circonusgometrics provides instrumentation for your applications in the form
// of counters, gauges and histograms and allows you to publish them to
// Circonus
//
// Counters
//
// A counter is a monotonically-increasing, unsigned, 64-bit integer used to
// represent the number of times an event has occurred. By tracking the deltas
// between measurements of a counter over intervals of time, an aggregation
// layer can derive rates, acceleration, etc.
//
// Gauges
//
// A gauge returns instantaneous measurements of something using signed, 64-bit
// integers. This value does not need to be monotonic.
//
// Histograms
//
// A histogram tracks the distribution of a stream of values (e.g. the number of
// seconds it takes to handle requests).  Circonus can calculate complex
// analytics on these.
//
// Reporting
//
// A period push to a Circonus httptrap is confgurable.
package circonusgometrics

import (
	"errors"
	"io/ioutil"
	"log"
	"os"
	"sync"
	"time"

	"github.com/circonus-labs/circonus-gometrics/api"
	"github.com/circonus-labs/circonus-gometrics/checkmgr"
)

const (
	defaultFlushInterval = "10s" // 10 * time.Second
)

// Config options for circonus-gometrics
type Config struct {
	Log   *log.Logger
	Debug bool

	// API, Check and Broker configuration options
	CheckManager checkmgr.Config

	// how frequenly to submit metrics to Circonus, default 10 seconds
	Interval string
}

// CirconusMetrics state
type CirconusMetrics struct {
	Log           *log.Logger
	Debug         bool
	flushInterval time.Duration
	flushing      bool
	flushmu       sync.Mutex
	check         *checkmgr.CheckManager

	counters map[string]uint64
	cm       sync.Mutex

	counterFuncs map[string]func() uint64
	cfm          sync.Mutex

	gauges map[string]string
	gm     sync.Mutex

	gaugeFuncs map[string]func() int64
	gfm        sync.Mutex

	histograms map[string]*Histogram
	hm         sync.Mutex

	text map[string]string
	tm   sync.Mutex

	textFuncs map[string]func() string
	tfm       sync.Mutex
}

// NewCirconusMetrics returns a CirconusMetrics instance
func NewCirconusMetrics(cfg *Config) (*CirconusMetrics, error) {

	if cfg == nil {
		return nil, errors.New("Invalid configuration (nil).")
	}

	cm := &CirconusMetrics{
		counters:     make(map[string]uint64),
		counterFuncs: make(map[string]func() uint64),
		gauges:       make(map[string]string),
		gaugeFuncs:   make(map[string]func() int64),
		histograms:   make(map[string]*Histogram),
		text:         make(map[string]string),
		textFuncs:    make(map[string]func() string),
	}

	cm.Debug = cfg.Debug
	if cm.Debug {
		if cfg.Log == nil {
			cm.Log = log.New(os.Stderr, "", log.LstdFlags)
		} else {
			cm.Log = cfg.Log
		}
	}
	if cm.Log == nil {
		cm.Log = log.New(ioutil.Discard, "", log.LstdFlags)
	}

	fi := defaultFlushInterval
	if cfg.Interval != "" {
		fi = cfg.Interval
	}

	dur, err := time.ParseDuration(fi)
	if err != nil {
		return nil, err
	}
	cm.flushInterval = dur

	cfg.CheckManager.Debug = cm.Debug
	cfg.CheckManager.Log = cm.Log

	check, err := checkmgr.NewCheckManager(&cfg.CheckManager)
	if err != nil {
		return nil, err
	}
	cm.check = check

	if _, err := cm.check.GetTrap(); err != nil {
		return nil, err
	}

	return cm, nil
}

// Start initializes the CirconusMetrics instance based on
// configuration settings and sets the httptrap check url to
// which metrics should be sent. It then starts a perdiodic
// submission process of all metrics collected.
func (m *CirconusMetrics) Start() {
	go func() {
		for _ = range time.NewTicker(m.flushInterval).C {
			m.Flush()
		}
	}()
}

// Flush metrics kicks off the process of sending metrics to Circonus
func (m *CirconusMetrics) Flush() {
	if m.flushing {
		return
	}
	m.flushmu.Lock()
	m.flushing = true
	m.flushmu.Unlock()

	if m.Debug {
		m.Log.Println("[DEBUG] Flushing metrics")
	}

	// check for new metrics and enable them automatically
	newMetrics := make(map[string]*api.CheckBundleMetric)

	counters, gauges, histograms, text := m.snapshot()
	output := make(map[string]interface{})
	for name, value := range counters {
		send := m.check.IsMetricActive(name)
		if !send && m.check.ActivateMetric(name) {
			send = true
			newMetrics[name] = &api.CheckBundleMetric{
				Name:   name,
				Type:   "numeric",
				Status: "active",
			}
		}
		if send {
			output[name] = map[string]interface{}{
				"_type":  "n",
				"_value": value,
			}
		}
	}

	for name, value := range gauges {
		send := m.check.IsMetricActive(name)
		if !send && m.check.ActivateMetric(name) {
			send = true
			newMetrics[name] = &api.CheckBundleMetric{
				Name:   name,
				Type:   "numeric",
				Status: "active",
			}
		}
		if send {
			output[name] = map[string]interface{}{
				"_type":  "n",
				"_value": value,
			}
		}
	}

	for name, value := range histograms {
		send := m.check.IsMetricActive(name)
		if !send && m.check.ActivateMetric(name) {
			send = true
			newMetrics[name] = &api.CheckBundleMetric{
				Name:   name,
				Type:   "histogram",
				Status: "active",
			}
		}
		if send {
			output[name] = map[string]interface{}{
				"_type":  "n",
				"_value": value.DecStrings(),
			}
		}
	}

	for name, value := range text {
		send := m.check.IsMetricActive(name)
		if !send && m.check.ActivateMetric(name) {
			send = true
			newMetrics[name] = &api.CheckBundleMetric{
				Name:   name,
				Type:   "text",
				Status: "active",
			}
		}
		if send {
			output[name] = map[string]interface{}{
				"_type":  "s",
				"_value": value,
			}
		}
	}

	m.submit(output, newMetrics)

	m.flushmu.Lock()
	m.flushing = false
	m.flushmu.Unlock()
}
