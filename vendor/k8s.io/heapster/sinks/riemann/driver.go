// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package riemann

import (
	"fmt"
	"net/url"
	"reflect"
	"runtime"
	"strconv"
	"time"

	riemann_api "github.com/bigdatadev/goryman"
	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/sinks/util"
	kube_api "k8s.io/kubernetes/pkg/api"
)

// The basic internal type for this package; it obeys the sink_api.ExternalSink
// interface

type riemannSink struct {
	client riemannClient
	config riemannConfig
	ci     util.ClientInitializer
}

type riemannConfig struct {
	host        string
	ttl         float32
	state       string
	tags        []string
	storeEvents bool
}

func init() {
	extpoints.SinkFactories.Register(CreateRiemannSink, "riemann")
}

func CreateRiemannSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	c := riemannConfig{
		host:        "riemann-heapster:5555",
		ttl:         60.0,
		state:       "",
		tags:        make([]string, 0),
		storeEvents: true,
	}
	if len(uri.Host) > 0 {
		c.host = uri.Host
	}
	options := uri.Query()
	if len(options["ttl"]) > 0 {
		var ttl, err = strconv.ParseFloat(options["ttl"][0], 32)
		if err != nil {
			return nil, err
		}
		c.ttl = float32(ttl)
	}
	if len(options["state"]) > 0 {
		c.state = options["state"][0]
	}
	if len(options["tags"]) > 0 {
		c.tags = options["tags"]
	}
	if len(options["storeEvents"]) > 0 {
		var storeEvents, err = strconv.ParseBool(options["storeEvents"][0])
		if err != nil {
			return nil, err
		}
		c.storeEvents = storeEvents
	}
	glog.Infof("Riemann sink URI: '%+v', host: '%+v', options: '%+v', ", uri, c.host, options)
	rs := &riemannSink{
		client: riemann_api.NewGorymanClient(c.host),
		config: c,
	}
	rs.ci = util.NewClientInitializer("riemann", rs.setupRiemannClient, rs.ping, 10*time.Second)
	runtime.SetFinalizer(rs.client, func(c riemannClient) { c.Close() })
	return []sink_api.ExternalSink{rs}, nil
}

func (rs *riemannSink) setupRiemannClient() error {
	return rs.client.Connect()
}

func (rs *riemannSink) ping() error {
	_, err := rs.client.QueryEvents("false")
	if err != nil {
		rs.client.Close()
	}
	return err
}

// Abstracted for testing: this package works against any client that obeys the
// interface contract exposed by the goryman Riemann client

type riemannClient interface {
	Connect() error
	Close() error
	SendEvent(*riemann_api.Event) error
	QueryEvents(string) ([]riemann_api.Event, error)
}

func (rs *riemannSink) kubeEventToRiemannEvent(event kube_api.Event) (*riemann_api.Event, error) {
	glog.V(4).Infof("riemannSink.kubeEventToRiemannEvent(%+v)", event)

	var rv = riemann_api.Event{
		// LastTimestamp: the time at which the most recent occurance of this
		// event was recorded
		Time: event.LastTimestamp.UTC().Unix(),
		// Source: A short, machine-understantable string that gives the
		// component reporting this event
		Host:       event.Source.Host,
		Attributes: event.Labels, //TODO(jfoy) consider event.Annotations as well
		Service:    event.InvolvedObject.Name,
		// Reason: a short, machine-understandable string that gives the reason
		// for this event being generated
		// Message: A human-readable description of the status of this operation.
		Description: event.Reason + ": " + event.Message,
		// Count: the number of times this event has occurred
		Metric: event.Count,
		Ttl:    rs.config.ttl,
		State:  rs.config.state,
		Tags:   rs.config.tags,
	}
	glog.V(4).Infof("Returning Riemann event: %+v", rv)
	return &rv, nil
}

func (rs *riemannSink) timeseriesToRiemannEvent(ts sink_api.Timeseries) (*riemann_api.Event, error) {
	glog.V(4).Infof("riemannSink.timeseriesToRiemannEvent(%+v)", ts)

	var service string
	if ts.MetricDescriptor.Units.String() == "" {
		service = ts.Point.Name
	} else {
		service = ts.Point.Name + "_" + ts.MetricDescriptor.Units.String()
	}

	var rv = riemann_api.Event{
		Time:        ts.Point.End.UTC().Unix(),
		Service:     service,
		Host:        ts.Point.Labels[sink_api.LabelHostname.Key],
		Description: ts.MetricDescriptor.Description,
		Attributes:  ts.Point.Labels,
		Metric:      valueToMetric(ts.Point.Value),
		Ttl:         rs.config.ttl,
		State:       rs.config.state,
		Tags:        rs.config.tags,
	}
	glog.V(4).Infof("Returning Riemann event: %+v", rv)
	return &rv, nil
}

func valueToMetric(i interface{}) interface{} {
	value := reflect.ValueOf(i)
	kind := reflect.TypeOf(value.Interface()).Kind()

	switch kind {
	case reflect.Int, reflect.Int64:
		return int(value.Int())
	default:
		return i
	}
}

// Error is the type of a parse error; it satisfies the error interface.
type Error string

func (e Error) Error() string {
	return string(e)
}

func (rs *riemannSink) sendEvents(events []*riemann_api.Event) error {
	var result error
	for _, event := range events {
		defer func() {
			if res := recover(); res != nil {
				result = res.(Error)
			}
		}()
		err := rs.client.SendEvent(event)
		// FIXME handle multiple errors
		if err != nil {
			glog.V(2).Infof("Failed sending event to Riemann: %+v: %+v", event, err)
			result = err
		}
	}
	return result
}

// Riemann does not pre-register metrics, so Register() is a no-op
func (rs *riemannSink) Register(descriptor []sink_api.MetricDescriptor) error { return nil }

// Like Register
func (rs *riemannSink) Unregister(metrics []sink_api.MetricDescriptor) error { return nil }

// Send a collection of Timeseries to Riemann
func (rs *riemannSink) StoreTimeseries(inputs []sink_api.Timeseries) error {
	if !rs.ci.Done() {
		// Riemann backend isn't available.
		glog.V(4).Infof("Skipping timeseries data. Riemann backend isn't available. ")
		return nil
	}
	var events []*riemann_api.Event
	for _, input := range inputs {
		var event, err = rs.timeseriesToRiemannEvent(input)
		if err != nil {
			return err
		}
		events = append(events, event)
	}
	return rs.sendEvents(events)
}

// Send a collection of Kubernetes Events to Riemann
func (rs *riemannSink) StoreEvents(inputs []kube_api.Event) error {
	if !rs.ci.Done() {
		// Riemann backend isn't available.
		glog.V(4).Infof("Skipping events data. Riemann backend isn't available. ")
		return nil
	}
	if !rs.config.storeEvents {
		return nil
	}

	var events []*riemann_api.Event
	for _, input := range inputs {
		var event, err = rs.kubeEventToRiemannEvent(input)
		if err != nil {
			return err
		}
		events = append(events, event)
	}
	return rs.sendEvents(events)
}

// Return debug information specific to Riemann
func (rs *riemannSink) DebugInfo() string { return fmt.Sprintf("%s", rs.client) }

// Return a user-friendly string describing the sink
func (rs *riemannSink) Name() string { return "Riemann" }
