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
	riemann_api "github.com/bigdatadev/goryman"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type mockClient struct {
	connected bool
	closed    bool
}

func (rs *mockClient) Connect() error                                    { rs.connected = true; return nil }
func (rs *mockClient) Close() error                                      { rs.closed = true; return nil }
func (rs *mockClient) SendEvent(*riemann_api.Event) error                { return nil }
func (rs *mockClient) QueryEvents(_ string) ([]riemann_api.Event, error) { return nil, nil }
func getMock() riemannClient                                             { return &mockClient{} }

var _ = Describe("Driver", func() {
	Describe("Name", func() {
		It("gives a user-friendly string describing the sink", func() {
			var subject = &riemannSink{getMock(), riemannConfig{}, nil}

			var name = subject.Name()

			Expect(name).To(Equal("Riemann"))
		})
	})

	Describe("Register", func() {
		// func (rs *riemannSink) Register(descriptor []sink_api.MetricDescriptor) error { return nil }
		It("registers a metric with Riemann (no-op)", func() {})
	})

	PDescribe("StoreTimeseries", func() {
		// func (rs *riemannSink) StoreTimeseries(ts []sink_api.Timeseries) error { return nil }
		It("sends a collection of Timeseries to Riemann", func() {})
	})

	PDescribe("StoreEvents", func() {
		// func (rs *riemannSink) StoreEvents(event []kube_api.Event) error { return nil }
		It("sends a collection of Kubernetes Events to Riemann", func() {})
	})

	Describe("DebugInfo", func() {
		// func (rs *riemannSink) DebugInfo() string { return "" }
		It("gives debug information specific to Riemann", func() {
			var subject = &riemannSink{getMock(), riemannConfig{}, nil}
			var debugInfo = subject.DebugInfo()
			Expect(debugInfo).ToNot(Equal(""))
		})
	})
})
