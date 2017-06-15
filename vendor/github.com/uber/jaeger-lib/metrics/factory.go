// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package metrics

// Factory creates new metrics
type Factory interface {
	Counter(name string, tags map[string]string) Counter
	Timer(name string, tags map[string]string) Timer
	Gauge(name string, tags map[string]string) Gauge

	// Namespace returns a nested metrics factory.
	Namespace(name string, tags map[string]string) Factory
}

// NullFactory is a metrics factory that returns NullCounter, NullTimer, and NullGauge.
var NullFactory Factory = nullFactory{}

type nullFactory struct{}

func (nullFactory) Counter(name string, tags map[string]string) Counter   { return NullCounter }
func (nullFactory) Timer(name string, tags map[string]string) Timer       { return NullTimer }
func (nullFactory) Gauge(name string, tags map[string]string) Gauge       { return NullGauge }
func (nullFactory) Namespace(name string, tags map[string]string) Factory { return NullFactory }
