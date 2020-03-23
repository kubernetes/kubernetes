// Copyright 2014 The Prometheus Authors
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

package prometheus

// UntypedOpts is an alias for Opts. See there for doc comments.
type UntypedOpts Opts

// UntypedFunc works like GaugeFunc but the collected metric is of type
// "Untyped". UntypedFunc is useful to mirror an external metric of unknown
// type.
//
// To create UntypedFunc instances, use NewUntypedFunc.
type UntypedFunc interface {
	Metric
	Collector
}

// NewUntypedFunc creates a new UntypedFunc based on the provided
// UntypedOpts. The value reported is determined by calling the given function
// from within the Write method. Take into account that metric collection may
// happen concurrently. If that results in concurrent calls to Write, like in
// the case where an UntypedFunc is directly registered with Prometheus, the
// provided function must be concurrency-safe.
func NewUntypedFunc(opts UntypedOpts, function func() float64) UntypedFunc {
	return newValueFunc(NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		nil,
		opts.ConstLabels,
	), UntypedValue, function)
}
