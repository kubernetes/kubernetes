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

package workqueue

type QueueConfig struct {
	metricsProvider MetricsProvider
}

// QueueOption is an interface for applying queue configuration options.
type QueueOption interface {
	apply(*QueueConfig)
}

type optionFunc func(*QueueConfig)

func (fn optionFunc) apply(config *QueueConfig) {
	fn(config)
}

var _ QueueOption = optionFunc(nil)

// NewConfig creates a new QueueConfig and applies all the given options.
func NewConfig(opts ...QueueOption) *QueueConfig {
	config := &QueueConfig{}
	for _, o := range opts {
		o.apply(config)
	}
	return config
}

// WithMetricsProvider allows specifying a metrics provider to use for the queue
// instead of the global provider.
func WithMetricsProvider(provider MetricsProvider) QueueOption {
	return optionFunc(func(config *QueueConfig) {
		config.metricsProvider = provider
	})
}
