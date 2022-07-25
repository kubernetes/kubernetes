// Copyright The OpenTelemetry Authors
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

package basic // import "go.opentelemetry.io/otel/sdk/metric/processor/basic"

// Config contains the options for configuring a basic metric processor.
type Config struct {
	// Memory controls whether the processor remembers metric
	// instruments and label sets that were previously reported.
	// When Memory is true, CheckpointSet.ForEach() will visit
	// metrics that were not updated in the most recent interval.
	Memory bool
}

type Option interface {
	ApplyProcessor(*Config)
}

// WithMemory sets the memory behavior of a Processor.  If this is
// true, the processor will report metric instruments and label sets
// that were previously reported but not updated in the most recent
// interval.
func WithMemory(memory bool) Option {
	return memoryOption(memory)
}

type memoryOption bool

func (m memoryOption) ApplyProcessor(config *Config) {
	config.Memory = bool(m)
}
