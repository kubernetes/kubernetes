// Copyright 2018, OpenCensus Authors
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

package trace

import (
	"sync"

	"go.opencensus.io/trace/internal"
)

// Config represents the global tracing configuration.
type Config struct {
	// DefaultSampler is the default sampler used when creating new spans.
	DefaultSampler Sampler

	// IDGenerator is for internal use only.
	IDGenerator internal.IDGenerator
}

var configWriteMu sync.Mutex

// ApplyConfig applies changes to the global tracing configuration.
//
// Fields not provided in the given config are going to be preserved.
func ApplyConfig(cfg Config) {
	configWriteMu.Lock()
	defer configWriteMu.Unlock()
	c := *config.Load().(*Config)
	if cfg.DefaultSampler != nil {
		c.DefaultSampler = cfg.DefaultSampler
	}
	if cfg.IDGenerator != nil {
		c.IDGenerator = cfg.IDGenerator
	}
	config.Store(&c)
}
