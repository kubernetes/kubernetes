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

	// MaxAnnotationEventsPerSpan is max number of annotation events per span
	MaxAnnotationEventsPerSpan int

	// MaxMessageEventsPerSpan is max number of message events per span
	MaxMessageEventsPerSpan int

	// MaxAnnotationEventsPerSpan is max number of attributes per span
	MaxAttributesPerSpan int

	// MaxLinksPerSpan is max number of links per span
	MaxLinksPerSpan int
}

var configWriteMu sync.Mutex

const (
	// DefaultMaxAnnotationEventsPerSpan is default max number of annotation events per span
	DefaultMaxAnnotationEventsPerSpan = 32

	// DefaultMaxMessageEventsPerSpan is default max number of message events per span
	DefaultMaxMessageEventsPerSpan = 128

	// DefaultMaxAttributesPerSpan is default max number of attributes per span
	DefaultMaxAttributesPerSpan = 32

	// DefaultMaxLinksPerSpan is default max number of links per span
	DefaultMaxLinksPerSpan = 32
)

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
	if cfg.MaxAnnotationEventsPerSpan > 0 {
		c.MaxAnnotationEventsPerSpan = cfg.MaxAnnotationEventsPerSpan
	}
	if cfg.MaxMessageEventsPerSpan > 0 {
		c.MaxMessageEventsPerSpan = cfg.MaxMessageEventsPerSpan
	}
	if cfg.MaxAttributesPerSpan > 0 {
		c.MaxAttributesPerSpan = cfg.MaxAttributesPerSpan
	}
	if cfg.MaxLinksPerSpan > 0 {
		c.MaxLinksPerSpan = cfg.MaxLinksPerSpan
	}
	config.Store(&c)
}
