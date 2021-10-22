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
	"reflect"
	"testing"
)

func TestApplyConfig(t *testing.T) {
	cfg := config.Load().(*Config)
	defaultCfg := Config{
		DefaultSampler:             cfg.DefaultSampler,
		IDGenerator:                cfg.IDGenerator,
		MaxAttributesPerSpan:       DefaultMaxAttributesPerSpan,
		MaxAnnotationEventsPerSpan: DefaultMaxAnnotationEventsPerSpan,
		MaxMessageEventsPerSpan:    DefaultMaxMessageEventsPerSpan,
		MaxLinksPerSpan:            DefaultMaxLinksPerSpan,
	}
	testCases := []struct {
		name    string
		newCfg  Config
		wantCfg Config
	}{
		{
			name:    "Initialize to default config",
			newCfg:  defaultCfg,
			wantCfg: defaultCfg,
		},
		{
			name:    "Empty Config",
			newCfg:  Config{},
			wantCfg: defaultCfg,
		},
		{
			name: "Valid non-default config",
			newCfg: Config{
				MaxAttributesPerSpan:       1,
				MaxAnnotationEventsPerSpan: 2,
				MaxMessageEventsPerSpan:    3,
				MaxLinksPerSpan:            4,
			},
			wantCfg: Config{
				DefaultSampler:             cfg.DefaultSampler,
				IDGenerator:                cfg.IDGenerator,
				MaxAttributesPerSpan:       1,
				MaxAnnotationEventsPerSpan: 2,
				MaxMessageEventsPerSpan:    3,
				MaxLinksPerSpan:            4,
			},
		},
		{
			name: "Partially invalid config",
			newCfg: Config{
				MaxAttributesPerSpan:       -1,
				MaxAnnotationEventsPerSpan: 3,
				MaxMessageEventsPerSpan:    -3,
				MaxLinksPerSpan:            5,
			},
			wantCfg: Config{
				DefaultSampler:             cfg.DefaultSampler,
				IDGenerator:                cfg.IDGenerator,
				MaxAttributesPerSpan:       1,
				MaxAnnotationEventsPerSpan: 3,
				MaxMessageEventsPerSpan:    3,
				MaxLinksPerSpan:            5,
			},
		},
	}

	for i, tt := range testCases {
		newCfg := tt.newCfg
		ApplyConfig(newCfg)
		gotCfg := config.Load().(*Config)
		wantCfg := tt.wantCfg

		if got, want := reflect.ValueOf(gotCfg.DefaultSampler).Pointer(), reflect.ValueOf(wantCfg.DefaultSampler).Pointer(); got != want {
			t.Fatalf("testId = %d, testName = %s: config.DefaultSampler = %#v; want %#v", i, tt.name, got, want)
		}
		if got, want := gotCfg.IDGenerator, wantCfg.IDGenerator; got != want {
			t.Fatalf("testId = %d, testName = %s: config.IDGenerator = %#v; want %#v", i, tt.name, got, want)
		}
		if got, want := gotCfg.MaxAttributesPerSpan, wantCfg.MaxAttributesPerSpan; got != want {
			t.Fatalf("testId = %d, testName = %s: config.MaxAttributesPerSpan = %#v; want %#v", i, tt.name, got, want)
		}
		if got, want := gotCfg.MaxLinksPerSpan, wantCfg.MaxLinksPerSpan; got != want {
			t.Fatalf("testId = %d, testName = %s: config.MaxLinksPerSpan = %#v; want %#v", i, tt.name, got, want)
		}
		if got, want := gotCfg.MaxAnnotationEventsPerSpan, wantCfg.MaxAnnotationEventsPerSpan; got != want {
			t.Fatalf("testId = %d, testName = %s: config.MaxAnnotationEventsPerSpan = %#v; want %#v", i, tt.name, got, want)
		}
		if got, want := gotCfg.MaxMessageEventsPerSpan, wantCfg.MaxMessageEventsPerSpan; got != want {
			t.Fatalf("testId = %d, testName = %s: config.MaxMessageEventsPerSpan = %#v; want %#v", i, tt.name, got, want)
		}

	}
}
