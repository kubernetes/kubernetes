// Copyright 2020 Google Inc. All Rights Reserved.
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

// Configuration for perf event manager.
package perf

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"k8s.io/klog/v2"
)

type PerfEvents struct {
	// Core perf events to be measured.
	Core Events `json:"core,omitempty"`

	// Uncore perf events to be measured.
	Uncore Events `json:"uncore,omitempty"`
}

type Events struct {
	// List of perf events' names to be measured.
	Events []Group `json:"events"`

	// List of custom perf events' to be measured. It is impossible to
	// specify some events using their names and in such case you have
	// to provide lower level configuration.
	CustomEvents []CustomEvent `json:"custom_events"`
}

type Event string

type CustomEvent struct {
	// Type of the event. See perf_event_attr documentation
	// at man perf_event_open.
	Type uint32 `json:"type,omitempty"`

	// Symbolically formed event like:
	// pmu/config=PerfEvent.Config[0],config1=PerfEvent.Config[1],config2=PerfEvent.Config[2]
	// as described in man perf-stat.
	Config Config `json:"config"`

	// Human readable name of metric that will be created from the event.
	Name Event `json:"name"`
}

type Config []uint64

func (c *Config) UnmarshalJSON(b []byte) error {
	config := []string{}
	err := json.Unmarshal(b, &config)
	if err != nil {
		klog.Errorf("Unmarshalling %s into slice of strings failed: %q", b, err)
		return fmt.Errorf("unmarshalling %s into slice of strings failed: %q", b, err)
	}
	intermediate := []uint64{}
	for _, v := range config {
		uintValue, err := strconv.ParseUint(v, 0, 64)
		if err != nil {
			klog.Errorf("Parsing %#v into uint64 failed: %q", v, err)
			return fmt.Errorf("parsing %#v into uint64 failed: %q", v, err)
		}
		intermediate = append(intermediate, uintValue)
	}
	*c = intermediate
	return nil
}

func parseConfig(file *os.File) (events PerfEvents, err error) {
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&events)
	if err != nil {
		err = fmt.Errorf("unable to load perf events configuration from %q: %q", file.Name(), err)
		return
	}
	return
}

type Group struct {
	events []Event
	array  bool
}

func (g *Group) UnmarshalJSON(b []byte) error {
	var jsonObj interface{}
	err := json.Unmarshal(b, &jsonObj)
	if err != nil {
		return err
	}
	switch obj := jsonObj.(type) {
	case string:
		*g = Group{
			events: []Event{Event(obj)},
			array:  false,
		}
		return nil
	case []interface{}:
		group := Group{
			events: make([]Event, 0, len(obj)),
			array:  true,
		}
		for _, v := range obj {
			value, ok := v.(string)
			if !ok {
				return fmt.Errorf("cannot unmarshal %v", value)
			}
			group.events = append(group.events, Event(value))
		}
		*g = group
		return nil
	}
	return fmt.Errorf("unsupported type")
}
