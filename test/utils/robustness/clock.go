/*
Copyright The Kubernetes Authors.

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

package robustness

import (
	"context"
	"time"

	"k8s.io/utils/clock"
)

// ShiftTime is a ClockFault that advances the wrapped clock by Offset. Valid only
// at clock.* points.
type ShiftTime struct {
	Offset time.Duration
}

func (a ShiftTime) ApplyClock() ClockVerdict {
	return ClockVerdict{Shift: a.Offset}
}

// FaultInjectingClock wraps k8s.io/utils/clock.Clock and intercepts Now() queries.
type FaultInjectingClock struct {
	clock.Clock
	registry *FaultRegistry
	name     string
}

// NewFaultInjectingClock creates a wrapped clock.Clock hooked to the registry.
func NewFaultInjectingClock(realClock clock.Clock, registry *FaultRegistry, name string) clock.Clock {
	return &FaultInjectingClock{
		Clock:    realClock,
		registry: registry,
		name:     name,
	}
}

func (c *FaultInjectingClock) Now() time.Time {
	realNow := c.Clock.Now()
	shift := c.registry.ResolveClock(context.Background(), ClockFacts{Clock: c.name})
	return realNow.Add(shift)
}

func (c *FaultInjectingClock) Since(t time.Time) time.Duration {
	return c.Now().Sub(t)
}

func (c *FaultInjectingClock) Sleep(d time.Duration) {
	c.Clock.Sleep(d)
}

func (c *FaultInjectingClock) After(d time.Duration) <-chan time.Time {
	return c.Clock.After(d)
}

func (c *FaultInjectingClock) NewTimer(d time.Duration) clock.Timer {
	return c.Clock.NewTimer(d)
}

func (c *FaultInjectingClock) Tick(d time.Duration) <-chan time.Time {
	return c.Clock.Tick(d)
}
