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

package time // import "go.opentelemetry.io/otel/sdk/metric/controller/time"

import (
	"time"
	lib "time"
)

// Several types below are created to match "github.com/benbjohnson/clock"
// so that it remains a test-only dependency.

type Clock interface {
	Now() lib.Time
	Ticker(duration lib.Duration) Ticker
}

type Ticker interface {
	Stop()
	C() <-chan lib.Time
}

type RealClock struct {
}

type RealTicker struct {
	ticker *lib.Ticker
}

var _ Clock = RealClock{}
var _ Ticker = RealTicker{}

func (RealClock) Now() time.Time {
	return time.Now()
}

func (RealClock) Ticker(period time.Duration) Ticker {
	return RealTicker{time.NewTicker(period)}
}

func (t RealTicker) Stop() {
	t.ticker.Stop()
}

func (t RealTicker) C() <-chan time.Time {
	return t.ticker.C
}
