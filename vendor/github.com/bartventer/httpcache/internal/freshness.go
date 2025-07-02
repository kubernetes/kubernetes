// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"cmp"
	"log/slog"
	"net/http"
	"strconv"
	"time"
)

type Age struct {
	Value     time.Duration // Age of the cached response (RFC9111 §4.2.3)
	Timestamp time.Time     // Time when the age was calculated
}

var _ slog.LogValuer = (*Age)(nil)

func (a Age) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Duration("value", a.Value),
		slog.Time("timestamp", a.Timestamp),
	)
}

type Freshness struct {
	IsStale    bool          // Whether the response is stale
	Age        *Age          // Current age (seconds) of the response (RFC9111 §4.2.3)
	UsefulLife time.Duration // Freshness lifetime (seconds) of the response (RFC9111 §4.2.1)
}

var _ slog.LogValuer = (*Freshness)(nil)

func (f Freshness) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Bool("is_stale", f.IsStale),
		slog.Any("age", cmp.Or(f.Age, &Age{Value: 0, Timestamp: time.Time{}})),
		slog.Duration("useful_life", f.UsefulLife),
	)
}

// heuristicFreshness calculates freshness lifetime using heuristics (10% of (date - last-modified)),
// per RFC9111 §4.2.2.
func heuristicFreshness(h http.Header, date time.Time) time.Duration {
	lastMod, ok := RawTime(h.Get("Last-Modified")).Value()
	if !ok || !lastMod.Before(date) {
		return 0
	}
	delta := date.Sub(lastMod)
	return time.Duration(float64(delta) * 0.1).Round(time.Second)
}

// calculateCurrentAge implements RFC9111 §4.2.3 for calculating the current age of a cached response
// based on the Age header, response time, request time, and the current clock time.
func calculateCurrentAge(
	clock Clock,
	h http.Header,
	date, requestTime, responseTime time.Time,
) *Age {
	ageVal := 0
	if ageStr := h.Get("Age"); ageStr != "" {
		ageVal, _ = strconv.Atoi(ageStr)
	}
	apparentAge := max(responseTime.Sub(date), 0)
	responseDelay := max(responseTime.Sub(requestTime), 0)
	correctedAgeValue := time.Duration(ageVal)*time.Second + responseDelay
	correctedInitialAge := max(apparentAge, correctedAgeValue)
	residentTime := max(clock.Since(responseTime), 0)
	return &Age{
		Value:     correctedInitialAge + residentTime,
		Timestamp: clock.Now(),
	}
}

const maxDuration = 1<<63 - 1

// FreshnessCalculator describes the interface implemented by types that can
// calculate the freshness of a cached response based on request and response
// cache control directives according to RFC 9111 §4.2.
type FreshnessCalculator interface {
	// CalculateFreshness calculates the freshness of a cached response
	// based on the request and response cache control directives.
	CalculateFreshness(
		resp *Response,
		reqCC CCRequestDirectives,
		resCC CCResponseDirectives,
	) *Freshness
}

func NewFreshnessCalculator(clock Clock) *freshnessCalculator {
	return &freshnessCalculator{clock}
}

type freshnessCalculator struct {
	clock Clock // Clock interface to get current time
}

// calculateFreshnessStatus determines if a cached response is fresh or stale based on RFC9111 §4.2.
// It considers request and response headers, cache directives, and timestamps.
//
//nolint:cyclop // Cyclomatic complexity is high due to multiple conditions, but it's necessary for RFC compliance.
func (f *freshnessCalculator) CalculateFreshness(
	entry *Response,
	reqCC CCRequestDirectives,
	resCC CCResponseDirectives,
) *Freshness {
	if reqMaxAge, ok := reqCC.MaxAge(); ok && reqMaxAge == 0 {
		return &Freshness{
			IsStale:    true,
			Age:        &Age{Value: 0, Timestamp: f.clock.Now()},
			UsefulLife: 0,
		}
	}

	resp := entry.Data
	date := entry.DateHeader()
	currentAge := calculateCurrentAge(
		f.clock,
		resp.Header,
		date,
		entry.RequestedAt,
		entry.ReceivedAt,
	)

	// Freshness lifetime (private cache: ignore s-maxage)
	usefulLife := time.Duration(0)
	if maxAge, ok := resCC.MaxAge(); ok && maxAge >= 0 {
		usefulLife = maxAge // Response is fresh for max-age seconds
	}

	if usefulLife == 0 {
		expires, found, valid := entry.ExpiresHeader()
		switch {
		case valid && expires.After(date):
			// Use Expires header if available
			usefulLife = expires.Sub(date)
		case !found && (isHeuristicallyCacheableCode(resp.StatusCode) || resCC.Public()):
			// Heuristic fallback if allowed by RFC9111 §4.2.2 (only if expires is not set)
			usefulLife = heuristicFreshness(resp.Header, date)
		}
	}

	if reqMaxAge, ok := reqCC.MaxAge(); ok && reqMaxAge > 0 {
		usefulLife = min(usefulLife, reqMaxAge) // Client prefers a response no older than max-age
	}
	if reqMinFresh, ok := reqCC.MinFresh(); ok && reqMinFresh > 0 &&
		(usefulLife-currentAge.Value) < reqMinFresh {
		return &Freshness{IsStale: true, Age: currentAge, UsefulLife: usefulLife}
	}

	maxStale := time.Duration(0)
	if reqMaxStaleStr, ok := reqCC.MaxStale(); ok {
		if reqMaxStaleStr == "" {
			maxStale = maxDuration // accept any staleness
		} else if reqMaxStale, valid := reqMaxStaleStr.Value(); valid && reqMaxStale >= 0 {
			maxStale = reqMaxStale
		}
	}

	isStale := currentAge.Value >= usefulLife
	// If max-stale present, allow extra staleness
	if isStale && maxStale > 0 && currentAge.Value < max(usefulLife+maxStale, maxStale) {
		isStale = false
	}

	return &Freshness{IsStale: isStale, Age: currentAge, UsefulLife: usefulLife}
}
