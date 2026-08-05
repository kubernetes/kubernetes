// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/internal/errorhandler"
)

const (
	baggageHeader = "baggage"

	maxParseErrors = 5

	// W3C Baggage specification limits.
	// https://www.w3.org/TR/baggage/#limits
	maxMembers               = 64
	maxBytesPerBaggageString = 8192
)

// handleExtractErrOnce limits error reporting for attacker-controlled baggage headers
// to one process-wide emission, preventing repeated extraction from flooding logs.
var handleExtractErrOnce sync.Once

// Baggage is a propagator that supports the W3C Baggage format.
//
// This propagates user-defined baggage associated with a trace. The complete
// specification is defined at https://www.w3.org/TR/baggage/.
type Baggage struct{}

var _ TextMapPropagator = Baggage{}

// Inject sets baggage key-values from ctx into the carrier.
func (Baggage) Inject(ctx context.Context, carrier TextMapCarrier) {
	bStr := baggage.FromContext(ctx).String()
	if bStr != "" {
		carrier.Set(baggageHeader, bStr)
	}
}

// Extract returns a copy of parent with the baggage from the carrier added.
// If carrier implements [ValuesGetter] (e.g. [HeaderCarrier]), Values is invoked
// for multiple values extraction. Otherwise, Get is called.
func (Baggage) Extract(parent context.Context, carrier TextMapCarrier) context.Context {
	if multiCarrier, ok := carrier.(ValuesGetter); ok {
		return extractMultiBaggage(parent, multiCarrier)
	}
	return extractSingleBaggage(parent, carrier)
}

// Fields returns the keys who's values are set with Inject.
func (Baggage) Fields() []string {
	return []string{baggageHeader}
}

func extractSingleBaggage(parent context.Context, carrier TextMapCarrier) context.Context {
	bStr := carrier.Get(baggageHeader)
	if bStr == "" {
		return parent
	}

	bag, err := baggage.Parse(bStr)
	if err != nil {
		handleExtractErrOnce.Do(func() {
			errorhandler.GetErrorHandler().Handle(err)
		})
	}
	if bag.Len() == 0 {
		return parent
	}
	return baggage.ContextWithBaggage(parent, bag)
}

func extractMultiBaggage(parent context.Context, carrier ValuesGetter) context.Context {
	bVals := carrier.Values(baggageHeader)
	if len(bVals) == 0 {
		return parent
	}

	var members []baggage.Member
	var totalBytes int
	var parseErrors int
	var truncateErr error
	for i, bStr := range bVals {
		if i > 0 {
			totalBytes++ // comma separator between combined header values
		}
		totalBytes += len(bStr)
		if totalBytes > maxBytesPerBaggageString {
			// Per the W3C Baggage spec, the byte limit applies to the
			// combination of all baggage headers, not each header
			// individually. Mirror the single-header behavior of
			// reporting the error and returning the parent context
			// with no baggage attached.
			handleExtractErrOnce.Do(func() {
				errorhandler.GetErrorHandler().Handle(fmt.Errorf(
					"baggage: aggregate header size %d exceeds %d byte limit",
					totalBytes,
					maxBytesPerBaggageString,
				))
			})
			return parent
		}

		// If members exceed the limit, stop parsing baggage.
		if len(members) <= maxMembers {
			currBag, err := baggage.Parse(bStr)
			if err != nil {
				parseErrors++
				if parseErrors <= maxParseErrors {
					truncateErr = errors.Join(truncateErr, err)
				}
			}
			if currBag.Len() == 0 {
				continue
			}
			members = append(members, currBag.Members()...)
		}
	}

	if dropped := parseErrors - maxParseErrors; dropped > 0 {
		truncateErr = errors.Join(truncateErr, fmt.Errorf("and %d more error(s)", dropped))
	}

	b, err := baggage.New(members...)
	if err != nil {
		truncateErr = errors.Join(truncateErr, err)
	}
	if truncateErr != nil {
		handleExtractErrOnce.Do(func() {
			errorhandler.GetErrorHandler().Handle(truncateErr)
		})
	}

	if b.Len() == 0 {
		return parent
	}
	return baggage.ContextWithBaggage(parent, b)
}
