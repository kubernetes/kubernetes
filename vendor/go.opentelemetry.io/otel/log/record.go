// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package log

import (
	"slices"
	"time"

	"go.opentelemetry.io/otel/attribute"
)

// attributesInlineCount is the number of attributes that are efficiently
// stored in an array within a Record. This value is borrowed from slog, which
// performed a quantitative survey of log library use and found that this value
// covers 95% of all use cases (https://go.dev/blog/slog#performance).
const attributesInlineCount = 5

// Record represents a log record.
// A log record with a non-empty event name is interpreted as an event record.
type Record struct {
	// Ensure forward compatibility by explicitly making this not comparable.
	noCmp [0]func() //nolint: unused  // This is indeed used.

	eventName         string
	timestamp         time.Time
	observedTimestamp time.Time
	severity          Severity
	severityText      string
	body              attribute.Value
	err               error

	// The fields below are for optimizing the implementation of Attributes and
	// AddAttributes. This design is borrowed from the slog Record type:
	// https://cs.opensource.google/go/go/+/refs/tags/go1.22.0:src/log/slog/record.go;l=20

	// Allocation optimization: an inline array sized to hold
	// the majority of log calls (based on examination of open-source
	// code). It holds the start of the list of attributes.
	front [attributesInlineCount]attribute.KeyValue

	// The number of attributes in front.
	nFront int

	// The list of attributes except for those in front.
	// Invariants:
	//   - len(back) > 0 if nFront == len(front)
	//   - Unused array elements are zeroed to detect mistakes.
	back []attribute.KeyValue
}

// EventName returns the event name.
// A log record with a non-empty event name is interpreted as an event record.
func (r *Record) EventName() string {
	return r.eventName
}

// SetEventName sets the event name.
// A log record with a non-empty event name is interpreted as an event record.
// Event names should uniquely identify the structure of the event's attributes
// and body.
func (r *Record) SetEventName(s string) {
	r.eventName = s
}

// Timestamp returns the time when the log record occurred.
func (r *Record) Timestamp() time.Time {
	return r.timestamp
}

// SetTimestamp sets the time when the log record occurred.
func (r *Record) SetTimestamp(t time.Time) {
	r.timestamp = t
}

// ObservedTimestamp returns the time when the log record was observed.
func (r *Record) ObservedTimestamp() time.Time {
	return r.observedTimestamp
}

// SetObservedTimestamp sets the time when the log record was observed.
func (r *Record) SetObservedTimestamp(t time.Time) {
	r.observedTimestamp = t
}

// Severity returns the [Severity] of the log record.
func (r *Record) Severity() Severity {
	return r.severity
}

// SetSeverity sets the [Severity] level of the log record.
func (r *Record) SetSeverity(level Severity) {
	r.severity = level
}

// SeverityText returns the text of the severity (also known as the log level)
// as originally known at the source.
func (r *Record) SeverityText() string {
	return r.severityText
}

// SetSeverityText sets the text of the severity (also known as the log level)
// as originally known at the source.
func (r *Record) SetSeverityText(text string) {
	r.severityText = text
}

// Body returns the body of the log record.
func (r *Record) Body() attribute.Value {
	return r.body
}

// SetBody sets the body of the log record.
func (r *Record) SetBody(v attribute.Value) {
	r.body = v
}

// Err returns the associated error if one has been set.
func (r *Record) Err() error {
	return r.err
}

// SetErr sets the associated error. Passing nil clears the error.
func (r *Record) SetErr(err error) {
	r.err = err
}

// WalkAttributes walks all attributes in the log record by calling f for each
// [attribute.KeyValue] in the [Record]. Iteration stops if f returns false.
func (r *Record) WalkAttributes(f func(attribute.KeyValue) bool) {
	for i := 0; i < r.nFront; i++ {
		if !f(r.front[i]) {
			return
		}
	}
	for _, a := range r.back {
		if !f(a) {
			return
		}
	}
}

// AddAttributes adds attributes to the log record.
func (r *Record) AddAttributes(attrs ...attribute.KeyValue) {
	var i int
	for i = 0; i < len(attrs) && r.nFront < len(r.front); i++ {
		a := attrs[i]
		r.front[r.nFront] = a
		r.nFront++
	}

	r.back = slices.Grow(r.back, len(attrs[i:]))
	r.back = append(r.back, attrs[i:]...)
}

// AttributesLen returns the number of attributes in the log record.
func (r *Record) AttributesLen() int {
	return r.nFront + len(r.back)
}

// Clone returns a copy of the record with no shared state.
// The original record and the clone can both be modified without interfering
// with each other.
func (r *Record) Clone() Record {
	res := *r
	res.back = slices.Clone(r.back)
	return res
}
