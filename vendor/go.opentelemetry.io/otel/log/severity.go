// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:generate stringer -type=Severity -linecomment

package log

// Severity represents a log record's severity (also known as its log level).
// Smaller numerical values correspond to less severe log records (such as
// debug events); larger numerical values correspond to more severe log records
// (such as errors and critical events).
type Severity int

// The following Severity values are defined by OpenTelemetry.
const (
	// SeverityUndefined represents an unset Severity.
	SeverityUndefined Severity = 0 // UNDEFINED

	// The following are severity values for fine-grained debugging log records.
	// They are typically disabled in default configurations.
	SeverityTrace1 Severity = 1 // TRACE
	SeverityTrace2 Severity = 2 // TRACE2
	SeverityTrace3 Severity = 3 // TRACE3
	SeverityTrace4 Severity = 4 // TRACE4

	// The following are severity values for debugging log records.
	SeverityDebug1 Severity = 5 // DEBUG
	SeverityDebug2 Severity = 6 // DEBUG2
	SeverityDebug3 Severity = 7 // DEBUG3
	SeverityDebug4 Severity = 8 // DEBUG4

	// The following are severity values for informational log records indicating
	// that an event happened.
	SeverityInfo1 Severity = 9  // INFO
	SeverityInfo2 Severity = 10 // INFO2
	SeverityInfo3 Severity = 11 // INFO3
	SeverityInfo4 Severity = 12 // INFO4

	// The following are severity values for warning log records. These records
	// are not errors, but they are likely more important than informational
	// events.
	SeverityWarn1 Severity = 13 // WARN
	SeverityWarn2 Severity = 14 // WARN2
	SeverityWarn3 Severity = 15 // WARN3
	SeverityWarn4 Severity = 16 // WARN4

	// The following are severity values for error log records indicating that
	// something went wrong.
	SeverityError1 Severity = 17 // ERROR
	SeverityError2 Severity = 18 // ERROR2
	SeverityError3 Severity = 19 // ERROR3
	SeverityError4 Severity = 20 // ERROR4

	// The following are severity values for fatal log records, such as those
	// associated with an application or system crash.
	SeverityFatal1 Severity = 21 // FATAL
	SeverityFatal2 Severity = 22 // FATAL2
	SeverityFatal3 Severity = 23 // FATAL3
	SeverityFatal4 Severity = 24 // FATAL4

	// The following are convenience definitions for the base severity of each
	// level.
	SeverityTrace = SeverityTrace1
	SeverityDebug = SeverityDebug1
	SeverityInfo  = SeverityInfo1
	SeverityWarn  = SeverityWarn1
	SeverityError = SeverityError1
	SeverityFatal = SeverityFatal1
)
