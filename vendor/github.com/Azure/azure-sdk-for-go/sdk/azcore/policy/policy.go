//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package policy

import (
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/cloud"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
)

// Policy represents an extensibility point for the Pipeline that can mutate the specified
// Request and react to the received Response.
type Policy = exported.Policy

// Transporter represents an HTTP pipeline transport used to send HTTP requests and receive responses.
type Transporter = exported.Transporter

// Request is an abstraction over the creation of an HTTP request as it passes through the pipeline.
// Don't use this type directly, use runtime.NewRequest() instead.
type Request = exported.Request

// ClientOptions contains optional settings for a client's pipeline.
// All zero-value fields will be initialized with default values.
type ClientOptions struct {
	// Cloud specifies a cloud for the client. The default is Azure Public Cloud.
	Cloud cloud.Configuration

	// Logging configures the built-in logging policy.
	Logging LogOptions

	// Retry configures the built-in retry policy.
	Retry RetryOptions

	// Telemetry configures the built-in telemetry policy.
	Telemetry TelemetryOptions

	// Transport sets the transport for HTTP requests.
	Transport Transporter

	// PerCallPolicies contains custom policies to inject into the pipeline.
	// Each policy is executed once per request.
	PerCallPolicies []Policy

	// PerRetryPolicies contains custom policies to inject into the pipeline.
	// Each policy is executed once per request, and for each retry of that request.
	PerRetryPolicies []Policy
}

// LogOptions configures the logging policy's behavior.
type LogOptions struct {
	// IncludeBody indicates if request and response bodies should be included in logging.
	// The default value is false.
	// NOTE: enabling this can lead to disclosure of sensitive information, use with care.
	IncludeBody bool

	// AllowedHeaders is the slice of headers to log with their values intact.
	// All headers not in the slice will have their values REDACTED.
	// Applies to request and response headers.
	AllowedHeaders []string

	// AllowedQueryParams is the slice of query parameters to log with their values intact.
	// All query parameters not in the slice will have their values REDACTED.
	AllowedQueryParams []string
}

// RetryOptions configures the retry policy's behavior.
// Call NewRetryOptions() to create an instance with default values.
type RetryOptions struct {
	// MaxRetries specifies the maximum number of attempts a failed operation will be retried
	// before producing an error.
	// The default value is three.  A value less than zero means one try and no retries.
	MaxRetries int32

	// TryTimeout indicates the maximum time allowed for any single try of an HTTP request.
	// This is disabled by default.  Specify a value greater than zero to enable.
	// NOTE: Setting this to a small value might cause premature HTTP request time-outs.
	TryTimeout time.Duration

	// RetryDelay specifies the initial amount of delay to use before retrying an operation.
	// The delay increases exponentially with each retry up to the maximum specified by MaxRetryDelay.
	// The default value is four seconds.  A value less than zero means no delay between retries.
	RetryDelay time.Duration

	// MaxRetryDelay specifies the maximum delay allowed before retrying an operation.
	// Typically the value is greater than or equal to the value specified in RetryDelay.
	// The default Value is 120 seconds.  A value less than zero means there is no cap.
	MaxRetryDelay time.Duration

	// StatusCodes specifies the HTTP status codes that indicate the operation should be retried.
	// The default value is the status codes in StatusCodesForRetry.
	// Specifying an empty slice will cause retries to happen only for transport errors.
	StatusCodes []int
}

// TelemetryOptions configures the telemetry policy's behavior.
type TelemetryOptions struct {
	// ApplicationID is an application-specific identification string to add to the User-Agent.
	// It has a maximum length of 24 characters and must not contain any spaces.
	ApplicationID string

	// Disabled will prevent the addition of any telemetry data to the User-Agent.
	Disabled bool
}

// TokenRequestOptions contain specific parameter that may be used by credentials types when attempting to get a token.
type TokenRequestOptions struct {
	// Scopes contains the list of permission scopes required for the token.
	Scopes []string
}

// BearerTokenOptions configures the bearer token policy's behavior.
type BearerTokenOptions struct {
	// placeholder for future options
}
