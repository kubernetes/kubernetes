# Release History

## 1.0.0 (2022-05-12)

### Features Added
* Added interface `runtime.PollingHandler` to support custom poller implementations.
  * Added field `PollingHandler` of this type to `runtime.NewPollerOptions[T]` and `runtime.NewPollerFromResumeTokenOptions[T]`.

### Breaking Changes
* Renamed `cloud.Configuration.LoginEndpoint` to `.ActiveDirectoryAuthorityHost`
* Renamed `cloud.AzurePublicCloud` to `cloud.AzurePublic`
* Removed `AuxiliaryTenants` field from `arm/ClientOptions` and `arm/policy/BearerTokenOptions`
* Removed `TokenRequestOptions.TenantID`
* `Poller[T].PollUntilDone()` now takes an `options *PollUntilDoneOptions` param instead of `freq time.Duration`
* Removed `arm/runtime.Poller[T]`, `arm/runtime.NewPoller[T]()` and `arm/runtime.NewPollerFromResumeToken[T]()`
* Removed `arm/runtime.FinalStateVia` and related `const` values
* Renamed `runtime.PageProcessor` to `runtime.PagingHandler`
* The `arm/runtime.ProviderRepsonse` and `arm/runtime.Provider` types are no longer exported.
* Renamed `NewRequestIdPolicy()` to `NewRequestIDPolicy()`
* `TokenCredential.GetToken` now returns `AccessToken` by value.

### Bugs Fixed
* When per-try timeouts are enabled, only cancel the context after the body has been read and closed.
* The `Operation-Location` poller now properly handles `final-state-via` values.
* Improvements in `runtime.Poller[T]`
  * `Poll()` shouldn't cache errors, allowing for additional retries when in a non-terminal state.
  * `Result()` will cache the terminal result or error but not transient errors, allowing for additional retries.

### Other Changes
* Updated to latest `internal` module and absorbed breaking changes.
  * Use `temporal.Resource` and deleted copy.
* The internal poller implementation has been refactored.
  * The implementation in `internal/pollers/poller.go` has been merged into `runtime/poller.go` with some slight modification.
  * The internal poller types had their methods updated to conform to the `runtime.PollingHandler` interface.
  * The creation of resume tokens has been refactored so that implementers of `runtime.PollingHandler` don't need to know about it.
* `NewPipeline()` places policies from `ClientOptions` after policies from `PipelineOptions`
* Default User-Agent headers no longer include `azcore` version information

## 0.23.1 (2022-04-14)

### Bugs Fixed
* Include XML header when marshalling XML content.
* Handle XML namespaces when searching for error code.
* Handle `odata.error` when searching for error code.

## 0.23.0 (2022-04-04)

### Features Added
* Added `runtime.Pager[T any]` and `runtime.Poller[T any]` supporting types for central, generic, implementations.
* Added `cloud` package with a new API for cloud configuration
* Added `FinalStateVia` field to `runtime.NewPollerOptions[T any]` type.

### Breaking Changes
* Removed the `Poller` type-alias to the internal poller implementation.
* Added `Ptr[T any]` and `SliceOfPtrs[T any]` in the `to` package and removed all non-generic implementations.
* `NullValue` and `IsNullValue` now take a generic type parameter instead of an interface func parameter.
* Replaced `arm.Endpoint` with `cloud` API
  * Removed the `endpoint` parameter from `NewRPRegistrationPolicy()`
  * `arm/runtime.NewPipeline()` and `.NewRPRegistrationPolicy()` now return an `error`
* Refactored `NewPoller` and `NewPollerFromResumeToken` funcs in `arm/runtime` and `runtime` packages.
  * Removed the `pollerID` parameter as it's no longer required.
  * Created optional parameter structs and moved optional parameters into them.
* Changed `FinalStateVia` field to a `const` type.

### Other Changes
* Converted expiring resource and dependent types to use generics.

## 0.22.0 (2022-03-03)

### Features Added
* Added header `WWW-Authenticate` to the default allow-list of headers for logging.
* Added a pipeline policy that enables the retrieval of HTTP responses from API calls.
  * Added `runtime.WithCaptureResponse` to enable the policy at the API level (off by default).

### Breaking Changes
* Moved `WithHTTPHeader` and `WithRetryOptions` from the `policy` package to the `runtime` package.

## 0.21.1 (2022-02-04)

### Bugs Fixed
* Restore response body after reading in `Poller.FinalResponse()`. (#16911)
* Fixed bug in `NullValue` that could lead to incorrect comparisons for empty maps/slices (#16969)

### Other Changes
* `BearerTokenPolicy` is more resilient to transient authentication failures. (#16789)

## 0.21.0 (2022-01-11)

### Features Added
* Added `AllowedHeaders` and `AllowedQueryParams` to `policy.LogOptions` to control which headers and query parameters are written to the logger.
* Added `azcore.ResponseError` type which is returned from APIs when a non-success HTTP status code is received.

### Breaking Changes
* Moved `[]policy.Policy` parameters of `arm/runtime.NewPipeline` and `runtime.NewPipeline` into a new struct, `runtime.PipelineOptions`
* Renamed `arm/ClientOptions.Host` to `.Endpoint`
* Moved `Request.SkipBodyDownload` method to function `runtime.SkipBodyDownload`
* Removed `azcore.HTTPResponse` interface type
* `arm.NewPoller()` and `runtime.NewPoller()` no longer require an `eu` parameter
* `runtime.NewResponseError()` no longer requires an `error` parameter

## 0.20.0 (2021-10-22)

### Breaking Changes
* Removed `arm.Connection`
* Removed `azcore.Credential` and `.NewAnonymousCredential()`
  * `NewRPRegistrationPolicy` now requires an `azcore.TokenCredential`
* `runtime.NewPipeline` has a new signature that simplifies implementing custom authentication
* `arm/runtime.RegistrationOptions` embeds `policy.ClientOptions`
* Contents in the `log` package have been slightly renamed.
* Removed `AuthenticationOptions` in favor of `policy.BearerTokenOptions`
* Changed parameters for `NewBearerTokenPolicy()`
* Moved policy config options out of `arm/runtime` and into `arm/policy`

### Features Added
* Updating Documentation
* Added string typdef `arm.Endpoint` to provide a hint toward expected ARM client endpoints
* `azcore.ClientOptions` contains common pipeline configuration settings
* Added support for multi-tenant authorization in `arm/runtime`
* Require one second minimum when calling `PollUntilDone()`

### Bug Fixes
* Fixed a potential panic when creating the default Transporter.
* Close LRO initial response body when creating a poller.
* Fixed a panic when recursively cloning structs that contain time.Time.

## 0.19.0 (2021-08-25)

### Breaking Changes
* Split content out of `azcore` into various packages.  The intent is to separate content based on its usage (common, uncommon, SDK authors).
  * `azcore` has all core functionality.
  * `log` contains facilities for configuring in-box logging.
  * `policy` is used for configuring pipeline options and creating custom pipeline policies.
  * `runtime` contains various helpers used by SDK authors and generated content.
  * `streaming` has helpers for streaming IO operations.
* `NewTelemetryPolicy()` now requires module and version parameters and the `Value` option has been removed.
  * As a result, the `Request.Telemetry()` method has been removed.
* The telemetry policy now includes the SDK prefix `azsdk-go-` so callers no longer need to provide it.
* The `*http.Request` in `runtime.Request` is no longer anonymously embedded.  Use the `Raw()` method to access it.
* The `UserAgent` and `Version` constants have been made internal, `Module` and `Version` respectively.

### Bug Fixes
* Fixed an issue in the retry policy where the request body could be overwritten after a rewind.

### Other Changes
* Moved modules `armcore` and `to` content into `arm` and `to` packages respectively.
  * The `Pipeline()` method on `armcore.Connection` has been replaced by `NewPipeline()` in `arm.Connection`.  It takes module and version parameters used by the telemetry policy.
* Poller logic has been consolidated across ARM and core implementations.
  * This required some changes to the internal interfaces for core pollers.
* The core poller types have been improved, including more logging and test coverage.

## 0.18.1 (2021-08-20)

### Features Added
* Adds an `ETag` type for comparing etags and handling etags on requests
* Simplifies the `requestBodyProgess` and `responseBodyProgress` into a single `progress` object

### Bugs Fixed
* `JoinPaths` will preserve query parameters encoded in the `root` url.

### Other Changes
* Bumps dependency on `internal` module to the latest version (v0.7.0)

## 0.18.0 (2021-07-29)
### Features Added
* Replaces methods from Logger type with two package methods for interacting with the logging functionality.
* `azcore.SetClassifications` replaces `azcore.Logger().SetClassifications`
* `azcore.SetListener` replaces `azcore.Logger().SetListener`

### Breaking Changes
* Removes `Logger` type from `azcore`


## 0.17.0 (2021-07-27)
### Features Added
* Adding TenantID to TokenRequestOptions (https://github.com/Azure/azure-sdk-for-go/pull/14879)
* Adding AuxiliaryTenants to AuthenticationOptions (https://github.com/Azure/azure-sdk-for-go/pull/15123)

### Breaking Changes
* Rename `AnonymousCredential` to `NewAnonymousCredential` (https://github.com/Azure/azure-sdk-for-go/pull/15104)
* rename `AuthenticationPolicyOptions` to `AuthenticationOptions` (https://github.com/Azure/azure-sdk-for-go/pull/15103)
* Make Header constants private (https://github.com/Azure/azure-sdk-for-go/pull/15038)


## 0.16.2 (2021-05-26)
### Features Added
* Improved support for byte arrays [#14715](https://github.com/Azure/azure-sdk-for-go/pull/14715)


## 0.16.1 (2021-05-19)
### Features Added
* Add license.txt to azcore module [#14682](https://github.com/Azure/azure-sdk-for-go/pull/14682)


## 0.16.0 (2021-05-07)
### Features Added
* Remove extra `*` in UnmarshalAsByteArray() [#14642](https://github.com/Azure/azure-sdk-for-go/pull/14642)


## 0.15.1 (2021-05-06)
### Features Added
* Cache the original request body on Request [#14634](https://github.com/Azure/azure-sdk-for-go/pull/14634)


## 0.15.0 (2021-05-05)
### Features Added
* Add support for null map and slice
* Export `Response.Payload` method

### Breaking Changes
* remove `Response.UnmarshalError` as it's no longer required


## 0.14.5 (2021-04-23)
### Features Added
* Add `UnmarshalError()` on `azcore.Response`


## 0.14.4 (2021-04-22)
### Features Added
* Support for basic LRO polling
* Added type `LROPoller` and supporting types for basic polling on long running operations.
* rename poller param and added doc comment

### Bugs Fixed
* Fixed content type detection bug in logging.


## 0.14.3 (2021-03-29)
### Features Added
* Add support for multi-part form data
* Added method `WriteMultipartFormData()` to Request.


## 0.14.2 (2021-03-17)
### Features Added
* Add support for encoding JSON null values
* Adds `NullValue()` and `IsNullValue()` functions for setting and detecting sentinel values used for encoding a JSON null.
* Documentation fixes

### Bugs Fixed
* Fixed improper error wrapping


## 0.14.1 (2021-02-08)
### Features Added
* Add `Pager` and `Poller` interfaces to azcore


## 0.14.0 (2021-01-12)
### Features Added
* Accept zero-value options for default values
* Specify zero-value options structs to accept default values.
* Remove `DefaultXxxOptions()` methods.
* Do not silently change TryTimeout on negative values
* make per-try timeout opt-in


## 0.13.4 (2020-11-20)
### Features Added
* Include telemetry string in User Agent


## 0.13.3 (2020-11-20)
### Features Added
* Updating response body handling on `azcore.Response`


## 0.13.2 (2020-11-13)
### Features Added
* Remove implementation of stateless policies as first-class functions.


## 0.13.1 (2020-11-05)
### Features Added
* Add `Telemetry()` method to `azcore.Request()`


## 0.13.0 (2020-10-14)
### Features Added
* Rename `log` to `logger` to avoid name collision with the log package.
* Documentation improvements
* Simplified `DefaultHTTPClientTransport()` implementation


## 0.12.1 (2020-10-13)
### Features Added
* Update `internal` module dependence to `v0.5.0`


## 0.12.0 (2020-10-08)
### Features Added
* Removed storage specific content
* Removed internal content to prevent API clutter
* Refactored various policy options to conform with our options pattern


## 0.11.0 (2020-09-22)
### Features Added

* Removed `LogError` and `LogSlowResponse`.
* Renamed `options` in `RequestLogOptions`.
* Updated `NewRequestLogPolicy()` to follow standard pattern for options.
* Refactored `requestLogPolicy.Do()` per above changes.
* Cleaned up/added logging in retry policy.
* Export `NewResponseError()`
* Fix `RequestLogOptions` comment


## 0.10.1 (2020-09-17)
### Features Added
* Add default console logger
* Default console logger writes to stderr. To enable it, set env var `AZURE_SDK_GO_LOGGING` to the value 'all'.
* Added `Logger.Writef()` to reduce the need for `ShouldLog()` checks.
* Add `LogLongRunningOperation`


## 0.10.0 (2020-09-10)
### Features Added
* The `request` and `transport` interfaces have been refactored to align with the patterns in the standard library.
* `NewRequest()` now uses `http.NewRequestWithContext()` and performs additional validation, it also requires a context parameter.
* The `Policy` and `Transport` interfaces have had their context parameter removed as the context is associated with the underlying `http.Request`.
* `Pipeline.Do()` will validate the HTTP request before sending it through the pipeline, avoiding retries on a malformed request.
* The `Retrier` interface has been replaced with the `NonRetriableError` interface, and the retry policy updated to test for this.
* `Request.SetBody()` now requires a content type parameter for setting the request's MIME type.
* moved path concatenation into `JoinPaths()` func


## 0.9.6 (2020-08-18)
### Features Added
* Improvements to body download policy
* Always download the response body for error responses, i.e. HTTP status codes >= 400.
* Simplify variable declarations


## 0.9.5 (2020-08-11)
### Features Added
* Set the Content-Length header in `Request.SetBody`


## 0.9.4 (2020-08-03)
### Features Added
* Fix cancellation of per try timeout
* Per try timeout is used to ensure that an HTTP operation doesn't take too long, e.g. that a GET on some URL doesn't take an inordinant amount of time.
* Once the HTTP request returns, the per try timeout should be cancelled, not when the response has been read to completion.
* Do not drain response body if there are no more retries
* Do not retry non-idempotent operations when body download fails


## 0.9.3 (2020-07-28)
### Features Added
* Add support for custom HTTP request headers
* Inserts an internal policy into the pipeline that can extract HTTP header values from the caller's context, adding them to the request.
* Use `azcore.WithHTTPHeader` to add HTTP headers to a context.
* Remove method specific to Go 1.14


## 0.9.2 (2020-07-28)
### Features Added
* Omit read-only content from request payloads
* If any field in a payload's object graph contains `azure:"ro"`, make a clone of the object graph, omitting all fields with this annotation.
* Verify no fields were dropped
* Handle embedded struct types
* Added test for cloning by value
* Add messages to failures


## 0.9.1 (2020-07-22)
### Features Added
* Updated dependency on internal module to fix race condition.


## 0.9.0 (2020-07-09)
### Features Added
* Add `HTTPResponse` interface to be used by callers to access the raw HTTP response from an error in the event of an API call failure.
* Updated `sdk/internal` dependency to latest version.
* Rename package alias


## 0.8.2 (2020-06-29)
### Features Added
* Added missing documentation comments

### Bugs Fixed
* Fixed a bug in body download policy.


## 0.8.1 (2020-06-26)
### Features Added
* Miscellaneous clean-up reported by linters


## 0.8.0 (2020-06-01)
### Features Added
* Differentiate between standard and URL encoding.


## 0.7.1 (2020-05-27)
### Features Added
* Add support for for base64 encoding and decoding of payloads.


## 0.7.0 (2020-05-12)
### Features Added
* Change `RetryAfter()` to a function.


## 0.6.0 (2020-04-29)
### Features Added
* Updating `RetryAfter` to only return the detaion in the RetryAfter header


## 0.5.0 (2020-03-23)
### Features Added
* Export `TransportFunc`

### Breaking Changes
* Removed `IterationDone`


## 0.4.1 (2020-02-25)
### Features Added
* Ensure per-try timeout is properly cancelled
* Explicitly call cancel the per-try timeout when the response body has been read/closed by the body download policy.
* When the response body is returned to the caller for reading/closing, wrap it in a `responseBodyReader` that will cancel the timeout when the body is closed.
* `Logger.Should()` will return false if no listener is set.


## 0.4.0 (2020-02-18)
### Features Added
* Enable custom `RetryOptions` to be specified per API call
* Added `WithRetryOptions()` that adds a custom `RetryOptions` to the provided context, allowing custom settings per API call.
* Remove 429 from the list of default HTTP status codes for retry.
* Change StatusCodesForRetry to a slice so consumers can append to it.
* Added support for retry-after in HTTP-date format.
* Cleaned up some comments specific to storage.
* Remove `Request.SetQueryParam()`
* Renamed `MaxTries` to `MaxRetries`

## 0.3.0 (2020-01-16)
### Features Added
* Added `DefaultRetryOptions` to create initialized default options.

### Breaking Changes
* Removed `Response.CheckStatusCode()`


## 0.2.0 (2020-01-15)
### Features Added
* Add support for marshalling and unmarshalling JSON
* Removed `Response.Payload` field
* Exit early when unmarsahlling if there is no payload


## 0.1.0 (2020-01-10)
### Features Added
* Initial release
