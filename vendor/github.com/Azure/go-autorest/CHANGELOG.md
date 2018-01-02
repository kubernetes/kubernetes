# CHANGELOG

## v9.1.0

### New Features

- In cases where there is a non-empty error from the service, attempt to unmarshal it instead of uniformly calling it an "Unknown" error.
- Support for loading Azure CLI Authentication files.
- Automatically register your subscription with the Azure Resource Provider if it hadn't been previously.

### Bug Fixes

 - RetriableRequest can now tolerate a ReadSeekable body being read but not reset.
 - Adding missing Apache Headers

## v9.0.0

> **IMPORTANT:** This release was intially labeled incorrectly as `v8.4.0`. From the time it was released, it should have been marked `v9.0.0` because it contains breaking changes to the MSI packages. We appologize for any inconvenience this causes.

Adding MSI Endpoint Support and CLI token rehydration.

## v8.3.1

Pick up bug fix in adal for MSI support.

## v8.3.0

Updates to Error string formats for clarity. Also, adding a copy of the http.Response to errors for an improved debugging experience.

## v8.2.0

### New Features

- Add support for bearer authentication callbacks
- Support 429 response codes that include "Retry-After" header
- Support validation constraint "Pattern" for map keys

### Bug Fixes

- Make RetriableRequest work with multiple versions of Go

## v8.1.1
Updates the RetriableRequest to take advantage of GetBody() added in Go 1.8.

## v8.1.0
Adds RetriableRequest type for more efficient handling of retrying HTTP requests.

## v8.0.0

ADAL refactored into its own package.
Support for UNIX time.

## v7.3.1
- Version Testing now removed from production bits that are shipped with the library.

## v7.3.0
- Exposing new `RespondDecorator`, `ByDiscardingBody`. This allows operations
  to acknowledge that they do not need either the entire or a trailing portion
  of accepts response body. In doing so, Go's http library can reuse HTTP
  connections more readily.
- Adding `PrepareDecorator` to target custom BaseURLs.
- Adding ACR suffix to public cloud environment.
- Updating Glide dependencies.

## v7.2.5
- Fixed the Active Directory endpoint for the China cloud.
- Removes UTF-8 BOM if present in response payload.
- Added telemetry.

## v7.2.3
- Fixing bug in calls to `DelayForBackoff` that caused doubling of delay 
  duration.

## v7.2.2
- autorest/azure: added ASM and ARM VM DNS suffixes.

## v7.2.1
- fixed parsing of UTC times that are not RFC3339 conformant.

## v7.2.0
- autorest/validation: Reformat validation error for better error message.

## v7.1.0
- preparer: Added support for multipart formdata - WithMultiPartFormdata()
- preparer: Added support for sending file in request body - WithFile
- client: Added RetryDuration parameter.
- autorest/validation: new package for validation code for Azure Go SDK.

## v7.0.7
- Add trailing / to endpoint
- azure: add EnvironmentFromName

## v7.0.6
- Add retry logic for 408, 500, 502, 503 and 504 status codes.
- Change url path and query encoding logic.
- Fix DelayForBackoff for proper exponential delay.
- Add CookieJar in Client.

## v7.0.5
- Add check to start polling only when status is in [200,201,202].
- Refactoring for unchecked errors.
- azure/persist changes.
- Fix 'file in use' issue in renewing token in deviceflow.
- Store header RetryAfter for subsequent requests in polling.
- Add attribute details in service error.

## v7.0.4
- Better error messages for long running operation failures

## v7.0.3
- Corrected DoPollForAsynchronous to properly handle the initial response

## v7.0.2
- Corrected DoPollForAsynchronous to continue using the polling method first discovered

## v7.0.1
- Fixed empty JSON input error in ByUnmarshallingJSON
- Fixed polling support for GET calls
- Changed format name from TimeRfc1123 to TimeRFC1123

## v7.0.0
- Added ByCopying responder with supporting TeeReadCloser
- Rewrote Azure asynchronous handling
- Reverted to only unmarshalling JSON
- Corrected handling of RFC3339 time strings and added support for Rfc1123 time format

The `json.Decoder` does not catch bad data as thoroughly as `json.Unmarshal`. Since
`encoding/json` successfully deserializes all core types, and extended types normally provide
their custom JSON serialization handlers, the code has been reverted back to using
`json.Unmarshal`. The original change to use `json.Decode` was made to reduce duplicate
code; there is no loss of function, and there is a gain in accuracy, by reverting.

Additionally, Azure services indicate requests to be polled by multiple means. The existing code
only checked for one of those (that is, the presence of the `Azure-AsyncOperation` header).
The new code correctly covers all cases and aligns with the other Azure SDKs.

## v6.1.0
- Introduced `date.ByUnmarshallingJSONDate` and `date.ByUnmarshallingJSONTime` to enable JSON encoded values.

## v6.0.0
- Completely reworked the handling of polled and asynchronous requests
- Removed unnecessary routines
- Reworked `mocks.Sender` to replay a series of `http.Response` objects
- Added `PrepareDecorators` for primitive types (e.g., bool, int32)

Handling polled and asynchronous requests is no longer part of `Client#Send`. Instead new
`SendDecorators` implement different styles of polled behavior. See`autorest.DoPollForStatusCodes`
and `azure.DoPollForAsynchronous` for examples.

## v5.0.0
- Added new RespondDecorators unmarshalling primitive types
- Corrected application of inspection and authorization PrependDecorators

## v4.0.0
- Added support for Azure long-running operations.
- Added cancelation support to all decorators and functions that may delay.
- Breaking: `DelayForBackoff` now accepts a channel, which may be nil.

## v3.1.0
- Add support for OAuth Device Flow authorization.
- Add support for ServicePrincipalTokens that are backed by an existing token, rather than other secret material.
- Add helpers for persisting and restoring Tokens.
- Increased code coverage in the github.com/Azure/autorest/azure package

## v3.0.0
- Breaking: `NewErrorWithError` no longer takes `statusCode int`.
- Breaking: `NewErrorWithStatusCode` is replaced with `NewErrorWithResponse`.
- Breaking: `Client#Send()` no longer takes `codes ...int` argument.
- Add: XML unmarshaling support with `ByUnmarshallingXML()`
- Stopped vending dependencies locally and switched to [Glide](https://github.com/Masterminds/glide).
  Applications using this library should either use Glide or vendor dependencies locally some other way.
- Add: `azure.WithErrorUnlessStatusCode()` decorator to handle Azure errors.
- Fix: use `net/http.DefaultClient` as base client.
- Fix: Missing inspection for polling responses added.
- Add: CopyAndDecode helpers.
- Improved `./autorest/to` with `[]string` helpers.
- Removed golint suppressions in .travis.yml.

## v2.1.0

- Added `StatusCode` to `Error` for more easily obtaining the HTTP Reponse StatusCode (if any)

## v2.0.0

- Changed `to.StringMapPtr` method signature to return a pointer
- Changed `ServicePrincipalCertificateSecret` and `NewServicePrincipalTokenFromCertificate` to support generic certificate and private keys

## v1.0.0

- Added Logging inspectors to trace http.Request / Response
- Added support for User-Agent header
- Changed WithHeader PrepareDecorator to use set vs. add
- Added JSON to error when unmarshalling fails
- Added Client#Send method
- Corrected case of "Azure" in package paths
- Added "to" helpers, Azure helpers, and improved ease-of-use
- Corrected golint issues

## v1.0.1

- Added CHANGELOG.md

## v1.1.0

- Added mechanism to retrieve a ServicePrincipalToken using a certificate-signed JWT
- Added an example of creating a certificate-based ServicePrincipal and retrieving an OAuth token using the certificate

## v1.1.1

- Introduce godeps and vendor dependencies introduced in v1.1.1
