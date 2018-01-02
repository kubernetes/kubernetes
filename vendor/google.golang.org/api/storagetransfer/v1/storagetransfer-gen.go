// Package storagetransfer provides access to the Google Storage Transfer API.
//
// See https://cloud.google.com/storage/transfer
//
// Usage example:
//
//   import "google.golang.org/api/storagetransfer/v1"
//   ...
//   storagetransferService, err := storagetransfer.New(oauthHttpClient)
package storagetransfer // import "google.golang.org/api/storagetransfer/v1"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "storagetransfer:v1"
const apiName = "storagetransfer"
const apiVersion = "v1"
const basePath = "https://storagetransfer.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.GoogleServiceAccounts = NewGoogleServiceAccountsService(s)
	s.TransferJobs = NewTransferJobsService(s)
	s.TransferOperations = NewTransferOperationsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	GoogleServiceAccounts *GoogleServiceAccountsService

	TransferJobs *TransferJobsService

	TransferOperations *TransferOperationsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewGoogleServiceAccountsService(s *Service) *GoogleServiceAccountsService {
	rs := &GoogleServiceAccountsService{s: s}
	return rs
}

type GoogleServiceAccountsService struct {
	s *Service
}

func NewTransferJobsService(s *Service) *TransferJobsService {
	rs := &TransferJobsService{s: s}
	return rs
}

type TransferJobsService struct {
	s *Service
}

func NewTransferOperationsService(s *Service) *TransferOperationsService {
	rs := &TransferOperationsService{s: s}
	return rs
}

type TransferOperationsService struct {
	s *Service
}

// AwsAccessKey: AWS access key (see
// [AWS Security
// Credentials](http://docs.aws.amazon.com/general/latest/gr/aws-security
// -credentials.html)).
type AwsAccessKey struct {
	// AccessKeyId: AWS access key ID.
	// Required.
	AccessKeyId string `json:"accessKeyId,omitempty"`

	// SecretAccessKey: AWS secret access key. This field is not returned in
	// RPC responses.
	// Required.
	SecretAccessKey string `json:"secretAccessKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccessKeyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccessKeyId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AwsAccessKey) MarshalJSON() ([]byte, error) {
	type noMethod AwsAccessKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AwsS3Data: An AwsS3Data can be a data source, but not a data sink.
// In an AwsS3Data, an object's name is the S3 object's key name.
type AwsS3Data struct {
	// AwsAccessKey: AWS access key used to sign the API requests to the AWS
	// S3 bucket.
	// Permissions on the bucket must be granted to the access ID of the
	// AWS access key.
	// Required.
	AwsAccessKey *AwsAccessKey `json:"awsAccessKey,omitempty"`

	// BucketName: S3 Bucket name (see
	// [Creating a
	// bucket](http://docs.aws.amazon.com/AmazonS3/latest/dev/create-bucket-g
	// et-location-example.html)).
	// Required.
	BucketName string `json:"bucketName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AwsAccessKey") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AwsAccessKey") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AwsS3Data) MarshalJSON() ([]byte, error) {
	type noMethod AwsS3Data
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Date: Represents a whole calendar date, e.g. date of birth. The time
// of day and
// time zone are either specified elsewhere or are not significant. The
// date
// is relative to the Proleptic Gregorian Calendar. The day may be 0
// to
// represent a year and month where the day is not significant, e.g.
// credit card
// expiration date. The year may be 0 to represent a month and day
// independent
// of year, e.g. anniversary date. Related types are
// google.type.TimeOfDay
// and `google.protobuf.Timestamp`.
type Date struct {
	// Day: Day of month. Must be from 1 to 31 and valid for the year and
	// month, or 0
	// if specifying a year/month where the day is not significant.
	Day int64 `json:"day,omitempty"`

	// Month: Month of year. Must be from 1 to 12.
	Month int64 `json:"month,omitempty"`

	// Year: Year of date. Must be from 1 to 9999, or 0 if specifying a date
	// without
	// a year.
	Year int64 `json:"year,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Day") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Date) MarshalJSON() ([]byte, error) {
	type noMethod Date
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated
// empty messages in your APIs. A typical example is to use it as the
// request
// or the response type of an API method. For instance:
//
//     service Foo {
//       rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
//     }
//
// The JSON representation for `Empty` is empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// ErrorLogEntry: An entry describing an error that has occurred.
type ErrorLogEntry struct {
	// ErrorDetails: A list of messages that carry the error details.
	ErrorDetails []string `json:"errorDetails,omitempty"`

	// Url: A URL that refers to the target (a data source, a data sink,
	// or an object) with which the error is associated.
	// Required.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorDetails") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ErrorDetails") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ErrorLogEntry) MarshalJSON() ([]byte, error) {
	type noMethod ErrorLogEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ErrorSummary: A summary of errors by error code, plus a count and
// sample error log
// entries.
type ErrorSummary struct {
	// ErrorCode: Required.
	//
	// Possible values:
	//   "OK" - Not an error; returned on success
	//
	// HTTP Mapping: 200 OK
	//   "CANCELLED" - The operation was cancelled, typically by the
	// caller.
	//
	// HTTP Mapping: 499 Client Closed Request
	//   "UNKNOWN" - Unknown error.  For example, this error may be returned
	// when
	// a `Status` value received from another address space belongs to
	// an error space that is not known in this address space.  Also
	// errors raised by APIs that do not return enough error information
	// may be converted to this error.
	//
	// HTTP Mapping: 500 Internal Server Error
	//   "INVALID_ARGUMENT" - The client specified an invalid argument.
	// Note that this differs
	// from `FAILED_PRECONDITION`.  `INVALID_ARGUMENT` indicates
	// arguments
	// that are problematic regardless of the state of the system
	// (e.g., a malformed file name).
	//
	// HTTP Mapping: 400 Bad Request
	//   "DEADLINE_EXCEEDED" - The deadline expired before the operation
	// could complete. For operations
	// that change the state of the system, this error may be returned
	// even if the operation has completed successfully.  For example,
	// a
	// successful response from a server could have been delayed long
	// enough for the deadline to expire.
	//
	// HTTP Mapping: 504 Gateway Timeout
	//   "NOT_FOUND" - Some requested entity (e.g., file or directory) was
	// not found.
	//
	// Note to server developers: if a request is denied for an entire
	// class
	// of users, such as gradual feature rollout or undocumented
	// whitelist,
	// `NOT_FOUND` may be used. If a request is denied for some users
	// within
	// a class of users, such as user-based access control,
	// `PERMISSION_DENIED`
	// must be used.
	//
	// HTTP Mapping: 404 Not Found
	//   "ALREADY_EXISTS" - The entity that a client attempted to create
	// (e.g., file or directory)
	// already exists.
	//
	// HTTP Mapping: 409 Conflict
	//   "PERMISSION_DENIED" - The caller does not have permission to
	// execute the specified
	// operation. `PERMISSION_DENIED` must not be used for rejections
	// caused by exhausting some resource (use `RESOURCE_EXHAUSTED`
	// instead for those errors). `PERMISSION_DENIED` must not be
	// used if the caller can not be identified (use
	// `UNAUTHENTICATED`
	// instead for those errors). This error code does not imply the
	// request is valid or the requested entity exists or satisfies
	// other pre-conditions.
	//
	// HTTP Mapping: 403 Forbidden
	//   "UNAUTHENTICATED" - The request does not have valid authentication
	// credentials for the
	// operation.
	//
	// HTTP Mapping: 401 Unauthorized
	//   "RESOURCE_EXHAUSTED" - Some resource has been exhausted, perhaps a
	// per-user quota, or
	// perhaps the entire file system is out of space.
	//
	// HTTP Mapping: 429 Too Many Requests
	//   "FAILED_PRECONDITION" - The operation was rejected because the
	// system is not in a state
	// required for the operation's execution.  For example, the
	// directory
	// to be deleted is non-empty, an rmdir operation is applied to
	// a non-directory, etc.
	//
	// Service implementors can use the following guidelines to
	// decide
	// between `FAILED_PRECONDITION`, `ABORTED`, and `UNAVAILABLE`:
	//  (a) Use `UNAVAILABLE` if the client can retry just the failing
	// call.
	//  (b) Use `ABORTED` if the client should retry at a higher level
	//      (e.g., when a client-specified test-and-set fails, indicating
	// the
	//      client should restart a read-modify-write sequence).
	//  (c) Use `FAILED_PRECONDITION` if the client should not retry until
	//      the system state has been explicitly fixed.  E.g., if an
	// "rmdir"
	//      fails because the directory is non-empty, `FAILED_PRECONDITION`
	//      should be returned since the client should not retry unless
	//      the files are deleted from the directory.
	//
	// HTTP Mapping: 400 Bad Request
	//   "ABORTED" - The operation was aborted, typically due to a
	// concurrency issue such as
	// a sequencer check failure or transaction abort.
	//
	// See the guidelines above for deciding between
	// `FAILED_PRECONDITION`,
	// `ABORTED`, and `UNAVAILABLE`.
	//
	// HTTP Mapping: 409 Conflict
	//   "OUT_OF_RANGE" - The operation was attempted past the valid range.
	// E.g., seeking or
	// reading past end-of-file.
	//
	// Unlike `INVALID_ARGUMENT`, this error indicates a problem that may
	// be fixed if the system state changes. For example, a 32-bit
	// file
	// system will generate `INVALID_ARGUMENT` if asked to read at an
	// offset that is not in the range [0,2^32-1], but it will
	// generate
	// `OUT_OF_RANGE` if asked to read from an offset past the current
	// file size.
	//
	// There is a fair bit of overlap between `FAILED_PRECONDITION`
	// and
	// `OUT_OF_RANGE`.  We recommend using `OUT_OF_RANGE` (the more
	// specific
	// error) when it applies so that callers who are iterating through
	// a space can easily look for an `OUT_OF_RANGE` error to detect
	// when
	// they are done.
	//
	// HTTP Mapping: 400 Bad Request
	//   "UNIMPLEMENTED" - The operation is not implemented or is not
	// supported/enabled in this
	// service.
	//
	// HTTP Mapping: 501 Not Implemented
	//   "INTERNAL" - Internal errors.  This means that some invariants
	// expected by the
	// underlying system have been broken.  This error code is reserved
	// for serious errors.
	//
	// HTTP Mapping: 500 Internal Server Error
	//   "UNAVAILABLE" - The service is currently unavailable.  This is most
	// likely a
	// transient condition, which can be corrected by retrying with
	// a backoff.
	//
	// See the guidelines above for deciding between
	// `FAILED_PRECONDITION`,
	// `ABORTED`, and `UNAVAILABLE`.
	//
	// HTTP Mapping: 503 Service Unavailable
	//   "DATA_LOSS" - Unrecoverable data loss or corruption.
	//
	// HTTP Mapping: 500 Internal Server Error
	ErrorCode string `json:"errorCode,omitempty"`

	// ErrorCount: Count of this type of error.
	// Required.
	ErrorCount int64 `json:"errorCount,omitempty,string"`

	// ErrorLogEntries: Error samples.
	ErrorLogEntries []*ErrorLogEntry `json:"errorLogEntries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ErrorCode") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ErrorSummary) MarshalJSON() ([]byte, error) {
	type noMethod ErrorSummary
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GcsData: In a GcsData, an object's name is the Google Cloud Storage
// object's name and
// its `lastModificationTime` refers to the object's updated time, which
// changes
// when the content or the metadata of the object is updated.
type GcsData struct {
	// BucketName: Google Cloud Storage bucket name (see
	// [Bucket Name
	// Requirements](https://cloud.google.com/storage/docs/bucket-naming#requ
	// irements)).
	// Required.
	BucketName string `json:"bucketName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BucketName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GcsData) MarshalJSON() ([]byte, error) {
	type noMethod GcsData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleServiceAccount: Google service account
type GoogleServiceAccount struct {
	// AccountEmail: Required.
	AccountEmail string `json:"accountEmail,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountEmail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountEmail") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleServiceAccount) MarshalJSON() ([]byte, error) {
	type noMethod GoogleServiceAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HttpData: An HttpData specifies a list of objects on the web to be
// transferred over
// HTTP.  The information of the objects to be transferred is contained
// in a
// file referenced by a URL. The first line in the file must
// be
// "TsvHttpData-1.0", which specifies the format of the file.
// Subsequent lines
// specify the information of the list of objects, one object per list
// entry.
// Each entry has the following tab-delimited fields:
//
// * HTTP URL - The location of the object.
//
// * Length - The size of the object in bytes.
//
// * MD5 - The base64-encoded MD5 hash of the object.
//
// For an example of a valid TSV file, see
// [Transferring data from
// URLs](https://cloud.google.com/storage/transfer/create-url-list).
//
// Whe
// n transferring data based on a URL list, keep the following in
// mind:
//
// * When an object located at `http(s)://hostname:port/<URL-path>` is
// transferred
// to a data sink, the name of the object at the data sink
// is
// `<hostname>/<URL-path>`.
//
// * If the specified size of an object does not match the actual size
// of the
// object fetched, the object will not be transferred.
//
// * If the specified MD5 does not match the MD5 computed from the
// transferred
// bytes, the object transfer will fail. For more information,
// see
// [Generating MD5
// hashes](https://cloud.google.com/storage/transfer/#md5)
//
// * Ensure that each URL you specify is publicly accessible.
// For
// example, in Google Cloud Storage you can
// [share an object
// publicly]
// (https://cloud.google.com/storage/docs/cloud-console#_sharin
// gdata) and get
// a link to it.
//
// * Storage Transfer Service obeys `robots.txt` rules and requires the
// source
// HTTP server to support `Range` requests and to return a
// `Content-Length`
// header in each response.
//
// * [ObjectConditions](#ObjectConditions) have no effect when filtering
// objects
// to transfer.
type HttpData struct {
	// ListUrl: The URL that points to the file that stores the object list
	// entries.
	// This file must allow public access.  Currently, only URLs with HTTP
	// and
	// HTTPS schemes are supported.
	// Required.
	ListUrl string `json:"listUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ListUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ListUrl") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HttpData) MarshalJSON() ([]byte, error) {
	type noMethod HttpData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListOperationsResponse: The response message for
// Operations.ListOperations.
type ListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that matches the specified filter in
	// the request.
	Operations []*Operation `json:"operations,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTransferJobsResponse: Response from ListTransferJobs.
type ListTransferJobsResponse struct {
	// NextPageToken: The list next page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TransferJobs: A list of transfer jobs.
	TransferJobs []*TransferJob `json:"transferJobs,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListTransferJobsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTransferJobsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ObjectConditions: Conditions that determine which objects will be
// transferred.
type ObjectConditions struct {
	// ExcludePrefixes: `excludePrefixes` must follow the requirements
	// described for
	// `includePrefixes`.
	//
	// The max size of `excludePrefixes` is 1000.
	ExcludePrefixes []string `json:"excludePrefixes,omitempty"`

	// IncludePrefixes: If `includePrefixes` is specified, objects that
	// satisfy the object
	// conditions must have names that start with one of the
	// `includePrefixes`
	// and that do not start with any of the `excludePrefixes`. If
	// `includePrefixes`
	// is not specified, all objects except those that have names starting
	// with
	// one of the `excludePrefixes` must satisfy the object
	// conditions.
	//
	// Requirements:
	//
	//   * Each include-prefix and exclude-prefix can contain any sequence
	// of
	//     Unicode characters, of max length 1024 bytes when UTF8-encoded,
	// and
	//     must not contain Carriage Return or Line Feed characters.
	// Wildcard
	//     matching and regular expression matching are not supported.
	//
	//   * Each include-prefix and exclude-prefix must omit the leading
	// slash.
	//     For example, to include the `requests.gz` object in a transfer
	// from
	//     `s3://my-aws-bucket/logs/y=2015/requests.gz`, specify the
	// include
	//     prefix as `logs/y=2015/requests.gz`.
	//
	//   * None of the include-prefix or the exclude-prefix values can be
	// empty,
	//     if specified.
	//
	//   * Each include-prefix must include a distinct portion of the
	// object
	//     namespace, i.e., no include-prefix may be a prefix of another
	//     include-prefix.
	//
	//   * Each exclude-prefix must exclude a distinct portion of the
	// object
	//     namespace, i.e., no exclude-prefix may be a prefix of another
	//     exclude-prefix.
	//
	//   * If `includePrefixes` is specified, then each exclude-prefix must
	// start
	//     with the value of a path explicitly included by
	// `includePrefixes`.
	//
	// The max size of `includePrefixes` is 1000.
	IncludePrefixes []string `json:"includePrefixes,omitempty"`

	// MaxTimeElapsedSinceLastModification:
	// `maxTimeElapsedSinceLastModification` is the complement
	// to
	// `minTimeElapsedSinceLastModification`.
	MaxTimeElapsedSinceLastModification string `json:"maxTimeElapsedSinceLastModification,omitempty"`

	// MinTimeElapsedSinceLastModification: If unspecified,
	// `minTimeElapsedSinceLastModification` takes a zero value
	// and `maxTimeElapsedSinceLastModification` takes the maximum
	// possible
	// value of Duration. Objects that satisfy the object conditions
	// must either have a `lastModificationTime` greater or equal to
	// `NOW` - `maxTimeElapsedSinceLastModification` and less than
	// `NOW` - `minTimeElapsedSinceLastModification`, or not have
	// a
	// `lastModificationTime`.
	MinTimeElapsedSinceLastModification string `json:"minTimeElapsedSinceLastModification,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExcludePrefixes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExcludePrefixes") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ObjectConditions) MarshalJSON() ([]byte, error) {
	type noMethod ObjectConditions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a
// network API call.
type Operation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *Status `json:"error,omitempty"`

	// Metadata: Represents the transfer operation object.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that originally returns it. If you use the default HTTP
	// mapping, the `name` should have the format of
	// `transferOperations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success.
	// If the original
	// method returns no data on success, such as `Delete`, the response
	// is
	// `google.protobuf.Empty`.  If the original method is
	// standard
	// `Get`/`Create`/`Update`, the response should be the resource.  For
	// other
	// methods, the response should have the type `XxxResponse`, where
	// `Xxx`
	// is the original method name.  For example, if the original method
	// name
	// is `TakeSnapshot()`, the inferred response type
	// is
	// `TakeSnapshotResponse`.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Done") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Done") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PauseTransferOperationRequest: Request passed to
// PauseTransferOperation.
type PauseTransferOperationRequest struct {
}

// ResumeTransferOperationRequest: Request passed to
// ResumeTransferOperation.
type ResumeTransferOperationRequest struct {
}

// Schedule: Transfers can be scheduled to recur or to run just once.
type Schedule struct {
	// ScheduleEndDate: The last day the recurring transfer will be run. If
	// `scheduleEndDate`
	// is the same as `scheduleStartDate`, the transfer will be executed
	// only
	// once.
	ScheduleEndDate *Date `json:"scheduleEndDate,omitempty"`

	// ScheduleStartDate: The first day the recurring transfer is scheduled
	// to run. If
	// `scheduleStartDate` is in the past, the transfer will run for the
	// first
	// time on the following day.
	// Required.
	ScheduleStartDate *Date `json:"scheduleStartDate,omitempty"`

	// StartTimeOfDay: The time in UTC at which the transfer will be
	// scheduled to start in a day.
	// Transfers may start later than this time. If not specified, recurring
	// and
	// one-time transfers that are scheduled to run today will run
	// immediately;
	// recurring transfers that are scheduled to run on a future date will
	// start
	// at approximately midnight UTC on that date. Note that when
	// configuring a
	// transfer with the Cloud Platform Console, the transfer's start time
	// in a
	// day is specified in your local timezone.
	StartTimeOfDay *TimeOfDay `json:"startTimeOfDay,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ScheduleEndDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ScheduleEndDate") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Schedule) MarshalJSON() ([]byte, error) {
	type noMethod Schedule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Status: The `Status` type defines a logical error model that is
// suitable for different
// programming environments, including REST APIs and RPC APIs. It is
// used by
// [gRPC](https://github.com/grpc). The error model is designed to
// be:
//
// - Simple to use and understand for most users
// - Flexible enough to meet unexpected needs
//
// # Overview
//
// The `Status` message contains three pieces of data: error code, error
// message,
// and error details. The error code should be an enum value
// of
// google.rpc.Code, but it may accept additional error codes if needed.
// The
// error message should be a developer-facing English message that
// helps
// developers *understand* and *resolve* the error. If a localized
// user-facing
// error message is needed, put the localized message in the error
// details or
// localize it in the client. The optional error details may contain
// arbitrary
// information about the error. There is a predefined set of error
// detail types
// in the package `google.rpc` that can be used for common error
// conditions.
//
// # Language mapping
//
// The `Status` message is the logical representation of the error
// model, but it
// is not necessarily the actual wire format. When the `Status` message
// is
// exposed in different client libraries and different wire protocols,
// it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions
// in Java, but more likely mapped to some error codes in C.
//
// # Other uses
//
// The error model and the `Status` message can be used in a variety
// of
// environments, either with or without APIs, to provide a
// consistent developer experience across different
// environments.
//
// Example uses of this error model include:
//
// - Partial errors. If a service needs to return partial errors to the
// client,
//     it may embed the `Status` in the normal response to indicate the
// partial
//     errors.
//
// - Workflow errors. A typical workflow has multiple steps. Each step
// may
//     have a `Status` message for error reporting.
//
// - Batch operations. If a client uses batch request and batch
// response, the
//     `Status` message should be used directly inside batch response,
// one for
//     each error sub-response.
//
// - Asynchronous operations. If an API call embeds asynchronous
// operation
//     results in its response, the status of those operations should
// be
//     represented directly using the `Status` message.
//
// - Logging. If some API errors are stored in logs, the message
// `Status` could
//     be used directly after any stripping needed for security/privacy
// reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details.  There is a
	// common set of
	// message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any
	// user-facing error message should be localized and sent in
	// the
	// google.rpc.Status.details field, or localized by the client.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TimeOfDay: Represents a time of day. The date and time zone are
// either not significant
// or are specified elsewhere. An API may choose to allow leap seconds.
// Related
// types are google.type.Date and `google.protobuf.Timestamp`.
type TimeOfDay struct {
	// Hours: Hours of day in 24 hour format. Should be from 0 to 23. An API
	// may choose
	// to allow the value "24:00:00" for scenarios like business closing
	// time.
	Hours int64 `json:"hours,omitempty"`

	// Minutes: Minutes of hour of day. Must be from 0 to 59.
	Minutes int64 `json:"minutes,omitempty"`

	// Nanos: Fractions of seconds in nanoseconds. Must be from 0 to
	// 999,999,999.
	Nanos int64 `json:"nanos,omitempty"`

	// Seconds: Seconds of minutes of the time. Must normally be from 0 to
	// 59. An API may
	// allow the value 60 if it allows leap-seconds.
	Seconds int64 `json:"seconds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Hours") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Hours") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimeOfDay) MarshalJSON() ([]byte, error) {
	type noMethod TimeOfDay
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransferCounters: A collection of counters that report the progress
// of a transfer operation.
type TransferCounters struct {
	// BytesCopiedToSink: Bytes that are copied to the data sink.
	BytesCopiedToSink int64 `json:"bytesCopiedToSink,omitempty,string"`

	// BytesDeletedFromSink: Bytes that are deleted from the data sink.
	BytesDeletedFromSink int64 `json:"bytesDeletedFromSink,omitempty,string"`

	// BytesDeletedFromSource: Bytes that are deleted from the data source.
	BytesDeletedFromSource int64 `json:"bytesDeletedFromSource,omitempty,string"`

	// BytesFailedToDeleteFromSink: Bytes that failed to be deleted from the
	// data sink.
	BytesFailedToDeleteFromSink int64 `json:"bytesFailedToDeleteFromSink,omitempty,string"`

	// BytesFoundFromSource: Bytes found in the data source that are
	// scheduled to be transferred,
	// which will be copied, excluded based on conditions, or skipped due
	// to
	// failures.
	BytesFoundFromSource int64 `json:"bytesFoundFromSource,omitempty,string"`

	// BytesFoundOnlyFromSink: Bytes found only in the data sink that are
	// scheduled to be deleted.
	BytesFoundOnlyFromSink int64 `json:"bytesFoundOnlyFromSink,omitempty,string"`

	// BytesFromSourceFailed: Bytes in the data source that failed during
	// the transfer.
	BytesFromSourceFailed int64 `json:"bytesFromSourceFailed,omitempty,string"`

	// BytesFromSourceSkippedBySync: Bytes in the data source that are not
	// transferred because they already
	// exist in the data sink.
	BytesFromSourceSkippedBySync int64 `json:"bytesFromSourceSkippedBySync,omitempty,string"`

	// ObjectsCopiedToSink: Objects that are copied to the data sink.
	ObjectsCopiedToSink int64 `json:"objectsCopiedToSink,omitempty,string"`

	// ObjectsDeletedFromSink: Objects that are deleted from the data sink.
	ObjectsDeletedFromSink int64 `json:"objectsDeletedFromSink,omitempty,string"`

	// ObjectsDeletedFromSource: Objects that are deleted from the data
	// source.
	ObjectsDeletedFromSource int64 `json:"objectsDeletedFromSource,omitempty,string"`

	// ObjectsFailedToDeleteFromSink: Objects that failed to be deleted from
	// the data sink.
	ObjectsFailedToDeleteFromSink int64 `json:"objectsFailedToDeleteFromSink,omitempty,string"`

	// ObjectsFoundFromSource: Objects found in the data source that are
	// scheduled to be transferred,
	// which will be copied, excluded based on conditions, or skipped due
	// to
	// failures.
	ObjectsFoundFromSource int64 `json:"objectsFoundFromSource,omitempty,string"`

	// ObjectsFoundOnlyFromSink: Objects found only in the data sink that
	// are scheduled to be deleted.
	ObjectsFoundOnlyFromSink int64 `json:"objectsFoundOnlyFromSink,omitempty,string"`

	// ObjectsFromSourceFailed: Objects in the data source that failed
	// during the transfer.
	ObjectsFromSourceFailed int64 `json:"objectsFromSourceFailed,omitempty,string"`

	// ObjectsFromSourceSkippedBySync: Objects in the data source that are
	// not transferred because they already
	// exist in the data sink.
	ObjectsFromSourceSkippedBySync int64 `json:"objectsFromSourceSkippedBySync,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "BytesCopiedToSink")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BytesCopiedToSink") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TransferCounters) MarshalJSON() ([]byte, error) {
	type noMethod TransferCounters
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransferJob: This resource represents the configuration of a transfer
// job that runs
// periodically.
type TransferJob struct {
	// CreationTime: This field cannot be changed by user requests.
	CreationTime string `json:"creationTime,omitempty"`

	// DeletionTime: This field cannot be changed by user requests.
	DeletionTime string `json:"deletionTime,omitempty"`

	// Description: A description provided by the user for the job. Its max
	// length is 1024
	// bytes when Unicode-encoded.
	Description string `json:"description,omitempty"`

	// LastModificationTime: This field cannot be changed by user requests.
	LastModificationTime string `json:"lastModificationTime,omitempty"`

	// Name: A globally unique name assigned by Storage Transfer Service
	// when the
	// job is created. This field should be left empty in requests to create
	// a new
	// transfer job; otherwise, the requests result in an
	// `INVALID_ARGUMENT`
	// error.
	Name string `json:"name,omitempty"`

	// ProjectId: The ID of the Google Cloud Platform Console project that
	// owns the job.
	// Required.
	ProjectId string `json:"projectId,omitempty"`

	// Schedule: Schedule specification.
	// Required.
	Schedule *Schedule `json:"schedule,omitempty"`

	// Status: Status of the job. This value MUST be specified
	// for
	// `CreateTransferJobRequests`.
	//
	// NOTE: The effect of the new job status takes place during a
	// subsequent job
	// run. For example, if you change the job status from `ENABLED`
	// to
	// `DISABLED`, and an operation spawned by the transfer is running, the
	// status
	// change would not affect the current operation.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - Zero is an illegal value.
	//   "ENABLED" - New transfers will be performed based on the schedule.
	//   "DISABLED" - New transfers will not be scheduled.
	//   "DELETED" - This is a soft delete state. After a transfer job is
	// set to this
	// state, the job and all the transfer executions are subject to
	// garbage collection.
	Status string `json:"status,omitempty"`

	// TransferSpec: Transfer specification.
	// Required.
	TransferSpec *TransferSpec `json:"transferSpec,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TransferJob) MarshalJSON() ([]byte, error) {
	type noMethod TransferJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransferOperation: A description of the execution of a transfer.
type TransferOperation struct {
	// Counters: Information about the progress of the transfer operation.
	Counters *TransferCounters `json:"counters,omitempty"`

	// EndTime: End time of this transfer execution.
	EndTime string `json:"endTime,omitempty"`

	// ErrorBreakdowns: Summarizes errors encountered with sample error log
	// entries.
	ErrorBreakdowns []*ErrorSummary `json:"errorBreakdowns,omitempty"`

	// Name: A globally unique ID assigned by the system.
	Name string `json:"name,omitempty"`

	// ProjectId: The ID of the Google Cloud Platform Console project that
	// owns the operation.
	// Required.
	ProjectId string `json:"projectId,omitempty"`

	// StartTime: Start time of this transfer execution.
	StartTime string `json:"startTime,omitempty"`

	// Status: Status of the transfer operation.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - Zero is an illegal value.
	//   "IN_PROGRESS" - In progress.
	//   "PAUSED" - Paused.
	//   "SUCCESS" - Completed successfully.
	//   "FAILED" - Terminated due to an unrecoverable failure.
	//   "ABORTED" - Aborted by the user.
	Status string `json:"status,omitempty"`

	// TransferJobName: The name of the transfer job that triggers this
	// transfer operation.
	TransferJobName string `json:"transferJobName,omitempty"`

	// TransferSpec: Transfer specification.
	// Required.
	TransferSpec *TransferSpec `json:"transferSpec,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Counters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Counters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TransferOperation) MarshalJSON() ([]byte, error) {
	type noMethod TransferOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransferOptions: TransferOptions uses three boolean parameters to
// define the actions
// to be performed on objects in a transfer.
type TransferOptions struct {
	// DeleteObjectsFromSourceAfterTransfer: Whether objects should be
	// deleted from the source after they are
	// transferred to the sink.  Note that this option
	// and
	// `deleteObjectsUniqueInSink` are mutually exclusive.
	DeleteObjectsFromSourceAfterTransfer bool `json:"deleteObjectsFromSourceAfterTransfer,omitempty"`

	// DeleteObjectsUniqueInSink: Whether objects that exist only in the
	// sink should be deleted.  Note that
	// this option and `deleteObjectsFromSourceAfterTransfer` are
	// mutually
	// exclusive.
	DeleteObjectsUniqueInSink bool `json:"deleteObjectsUniqueInSink,omitempty"`

	// OverwriteObjectsAlreadyExistingInSink: Whether overwriting objects
	// that already exist in the sink is allowed.
	OverwriteObjectsAlreadyExistingInSink bool `json:"overwriteObjectsAlreadyExistingInSink,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DeleteObjectsFromSourceAfterTransfer") to unconditionally include in
	// API requests. By default, fields with empty values are omitted from
	// API requests. However, any non-pointer, non-interface field appearing
	// in ForceSendFields will be sent to the server regardless of whether
	// the field is empty or not. This may be used to include empty fields
	// in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "DeleteObjectsFromSourceAfterTransfer") to include in API requests
	// with the JSON null value. By default, fields with empty values are
	// omitted from API requests. However, any field with an empty value
	// appearing in NullFields will be sent to the server as null. It is an
	// error if a field in this list has a non-empty value. This may be used
	// to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TransferOptions) MarshalJSON() ([]byte, error) {
	type noMethod TransferOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransferSpec: Configuration for running a transfer.
type TransferSpec struct {
	// AwsS3DataSource: An AWS S3 data source.
	AwsS3DataSource *AwsS3Data `json:"awsS3DataSource,omitempty"`

	// GcsDataSink: A Google Cloud Storage data sink.
	GcsDataSink *GcsData `json:"gcsDataSink,omitempty"`

	// GcsDataSource: A Google Cloud Storage data source.
	GcsDataSource *GcsData `json:"gcsDataSource,omitempty"`

	// HttpDataSource: An HTTP URL data source.
	HttpDataSource *HttpData `json:"httpDataSource,omitempty"`

	// ObjectConditions: Only objects that satisfy these object conditions
	// are included in the set
	// of data source and data sink objects.  Object conditions based
	// on
	// objects' `lastModificationTime` do not exclude objects in a data
	// sink.
	ObjectConditions *ObjectConditions `json:"objectConditions,omitempty"`

	// TransferOptions: If the option `deleteObjectsUniqueInSink` is `true`,
	// object conditions
	// based on objects' `lastModificationTime` are ignored and do not
	// exclude
	// objects in a data source or a data sink.
	TransferOptions *TransferOptions `json:"transferOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AwsS3DataSource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AwsS3DataSource") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TransferSpec) MarshalJSON() ([]byte, error) {
	type noMethod TransferSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateTransferJobRequest: Request passed to UpdateTransferJob.
type UpdateTransferJobRequest struct {
	// ProjectId: The ID of the Google Cloud Platform Console project that
	// owns the job.
	// Required.
	ProjectId string `json:"projectId,omitempty"`

	// TransferJob: The job to update. `transferJob` is expected to specify
	// only three fields:
	// `description`, `transferSpec`, and `status`.  An
	// UpdateTransferJobRequest
	// that specifies other fields will be rejected with an
	// error
	// `INVALID_ARGUMENT`.
	// Required.
	TransferJob *TransferJob `json:"transferJob,omitempty"`

	// UpdateTransferJobFieldMask: The field mask of the fields in
	// `transferJob` that are to be updated in
	// this request.  Fields in `transferJob` that can be updated
	// are:
	// `description`, `transferSpec`, and `status`.  To update the
	// `transferSpec`
	// of the job, a complete transfer specification has to be provided.
	// An
	// incomplete specification which misses any required fields will be
	// rejected
	// with the error `INVALID_ARGUMENT`.
	UpdateTransferJobFieldMask string `json:"updateTransferJobFieldMask,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProjectId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProjectId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateTransferJobRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateTransferJobRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "storagetransfer.googleServiceAccounts.get":

type GoogleServiceAccountsGetCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the Google service account that is used by Storage
// Transfer
// Service to access buckets in the project where transfers
// run or in other projects. Each Google service account is
// associated
// with one Google Cloud Platform Console project. Users
// should add this service account to the Google Cloud Storage
// bucket
// ACLs to grant access to Storage Transfer Service. This
// service
// account is created and owned by Storage Transfer Service and can
// only be used by Storage Transfer Service.
func (r *GoogleServiceAccountsService) Get(projectId string) *GoogleServiceAccountsGetCall {
	c := &GoogleServiceAccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GoogleServiceAccountsGetCall) Fields(s ...googleapi.Field) *GoogleServiceAccountsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GoogleServiceAccountsGetCall) IfNoneMatch(entityTag string) *GoogleServiceAccountsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GoogleServiceAccountsGetCall) Context(ctx context.Context) *GoogleServiceAccountsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GoogleServiceAccountsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GoogleServiceAccountsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/googleServiceAccounts/{projectId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.googleServiceAccounts.get" call.
// Exactly one of *GoogleServiceAccount or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GoogleServiceAccount.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GoogleServiceAccountsGetCall) Do(opts ...googleapi.CallOption) (*GoogleServiceAccount, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &GoogleServiceAccount{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the Google service account that is used by Storage Transfer\nService to access buckets in the project where transfers\nrun or in other projects. Each Google service account is associated\nwith one Google Cloud Platform Console project. Users\nshould add this service account to the Google Cloud Storage bucket\nACLs to grant access to Storage Transfer Service. This service\naccount is created and owned by Storage Transfer Service and can\nonly be used by Storage Transfer Service.",
	//   "flatPath": "v1/googleServiceAccounts/{projectId}",
	//   "httpMethod": "GET",
	//   "id": "storagetransfer.googleServiceAccounts.get",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the Google Cloud Platform Console project that the Google service\naccount is associated with.\nRequired.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/googleServiceAccounts/{projectId}",
	//   "response": {
	//     "$ref": "GoogleServiceAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferJobs.create":

type TransferJobsCreateCall struct {
	s           *Service
	transferjob *TransferJob
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Create: Creates a transfer job that runs periodically.
func (r *TransferJobsService) Create(transferjob *TransferJob) *TransferJobsCreateCall {
	c := &TransferJobsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.transferjob = transferjob
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferJobsCreateCall) Fields(s ...googleapi.Field) *TransferJobsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferJobsCreateCall) Context(ctx context.Context) *TransferJobsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferJobsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferJobsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.transferjob)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/transferJobs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferJobs.create" call.
// Exactly one of *TransferJob or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TransferJob.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TransferJobsCreateCall) Do(opts ...googleapi.CallOption) (*TransferJob, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TransferJob{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a transfer job that runs periodically.",
	//   "flatPath": "v1/transferJobs",
	//   "httpMethod": "POST",
	//   "id": "storagetransfer.transferJobs.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/transferJobs",
	//   "request": {
	//     "$ref": "TransferJob"
	//   },
	//   "response": {
	//     "$ref": "TransferJob"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferJobs.get":

type TransferJobsGetCall struct {
	s            *Service
	jobName      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a transfer job.
func (r *TransferJobsService) Get(jobName string) *TransferJobsGetCall {
	c := &TransferJobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobName = jobName
	return c
}

// ProjectId sets the optional parameter "projectId": The ID of the
// Google Cloud Platform Console project that owns the job.
// Required.
func (c *TransferJobsGetCall) ProjectId(projectId string) *TransferJobsGetCall {
	c.urlParams_.Set("projectId", projectId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferJobsGetCall) Fields(s ...googleapi.Field) *TransferJobsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TransferJobsGetCall) IfNoneMatch(entityTag string) *TransferJobsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferJobsGetCall) Context(ctx context.Context) *TransferJobsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferJobsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferJobsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+jobName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"jobName": c.jobName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferJobs.get" call.
// Exactly one of *TransferJob or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TransferJob.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TransferJobsGetCall) Do(opts ...googleapi.CallOption) (*TransferJob, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TransferJob{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a transfer job.",
	//   "flatPath": "v1/transferJobs/{transferJobsId}",
	//   "httpMethod": "GET",
	//   "id": "storagetransfer.transferJobs.get",
	//   "parameterOrder": [
	//     "jobName"
	//   ],
	//   "parameters": {
	//     "jobName": {
	//       "description": "The job to get.\nRequired.",
	//       "location": "path",
	//       "pattern": "^transferJobs/.+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The ID of the Google Cloud Platform Console project that owns the job.\nRequired.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+jobName}",
	//   "response": {
	//     "$ref": "TransferJob"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferJobs.list":

type TransferJobsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists transfer jobs.
func (r *TransferJobsService) List() *TransferJobsListCall {
	c := &TransferJobsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Filter sets the optional parameter "filter": A list of query
// parameters specified as JSON text in the form
// of
// {"project_id":"my_project_id",
// "job_names":["jobid1","jobid2",...],
//
// "job_statuses":["status1","status2",...]}.
// Since `job_names` and `job_statuses` support multiple values, their
// values
// must be specified with array notation. `project_id` is required.
// `job_names`
// and `job_statuses` are optional.  The valid values for `job_statuses`
// are
// case-insensitive: `ENABLED`, `DISABLED`, and `DELETED`.
func (c *TransferJobsListCall) Filter(filter string) *TransferJobsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The list page size.
// The max allowed value is 256.
func (c *TransferJobsListCall) PageSize(pageSize int64) *TransferJobsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The list page
// token.
func (c *TransferJobsListCall) PageToken(pageToken string) *TransferJobsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferJobsListCall) Fields(s ...googleapi.Field) *TransferJobsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TransferJobsListCall) IfNoneMatch(entityTag string) *TransferJobsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferJobsListCall) Context(ctx context.Context) *TransferJobsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferJobsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferJobsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/transferJobs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferJobs.list" call.
// Exactly one of *ListTransferJobsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListTransferJobsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TransferJobsListCall) Do(opts ...googleapi.CallOption) (*ListTransferJobsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &ListTransferJobsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists transfer jobs.",
	//   "flatPath": "v1/transferJobs",
	//   "httpMethod": "GET",
	//   "id": "storagetransfer.transferJobs.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "filter": {
	//       "description": "A list of query parameters specified as JSON text in the form of\n{\"project_id\":\"my_project_id\",\n\"job_names\":[\"jobid1\",\"jobid2\",...],\n\"job_statuses\":[\"status1\",\"status2\",...]}.\nSince `job_names` and `job_statuses` support multiple values, their values\nmust be specified with array notation. `project_id` is required. `job_names`\nand `job_statuses` are optional.  The valid values for `job_statuses` are\ncase-insensitive: `ENABLED`, `DISABLED`, and `DELETED`.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The list page size. The max allowed value is 256.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/transferJobs",
	//   "response": {
	//     "$ref": "ListTransferJobsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TransferJobsListCall) Pages(ctx context.Context, f func(*ListTransferJobsResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "storagetransfer.transferJobs.patch":

type TransferJobsPatchCall struct {
	s                        *Service
	jobName                  string
	updatetransferjobrequest *UpdateTransferJobRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Patch: Updates a transfer job. Updating a job's transfer spec does
// not affect
// transfer operations that are running already. Updating the
// scheduling
// of a job is not allowed.
func (r *TransferJobsService) Patch(jobName string, updatetransferjobrequest *UpdateTransferJobRequest) *TransferJobsPatchCall {
	c := &TransferJobsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobName = jobName
	c.updatetransferjobrequest = updatetransferjobrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferJobsPatchCall) Fields(s ...googleapi.Field) *TransferJobsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferJobsPatchCall) Context(ctx context.Context) *TransferJobsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferJobsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferJobsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updatetransferjobrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+jobName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"jobName": c.jobName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferJobs.patch" call.
// Exactly one of *TransferJob or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TransferJob.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TransferJobsPatchCall) Do(opts ...googleapi.CallOption) (*TransferJob, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TransferJob{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a transfer job. Updating a job's transfer spec does not affect\ntransfer operations that are running already. Updating the scheduling\nof a job is not allowed.",
	//   "flatPath": "v1/transferJobs/{transferJobsId}",
	//   "httpMethod": "PATCH",
	//   "id": "storagetransfer.transferJobs.patch",
	//   "parameterOrder": [
	//     "jobName"
	//   ],
	//   "parameters": {
	//     "jobName": {
	//       "description": "The name of job to update.\nRequired.",
	//       "location": "path",
	//       "pattern": "^transferJobs/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+jobName}",
	//   "request": {
	//     "$ref": "UpdateTransferJobRequest"
	//   },
	//   "response": {
	//     "$ref": "TransferJob"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferOperations.cancel":

type TransferOperationsCancelCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Cancel: Cancels a transfer. Use the get method to check whether the
// cancellation succeeded or whether the operation completed despite
// cancellation.
func (r *TransferOperationsService) Cancel(name string) *TransferOperationsCancelCall {
	c := &TransferOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsCancelCall) Fields(s ...googleapi.Field) *TransferOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsCancelCall) Context(ctx context.Context) *TransferOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TransferOperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Cancels a transfer. Use the get method to check whether the cancellation succeeded or whether the operation completed despite cancellation.",
	//   "flatPath": "v1/transferOperations/{transferOperationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "storagetransfer.transferOperations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^transferOperations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:cancel",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferOperations.delete":

type TransferOperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: This method is not supported and the server returns
// `UNIMPLEMENTED`.
func (r *TransferOperationsService) Delete(name string) *TransferOperationsDeleteCall {
	c := &TransferOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsDeleteCall) Fields(s ...googleapi.Field) *TransferOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsDeleteCall) Context(ctx context.Context) *TransferOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TransferOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "This method is not supported and the server returns `UNIMPLEMENTED`.",
	//   "flatPath": "v1/transferOperations/{transferOperationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "storagetransfer.transferOperations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^transferOperations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferOperations.get":

type TransferOperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation.  Clients can
// use this
// method to poll the operation result at intervals as recommended by
// the API
// service.
func (r *TransferOperationsService) Get(name string) *TransferOperationsGetCall {
	c := &TransferOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsGetCall) Fields(s ...googleapi.Field) *TransferOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TransferOperationsGetCall) IfNoneMatch(entityTag string) *TransferOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsGetCall) Context(ctx context.Context) *TransferOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TransferOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the latest state of a long-running operation.  Clients can use this\nmethod to poll the operation result at intervals as recommended by the API\nservice.",
	//   "flatPath": "v1/transferOperations/{transferOperationsId}",
	//   "httpMethod": "GET",
	//   "id": "storagetransfer.transferOperations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^transferOperations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferOperations.list":

type TransferOperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists operations that match the specified filter in the
// request. If the
// server doesn't support this method, it returns
// `UNIMPLEMENTED`.
//
// NOTE: the `name` binding allows API services to override the
// binding
// to use different resource name schemes, such as `users/*/operations`.
// To
// override the binding, API services can add a binding such
// as
// "/v1/{name=users/*}/operations" to their service configuration.
// For backwards compatibility, the default name includes the
// operations
// collection id, however overriding users must ensure the name
// binding
// is the parent resource, without the operations collection id.
func (r *TransferOperationsService) List(name string) *TransferOperationsListCall {
	c := &TransferOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": A list of query
// parameters specified as JSON text in the form of {\"project_id\" :
// \"my_project_id\", \"job_names\" : [\"jobid1\", \"jobid2\",...],
// \"operation_names\" : [\"opid1\", \"opid2\",...],
// \"transfer_statuses\":[\"status1\", \"status2\",...]}. Since
// `job_names`, `operation_names`, and `transfer_statuses` support
// multiple values, they must be specified with array notation.
// `job_names`, `operation_names`, and `transfer_statuses` are optional.
func (c *TransferOperationsListCall) Filter(filter string) *TransferOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The list page size.
// The max allowed value is 256.
func (c *TransferOperationsListCall) PageSize(pageSize int64) *TransferOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The list page
// token.
func (c *TransferOperationsListCall) PageToken(pageToken string) *TransferOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsListCall) Fields(s ...googleapi.Field) *TransferOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TransferOperationsListCall) IfNoneMatch(entityTag string) *TransferOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsListCall) Context(ctx context.Context) *TransferOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TransferOperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &ListOperationsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists operations that match the specified filter in the request. If the\nserver doesn't support this method, it returns `UNIMPLEMENTED`.\n\nNOTE: the `name` binding allows API services to override the binding\nto use different resource name schemes, such as `users/*/operations`. To\noverride the binding, API services can add a binding such as\n`\"/v1/{name=users/*}/operations\"` to their service configuration.\nFor backwards compatibility, the default name includes the operations\ncollection id, however overriding users must ensure the name binding\nis the parent resource, without the operations collection id.",
	//   "flatPath": "v1/transferOperations",
	//   "httpMethod": "GET",
	//   "id": "storagetransfer.transferOperations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "A list of query parameters specified as JSON text in the form of {\\\"project_id\\\" : \\\"my_project_id\\\", \\\"job_names\\\" : [\\\"jobid1\\\", \\\"jobid2\\\",...], \\\"operation_names\\\" : [\\\"opid1\\\", \\\"opid2\\\",...], \\\"transfer_statuses\\\":[\\\"status1\\\", \\\"status2\\\",...]}. Since `job_names`, `operation_names`, and `transfer_statuses` support multiple values, they must be specified with array notation. `job_names`, `operation_names`, and `transfer_statuses` are optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The value `transferOperations`.",
	//       "location": "path",
	//       "pattern": "^transferOperations$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The list page size. The max allowed value is 256.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TransferOperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "storagetransfer.transferOperations.pause":

type TransferOperationsPauseCall struct {
	s                             *Service
	name                          string
	pausetransferoperationrequest *PauseTransferOperationRequest
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// Pause: Pauses a transfer operation.
func (r *TransferOperationsService) Pause(name string, pausetransferoperationrequest *PauseTransferOperationRequest) *TransferOperationsPauseCall {
	c := &TransferOperationsPauseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.pausetransferoperationrequest = pausetransferoperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsPauseCall) Fields(s ...googleapi.Field) *TransferOperationsPauseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsPauseCall) Context(ctx context.Context) *TransferOperationsPauseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsPauseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsPauseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pausetransferoperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:pause")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.pause" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TransferOperationsPauseCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Pauses a transfer operation.",
	//   "flatPath": "v1/transferOperations/{transferOperationsId}:pause",
	//   "httpMethod": "POST",
	//   "id": "storagetransfer.transferOperations.pause",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the transfer operation.\nRequired.",
	//       "location": "path",
	//       "pattern": "^transferOperations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:pause",
	//   "request": {
	//     "$ref": "PauseTransferOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "storagetransfer.transferOperations.resume":

type TransferOperationsResumeCall struct {
	s                              *Service
	name                           string
	resumetransferoperationrequest *ResumeTransferOperationRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// Resume: Resumes a transfer operation that is paused.
func (r *TransferOperationsService) Resume(name string, resumetransferoperationrequest *ResumeTransferOperationRequest) *TransferOperationsResumeCall {
	c := &TransferOperationsResumeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.resumetransferoperationrequest = resumetransferoperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TransferOperationsResumeCall) Fields(s ...googleapi.Field) *TransferOperationsResumeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TransferOperationsResumeCall) Context(ctx context.Context) *TransferOperationsResumeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TransferOperationsResumeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TransferOperationsResumeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.resumetransferoperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:resume")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "storagetransfer.transferOperations.resume" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TransferOperationsResumeCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Resumes a transfer operation that is paused.",
	//   "flatPath": "v1/transferOperations/{transferOperationsId}:resume",
	//   "httpMethod": "POST",
	//   "id": "storagetransfer.transferOperations.resume",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the transfer operation.\nRequired.",
	//       "location": "path",
	//       "pattern": "^transferOperations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:resume",
	//   "request": {
	//     "$ref": "ResumeTransferOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
