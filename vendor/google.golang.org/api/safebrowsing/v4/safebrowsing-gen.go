// Package safebrowsing provides access to the Google Safe Browsing API.
//
// See https://developers.google.com/safe-browsing/
//
// Usage example:
//
//   import "google.golang.org/api/safebrowsing/v4"
//   ...
//   safebrowsingService, err := safebrowsing.New(oauthHttpClient)
package safebrowsing // import "google.golang.org/api/safebrowsing/v4"

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

const apiId = "safebrowsing:v4"
const apiName = "safebrowsing"
const apiVersion = "v4"
const basePath = "https://safebrowsing.googleapis.com/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.EncodedFullHashes = NewEncodedFullHashesService(s)
	s.EncodedUpdates = NewEncodedUpdatesService(s)
	s.FullHashes = NewFullHashesService(s)
	s.ThreatListUpdates = NewThreatListUpdatesService(s)
	s.ThreatLists = NewThreatListsService(s)
	s.ThreatMatches = NewThreatMatchesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	EncodedFullHashes *EncodedFullHashesService

	EncodedUpdates *EncodedUpdatesService

	FullHashes *FullHashesService

	ThreatListUpdates *ThreatListUpdatesService

	ThreatLists *ThreatListsService

	ThreatMatches *ThreatMatchesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewEncodedFullHashesService(s *Service) *EncodedFullHashesService {
	rs := &EncodedFullHashesService{s: s}
	return rs
}

type EncodedFullHashesService struct {
	s *Service
}

func NewEncodedUpdatesService(s *Service) *EncodedUpdatesService {
	rs := &EncodedUpdatesService{s: s}
	return rs
}

type EncodedUpdatesService struct {
	s *Service
}

func NewFullHashesService(s *Service) *FullHashesService {
	rs := &FullHashesService{s: s}
	return rs
}

type FullHashesService struct {
	s *Service
}

func NewThreatListUpdatesService(s *Service) *ThreatListUpdatesService {
	rs := &ThreatListUpdatesService{s: s}
	return rs
}

type ThreatListUpdatesService struct {
	s *Service
}

func NewThreatListsService(s *Service) *ThreatListsService {
	rs := &ThreatListsService{s: s}
	return rs
}

type ThreatListsService struct {
	s *Service
}

func NewThreatMatchesService(s *Service) *ThreatMatchesService {
	rs := &ThreatMatchesService{s: s}
	return rs
}

type ThreatMatchesService struct {
	s *Service
}

// Checksum: The expected state of a client's local database.
type Checksum struct {
	// Sha256: The SHA256 hash of the client state; that is, of the sorted
	// list of all
	// hashes present in the database.
	Sha256 string `json:"sha256,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Sha256") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Sha256") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Checksum) MarshalJSON() ([]byte, error) {
	type noMethod Checksum
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClientInfo: The client metadata associated with Safe Browsing API
// requests.
type ClientInfo struct {
	// ClientId: A client ID that (hopefully) uniquely identifies the client
	// implementation
	// of the Safe Browsing API.
	ClientId string `json:"clientId,omitempty"`

	// ClientVersion: The version of the client implementation.
	ClientVersion string `json:"clientVersion,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClientId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClientInfo) MarshalJSON() ([]byte, error) {
	type noMethod ClientInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Constraints: The constraints for this update.
type Constraints struct {
	// MaxDatabaseEntries: Sets the maximum number of entries that the
	// client is willing to have
	// in the local database. This should be a power of 2 between 2**10
	// and
	// 2**20. If zero, no database size limit is set.
	MaxDatabaseEntries int64 `json:"maxDatabaseEntries,omitempty"`

	// MaxUpdateEntries: The maximum size in number of entries. The update
	// will not contain more
	// entries than this value.  This should be a power of 2 between 2**10
	// and
	// 2**20.  If zero, no update size limit is set.
	MaxUpdateEntries int64 `json:"maxUpdateEntries,omitempty"`

	// Region: Requests the list for a specific geographic location. If not
	// set the
	// server may pick that value based on the user's IP address. Expects
	// ISO
	// 3166-1 alpha-2 format.
	Region string `json:"region,omitempty"`

	// SupportedCompressions: The compression types supported by the client.
	//
	// Possible values:
	//   "COMPRESSION_TYPE_UNSPECIFIED" - Unknown.
	//   "RAW" - Raw, uncompressed data.
	//   "RICE" - Rice-Golomb encoded data.
	SupportedCompressions []string `json:"supportedCompressions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxDatabaseEntries")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxDatabaseEntries") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Constraints) MarshalJSON() ([]byte, error) {
	type noMethod Constraints
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FetchThreatListUpdatesRequest: Describes a Safe Browsing API update
// request. Clients can request updates for
// multiple lists in a single request.
// NOTE: Field index 2 is unused.
// NEXT: 5
type FetchThreatListUpdatesRequest struct {
	// Client: The client metadata.
	Client *ClientInfo `json:"client,omitempty"`

	// ListUpdateRequests: The requested threat list updates.
	ListUpdateRequests []*ListUpdateRequest `json:"listUpdateRequests,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Client") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Client") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FetchThreatListUpdatesRequest) MarshalJSON() ([]byte, error) {
	type noMethod FetchThreatListUpdatesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type FetchThreatListUpdatesResponse struct {
	// ListUpdateResponses: The list updates requested by the clients.
	ListUpdateResponses []*ListUpdateResponse `json:"listUpdateResponses,omitempty"`

	// MinimumWaitDuration: The minimum duration the client must wait before
	// issuing any update
	// request. If this field is not set clients may update as soon as they
	// want.
	MinimumWaitDuration string `json:"minimumWaitDuration,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ListUpdateResponses")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ListUpdateResponses") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FetchThreatListUpdatesResponse) MarshalJSON() ([]byte, error) {
	type noMethod FetchThreatListUpdatesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindFullHashesRequest: Request to return full hashes matched by the
// provided hash prefixes.
type FindFullHashesRequest struct {
	// ApiClient: Client metadata associated with callers of higher-level
	// APIs built on top
	// of the client's implementation.
	ApiClient *ClientInfo `json:"apiClient,omitempty"`

	// Client: The client metadata.
	Client *ClientInfo `json:"client,omitempty"`

	// ClientStates: The current client states for each of the client's
	// local threat lists.
	ClientStates []string `json:"clientStates,omitempty"`

	// ThreatInfo: The lists and hashes to be checked.
	ThreatInfo *ThreatInfo `json:"threatInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApiClient") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApiClient") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindFullHashesRequest) MarshalJSON() ([]byte, error) {
	type noMethod FindFullHashesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type FindFullHashesResponse struct {
	// Matches: The full hashes that matched the requested prefixes.
	Matches []*ThreatMatch `json:"matches,omitempty"`

	// MinimumWaitDuration: The minimum duration the client must wait before
	// issuing any find hashes
	// request. If this field is not set, clients can issue a request as
	// soon as
	// they want.
	MinimumWaitDuration string `json:"minimumWaitDuration,omitempty"`

	// NegativeCacheDuration: For requested entities that did not match the
	// threat list, how long to
	// cache the response.
	NegativeCacheDuration string `json:"negativeCacheDuration,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Matches") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Matches") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindFullHashesResponse) MarshalJSON() ([]byte, error) {
	type noMethod FindFullHashesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindThreatMatchesRequest: Request to check entries against lists.
type FindThreatMatchesRequest struct {
	// Client: The client metadata.
	Client *ClientInfo `json:"client,omitempty"`

	// ThreatInfo: The lists and entries to be checked for matches.
	ThreatInfo *ThreatInfo `json:"threatInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Client") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Client") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindThreatMatchesRequest) MarshalJSON() ([]byte, error) {
	type noMethod FindThreatMatchesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type FindThreatMatchesResponse struct {
	// Matches: The threat list matches.
	Matches []*ThreatMatch `json:"matches,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Matches") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Matches") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindThreatMatchesResponse) MarshalJSON() ([]byte, error) {
	type noMethod FindThreatMatchesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListThreatListsResponse struct {
	// ThreatLists: The lists available for download by the client.
	ThreatLists []*ThreatListDescriptor `json:"threatLists,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ThreatLists") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ThreatLists") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListThreatListsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListThreatListsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListUpdateRequest: A single list update request.
type ListUpdateRequest struct {
	// Constraints: The constraints associated with this request.
	Constraints *Constraints `json:"constraints,omitempty"`

	// PlatformType: The type of platform at risk by entries present in the
	// list.
	//
	// Possible values:
	//   "PLATFORM_TYPE_UNSPECIFIED" - Unknown platform.
	//   "WINDOWS" - Threat posed to Windows.
	//   "LINUX" - Threat posed to Linux.
	//   "ANDROID" - Threat posed to Android.
	//   "OSX" - Threat posed to OS X.
	//   "IOS" - Threat posed to iOS.
	//   "ANY_PLATFORM" - Threat posed to at least one of the defined
	// platforms.
	//   "ALL_PLATFORMS" - Threat posed to all defined platforms.
	//   "CHROME" - Threat posed to Chrome.
	PlatformType string `json:"platformType,omitempty"`

	// State: The current state of the client for the requested list (the
	// encrypted
	// client state that was received from the last successful list update).
	State string `json:"state,omitempty"`

	// ThreatEntryType: The types of entries present in the list.
	//
	// Possible values:
	//   "THREAT_ENTRY_TYPE_UNSPECIFIED" - Unspecified.
	//   "URL" - A URL.
	//   "EXECUTABLE" - An executable program.
	//   "IP_RANGE" - An IP range.
	//   "CHROME_EXTENSION" - Chrome extension.
	//   "FILENAME" - Filename.
	//   "CERT" - CERT
	ThreatEntryType string `json:"threatEntryType,omitempty"`

	// ThreatType: The type of threat posed by entries present in the list.
	//
	// Possible values:
	//   "THREAT_TYPE_UNSPECIFIED" - Unknown.
	//   "MALWARE" - Malware threat type.
	//   "SOCIAL_ENGINEERING" - Social engineering threat type.
	//   "UNWANTED_SOFTWARE" - Unwanted software threat type.
	//   "POTENTIALLY_HARMFUL_APPLICATION" - Potentially harmful application
	// threat type.
	//   "SOCIAL_ENGINEERING_INTERNAL" - Social engineering threat type for
	// internal use.
	//   "API_ABUSE" - API abuse threat type.
	//   "MALICIOUS_BINARY" - Malicious binary threat type.
	//   "CSD_WHITELIST" - Client side detection whitelist threat type.
	//   "CSD_DOWNLOAD_WHITELIST" - Client side download detection whitelist
	// threat type.
	//   "CLIENT_INCIDENT" - Client incident threat type.
	//   "CLIENT_INCIDENT_WHITELIST" - Whitelist used when detecting client
	// incident threats.
	// This enum was never launched and should be re-used for the next list.
	//   "APK_MALWARE_OFFLINE" - List used for offline APK checks in PAM.
	//   "SUBRESOURCE_FILTER" - Patterns to be used for activating the
	// subresource filter. Interstitial
	// will not be shown for patterns from this list.
	ThreatType string `json:"threatType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Constraints") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Constraints") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListUpdateRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListUpdateRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListUpdateResponse: An update to an individual list.
type ListUpdateResponse struct {
	// Additions: A set of entries to add to a local threat type's list.
	// Repeated to allow
	// for a combination of compressed and raw data to be sent in a
	// single
	// response.
	Additions []*ThreatEntrySet `json:"additions,omitempty"`

	// Checksum: The expected SHA256 hash of the client state; that is, of
	// the sorted list
	// of all hashes present in the database after applying the provided
	// update.
	// If the client state doesn't match the expected state, the client
	// must
	// disregard this update and retry later.
	Checksum *Checksum `json:"checksum,omitempty"`

	// NewClientState: The new client state, in encrypted format. Opaque to
	// clients.
	NewClientState string `json:"newClientState,omitempty"`

	// PlatformType: The platform type for which data is returned.
	//
	// Possible values:
	//   "PLATFORM_TYPE_UNSPECIFIED" - Unknown platform.
	//   "WINDOWS" - Threat posed to Windows.
	//   "LINUX" - Threat posed to Linux.
	//   "ANDROID" - Threat posed to Android.
	//   "OSX" - Threat posed to OS X.
	//   "IOS" - Threat posed to iOS.
	//   "ANY_PLATFORM" - Threat posed to at least one of the defined
	// platforms.
	//   "ALL_PLATFORMS" - Threat posed to all defined platforms.
	//   "CHROME" - Threat posed to Chrome.
	PlatformType string `json:"platformType,omitempty"`

	// Removals: A set of entries to remove from a local threat type's list.
	// In practice,
	// this field is empty or contains exactly one ThreatEntrySet.
	Removals []*ThreatEntrySet `json:"removals,omitempty"`

	// ResponseType: The type of response. This may indicate that an action
	// is required by the
	// client when the response is received.
	//
	// Possible values:
	//   "RESPONSE_TYPE_UNSPECIFIED" - Unknown.
	//   "PARTIAL_UPDATE" - Partial updates are applied to the client's
	// existing local database.
	//   "FULL_UPDATE" - Full updates replace the client's entire local
	// database. This means
	// that either the client was seriously out-of-date or the client
	// is
	// believed to be corrupt.
	ResponseType string `json:"responseType,omitempty"`

	// ThreatEntryType: The format of the threats.
	//
	// Possible values:
	//   "THREAT_ENTRY_TYPE_UNSPECIFIED" - Unspecified.
	//   "URL" - A URL.
	//   "EXECUTABLE" - An executable program.
	//   "IP_RANGE" - An IP range.
	//   "CHROME_EXTENSION" - Chrome extension.
	//   "FILENAME" - Filename.
	//   "CERT" - CERT
	ThreatEntryType string `json:"threatEntryType,omitempty"`

	// ThreatType: The threat type for which data is returned.
	//
	// Possible values:
	//   "THREAT_TYPE_UNSPECIFIED" - Unknown.
	//   "MALWARE" - Malware threat type.
	//   "SOCIAL_ENGINEERING" - Social engineering threat type.
	//   "UNWANTED_SOFTWARE" - Unwanted software threat type.
	//   "POTENTIALLY_HARMFUL_APPLICATION" - Potentially harmful application
	// threat type.
	//   "SOCIAL_ENGINEERING_INTERNAL" - Social engineering threat type for
	// internal use.
	//   "API_ABUSE" - API abuse threat type.
	//   "MALICIOUS_BINARY" - Malicious binary threat type.
	//   "CSD_WHITELIST" - Client side detection whitelist threat type.
	//   "CSD_DOWNLOAD_WHITELIST" - Client side download detection whitelist
	// threat type.
	//   "CLIENT_INCIDENT" - Client incident threat type.
	//   "CLIENT_INCIDENT_WHITELIST" - Whitelist used when detecting client
	// incident threats.
	// This enum was never launched and should be re-used for the next list.
	//   "APK_MALWARE_OFFLINE" - List used for offline APK checks in PAM.
	//   "SUBRESOURCE_FILTER" - Patterns to be used for activating the
	// subresource filter. Interstitial
	// will not be shown for patterns from this list.
	ThreatType string `json:"threatType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Additions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Additions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListUpdateResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListUpdateResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetadataEntry: A single metadata entry.
type MetadataEntry struct {
	// Key: The metadata entry key. For JSON requests, the key is
	// base64-encoded.
	Key string `json:"key,omitempty"`

	// Value: The metadata entry value. For JSON requests, the value is
	// base64-encoded.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Key") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetadataEntry) MarshalJSON() ([]byte, error) {
	type noMethod MetadataEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RawHashes: The uncompressed threat entries in hash format of a
// particular prefix length.
// Hashes can be anywhere from 4 to 32 bytes in size. A large majority
// are 4
// bytes, but some hashes are lengthened if they collide with the hash
// of a
// popular URL.
//
// Used for sending ThreatEntrySet to clients that do not support
// compression,
// or when sending non-4-byte hashes to clients that do support
// compression.
type RawHashes struct {
	// PrefixSize: The number of bytes for each prefix encoded below.  This
	// field can be
	// anywhere from 4 (shortest prefix) to 32 (full SHA256 hash).
	PrefixSize int64 `json:"prefixSize,omitempty"`

	// RawHashes: The hashes, in binary format, concatenated into one long
	// string. Hashes are
	// sorted in lexicographic order. For JSON API users, hashes
	// are
	// base64-encoded.
	RawHashes string `json:"rawHashes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PrefixSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PrefixSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RawHashes) MarshalJSON() ([]byte, error) {
	type noMethod RawHashes
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RawIndices: A set of raw indices to remove from a local list.
type RawIndices struct {
	// Indices: The indices to remove from a lexicographically-sorted local
	// list.
	Indices []int64 `json:"indices,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Indices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Indices") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RawIndices) MarshalJSON() ([]byte, error) {
	type noMethod RawIndices
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RiceDeltaEncoding: The Rice-Golomb encoded data. Used for sending
// compressed 4-byte hashes or
// compressed removal indices.
type RiceDeltaEncoding struct {
	// EncodedData: The encoded deltas that are encoded using the
	// Golomb-Rice coder.
	EncodedData string `json:"encodedData,omitempty"`

	// FirstValue: The offset of the first entry in the encoded data, or, if
	// only a single
	// integer was encoded, that single integer's value.
	FirstValue int64 `json:"firstValue,omitempty,string"`

	// NumEntries: The number of entries that are delta encoded in the
	// encoded data. If only a
	// single integer was encoded, this will be zero and the single value
	// will be
	// stored in `first_value`.
	NumEntries int64 `json:"numEntries,omitempty"`

	// RiceParameter: The Golomb-Rice parameter, which is a number between 2
	// and 28. This field
	// is missing (that is, zero) if `num_entries` is zero.
	RiceParameter int64 `json:"riceParameter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EncodedData") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EncodedData") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RiceDeltaEncoding) MarshalJSON() ([]byte, error) {
	type noMethod RiceDeltaEncoding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatEntry: An individual threat; for example, a malicious URL or
// its hash
// representation. Only one of these fields should be set.
type ThreatEntry struct {
	// Digest: The digest of an executable in SHA256 format. The API
	// supports both
	// binary and hex digests. For JSON requests, digests are
	// base64-encoded.
	Digest string `json:"digest,omitempty"`

	// Hash: A hash prefix, consisting of the most significant 4-32 bytes of
	// a SHA256
	// hash. This field is in binary format. For JSON requests, hashes
	// are
	// base64-encoded.
	Hash string `json:"hash,omitempty"`

	// Url: A URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Digest") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Digest") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThreatEntry) MarshalJSON() ([]byte, error) {
	type noMethod ThreatEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatEntryMetadata: The metadata associated with a specific threat
// entry. The client is expected
// to know the metadata key/value pairs associated with each threat
// type.
type ThreatEntryMetadata struct {
	// Entries: The metadata entries.
	Entries []*MetadataEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThreatEntryMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ThreatEntryMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatEntrySet: A set of threats that should be added or removed from
// a client's local
// database.
type ThreatEntrySet struct {
	// CompressionType: The compression type for the entries in this set.
	//
	// Possible values:
	//   "COMPRESSION_TYPE_UNSPECIFIED" - Unknown.
	//   "RAW" - Raw, uncompressed data.
	//   "RICE" - Rice-Golomb encoded data.
	CompressionType string `json:"compressionType,omitempty"`

	// RawHashes: The raw SHA256-formatted entries.
	RawHashes *RawHashes `json:"rawHashes,omitempty"`

	// RawIndices: The raw removal indices for a local list.
	RawIndices *RawIndices `json:"rawIndices,omitempty"`

	// RiceHashes: The encoded 4-byte prefixes of SHA256-formatted entries,
	// using a
	// Golomb-Rice encoding. The hashes are converted to uint32, sorted
	// in
	// ascending order, then delta encoded and stored as encoded_data.
	RiceHashes *RiceDeltaEncoding `json:"riceHashes,omitempty"`

	// RiceIndices: The encoded local, lexicographically-sorted list
	// indices, using a
	// Golomb-Rice encoding. Used for sending compressed removal indices.
	// The
	// removal indices (uint32) are sorted in ascending order, then delta
	// encoded
	// and stored as encoded_data.
	RiceIndices *RiceDeltaEncoding `json:"riceIndices,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CompressionType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompressionType") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ThreatEntrySet) MarshalJSON() ([]byte, error) {
	type noMethod ThreatEntrySet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatInfo: The information regarding one or more threats that a
// client submits when
// checking for matches in threat lists.
type ThreatInfo struct {
	// PlatformTypes: The platform types to be checked.
	//
	// Possible values:
	//   "PLATFORM_TYPE_UNSPECIFIED" - Unknown platform.
	//   "WINDOWS" - Threat posed to Windows.
	//   "LINUX" - Threat posed to Linux.
	//   "ANDROID" - Threat posed to Android.
	//   "OSX" - Threat posed to OS X.
	//   "IOS" - Threat posed to iOS.
	//   "ANY_PLATFORM" - Threat posed to at least one of the defined
	// platforms.
	//   "ALL_PLATFORMS" - Threat posed to all defined platforms.
	//   "CHROME" - Threat posed to Chrome.
	PlatformTypes []string `json:"platformTypes,omitempty"`

	// ThreatEntries: The threat entries to be checked.
	ThreatEntries []*ThreatEntry `json:"threatEntries,omitempty"`

	// ThreatEntryTypes: The entry types to be checked.
	//
	// Possible values:
	//   "THREAT_ENTRY_TYPE_UNSPECIFIED" - Unspecified.
	//   "URL" - A URL.
	//   "EXECUTABLE" - An executable program.
	//   "IP_RANGE" - An IP range.
	//   "CHROME_EXTENSION" - Chrome extension.
	//   "FILENAME" - Filename.
	//   "CERT" - CERT
	ThreatEntryTypes []string `json:"threatEntryTypes,omitempty"`

	// ThreatTypes: The threat types to be checked.
	//
	// Possible values:
	//   "THREAT_TYPE_UNSPECIFIED" - Unknown.
	//   "MALWARE" - Malware threat type.
	//   "SOCIAL_ENGINEERING" - Social engineering threat type.
	//   "UNWANTED_SOFTWARE" - Unwanted software threat type.
	//   "POTENTIALLY_HARMFUL_APPLICATION" - Potentially harmful application
	// threat type.
	//   "SOCIAL_ENGINEERING_INTERNAL" - Social engineering threat type for
	// internal use.
	//   "API_ABUSE" - API abuse threat type.
	//   "MALICIOUS_BINARY" - Malicious binary threat type.
	//   "CSD_WHITELIST" - Client side detection whitelist threat type.
	//   "CSD_DOWNLOAD_WHITELIST" - Client side download detection whitelist
	// threat type.
	//   "CLIENT_INCIDENT" - Client incident threat type.
	//   "CLIENT_INCIDENT_WHITELIST" - Whitelist used when detecting client
	// incident threats.
	// This enum was never launched and should be re-used for the next list.
	//   "APK_MALWARE_OFFLINE" - List used for offline APK checks in PAM.
	//   "SUBRESOURCE_FILTER" - Patterns to be used for activating the
	// subresource filter. Interstitial
	// will not be shown for patterns from this list.
	ThreatTypes []string `json:"threatTypes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PlatformTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PlatformTypes") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThreatInfo) MarshalJSON() ([]byte, error) {
	type noMethod ThreatInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatListDescriptor: Describes an individual threat list. A list is
// defined by three parameters:
// the type of threat posed, the type of platform targeted by the
// threat, and
// the type of entries in the list.
type ThreatListDescriptor struct {
	// PlatformType: The platform type targeted by the list's entries.
	//
	// Possible values:
	//   "PLATFORM_TYPE_UNSPECIFIED" - Unknown platform.
	//   "WINDOWS" - Threat posed to Windows.
	//   "LINUX" - Threat posed to Linux.
	//   "ANDROID" - Threat posed to Android.
	//   "OSX" - Threat posed to OS X.
	//   "IOS" - Threat posed to iOS.
	//   "ANY_PLATFORM" - Threat posed to at least one of the defined
	// platforms.
	//   "ALL_PLATFORMS" - Threat posed to all defined platforms.
	//   "CHROME" - Threat posed to Chrome.
	PlatformType string `json:"platformType,omitempty"`

	// ThreatEntryType: The entry types contained in the list.
	//
	// Possible values:
	//   "THREAT_ENTRY_TYPE_UNSPECIFIED" - Unspecified.
	//   "URL" - A URL.
	//   "EXECUTABLE" - An executable program.
	//   "IP_RANGE" - An IP range.
	//   "CHROME_EXTENSION" - Chrome extension.
	//   "FILENAME" - Filename.
	//   "CERT" - CERT
	ThreatEntryType string `json:"threatEntryType,omitempty"`

	// ThreatType: The threat type posed by the list's entries.
	//
	// Possible values:
	//   "THREAT_TYPE_UNSPECIFIED" - Unknown.
	//   "MALWARE" - Malware threat type.
	//   "SOCIAL_ENGINEERING" - Social engineering threat type.
	//   "UNWANTED_SOFTWARE" - Unwanted software threat type.
	//   "POTENTIALLY_HARMFUL_APPLICATION" - Potentially harmful application
	// threat type.
	//   "SOCIAL_ENGINEERING_INTERNAL" - Social engineering threat type for
	// internal use.
	//   "API_ABUSE" - API abuse threat type.
	//   "MALICIOUS_BINARY" - Malicious binary threat type.
	//   "CSD_WHITELIST" - Client side detection whitelist threat type.
	//   "CSD_DOWNLOAD_WHITELIST" - Client side download detection whitelist
	// threat type.
	//   "CLIENT_INCIDENT" - Client incident threat type.
	//   "CLIENT_INCIDENT_WHITELIST" - Whitelist used when detecting client
	// incident threats.
	// This enum was never launched and should be re-used for the next list.
	//   "APK_MALWARE_OFFLINE" - List used for offline APK checks in PAM.
	//   "SUBRESOURCE_FILTER" - Patterns to be used for activating the
	// subresource filter. Interstitial
	// will not be shown for patterns from this list.
	ThreatType string `json:"threatType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PlatformType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PlatformType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThreatListDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod ThreatListDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ThreatMatch: A match when checking a threat entry in the Safe
// Browsing threat lists.
type ThreatMatch struct {
	// CacheDuration: The cache lifetime for the returned match. Clients
	// must not cache this
	// response for more than this duration to avoid false positives.
	CacheDuration string `json:"cacheDuration,omitempty"`

	// PlatformType: The platform type matching this threat.
	//
	// Possible values:
	//   "PLATFORM_TYPE_UNSPECIFIED" - Unknown platform.
	//   "WINDOWS" - Threat posed to Windows.
	//   "LINUX" - Threat posed to Linux.
	//   "ANDROID" - Threat posed to Android.
	//   "OSX" - Threat posed to OS X.
	//   "IOS" - Threat posed to iOS.
	//   "ANY_PLATFORM" - Threat posed to at least one of the defined
	// platforms.
	//   "ALL_PLATFORMS" - Threat posed to all defined platforms.
	//   "CHROME" - Threat posed to Chrome.
	PlatformType string `json:"platformType,omitempty"`

	// Threat: The threat matching this threat.
	Threat *ThreatEntry `json:"threat,omitempty"`

	// ThreatEntryMetadata: Optional metadata associated with this threat.
	ThreatEntryMetadata *ThreatEntryMetadata `json:"threatEntryMetadata,omitempty"`

	// ThreatEntryType: The threat entry type matching this threat.
	//
	// Possible values:
	//   "THREAT_ENTRY_TYPE_UNSPECIFIED" - Unspecified.
	//   "URL" - A URL.
	//   "EXECUTABLE" - An executable program.
	//   "IP_RANGE" - An IP range.
	//   "CHROME_EXTENSION" - Chrome extension.
	//   "FILENAME" - Filename.
	//   "CERT" - CERT
	ThreatEntryType string `json:"threatEntryType,omitempty"`

	// ThreatType: The threat type matching this threat.
	//
	// Possible values:
	//   "THREAT_TYPE_UNSPECIFIED" - Unknown.
	//   "MALWARE" - Malware threat type.
	//   "SOCIAL_ENGINEERING" - Social engineering threat type.
	//   "UNWANTED_SOFTWARE" - Unwanted software threat type.
	//   "POTENTIALLY_HARMFUL_APPLICATION" - Potentially harmful application
	// threat type.
	//   "SOCIAL_ENGINEERING_INTERNAL" - Social engineering threat type for
	// internal use.
	//   "API_ABUSE" - API abuse threat type.
	//   "MALICIOUS_BINARY" - Malicious binary threat type.
	//   "CSD_WHITELIST" - Client side detection whitelist threat type.
	//   "CSD_DOWNLOAD_WHITELIST" - Client side download detection whitelist
	// threat type.
	//   "CLIENT_INCIDENT" - Client incident threat type.
	//   "CLIENT_INCIDENT_WHITELIST" - Whitelist used when detecting client
	// incident threats.
	// This enum was never launched and should be re-used for the next list.
	//   "APK_MALWARE_OFFLINE" - List used for offline APK checks in PAM.
	//   "SUBRESOURCE_FILTER" - Patterns to be used for activating the
	// subresource filter. Interstitial
	// will not be shown for patterns from this list.
	ThreatType string `json:"threatType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CacheDuration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CacheDuration") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThreatMatch) MarshalJSON() ([]byte, error) {
	type noMethod ThreatMatch
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "safebrowsing.encodedFullHashes.get":

type EncodedFullHashesGetCall struct {
	s              *Service
	encodedRequest string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// Get:
func (r *EncodedFullHashesService) Get(encodedRequest string) *EncodedFullHashesGetCall {
	c := &EncodedFullHashesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.encodedRequest = encodedRequest
	return c
}

// ClientId sets the optional parameter "clientId": A client ID that
// (hopefully) uniquely identifies the client implementation
// of the Safe Browsing API.
func (c *EncodedFullHashesGetCall) ClientId(clientId string) *EncodedFullHashesGetCall {
	c.urlParams_.Set("clientId", clientId)
	return c
}

// ClientVersion sets the optional parameter "clientVersion": The
// version of the client implementation.
func (c *EncodedFullHashesGetCall) ClientVersion(clientVersion string) *EncodedFullHashesGetCall {
	c.urlParams_.Set("clientVersion", clientVersion)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EncodedFullHashesGetCall) Fields(s ...googleapi.Field) *EncodedFullHashesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EncodedFullHashesGetCall) IfNoneMatch(entityTag string) *EncodedFullHashesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EncodedFullHashesGetCall) Context(ctx context.Context) *EncodedFullHashesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EncodedFullHashesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EncodedFullHashesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/encodedFullHashes/{encodedRequest}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"encodedRequest": c.encodedRequest,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.encodedFullHashes.get" call.
// Exactly one of *FindFullHashesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *FindFullHashesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EncodedFullHashesGetCall) Do(opts ...googleapi.CallOption) (*FindFullHashesResponse, error) {
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
	ret := &FindFullHashesResponse{
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
	//   "description": "",
	//   "flatPath": "v4/encodedFullHashes/{encodedRequest}",
	//   "httpMethod": "GET",
	//   "id": "safebrowsing.encodedFullHashes.get",
	//   "parameterOrder": [
	//     "encodedRequest"
	//   ],
	//   "parameters": {
	//     "clientId": {
	//       "description": "A client ID that (hopefully) uniquely identifies the client implementation\nof the Safe Browsing API.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "clientVersion": {
	//       "description": "The version of the client implementation.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "encodedRequest": {
	//       "description": "A serialized FindFullHashesRequest proto.",
	//       "format": "byte",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/encodedFullHashes/{encodedRequest}",
	//   "response": {
	//     "$ref": "FindFullHashesResponse"
	//   }
	// }

}

// method id "safebrowsing.encodedUpdates.get":

type EncodedUpdatesGetCall struct {
	s              *Service
	encodedRequest string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// Get:
func (r *EncodedUpdatesService) Get(encodedRequest string) *EncodedUpdatesGetCall {
	c := &EncodedUpdatesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.encodedRequest = encodedRequest
	return c
}

// ClientId sets the optional parameter "clientId": A client ID that
// uniquely identifies the client implementation of the Safe
// Browsing API.
func (c *EncodedUpdatesGetCall) ClientId(clientId string) *EncodedUpdatesGetCall {
	c.urlParams_.Set("clientId", clientId)
	return c
}

// ClientVersion sets the optional parameter "clientVersion": The
// version of the client implementation.
func (c *EncodedUpdatesGetCall) ClientVersion(clientVersion string) *EncodedUpdatesGetCall {
	c.urlParams_.Set("clientVersion", clientVersion)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EncodedUpdatesGetCall) Fields(s ...googleapi.Field) *EncodedUpdatesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EncodedUpdatesGetCall) IfNoneMatch(entityTag string) *EncodedUpdatesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EncodedUpdatesGetCall) Context(ctx context.Context) *EncodedUpdatesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EncodedUpdatesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EncodedUpdatesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/encodedUpdates/{encodedRequest}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"encodedRequest": c.encodedRequest,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.encodedUpdates.get" call.
// Exactly one of *FetchThreatListUpdatesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *FetchThreatListUpdatesResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EncodedUpdatesGetCall) Do(opts ...googleapi.CallOption) (*FetchThreatListUpdatesResponse, error) {
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
	ret := &FetchThreatListUpdatesResponse{
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
	//   "description": "",
	//   "flatPath": "v4/encodedUpdates/{encodedRequest}",
	//   "httpMethod": "GET",
	//   "id": "safebrowsing.encodedUpdates.get",
	//   "parameterOrder": [
	//     "encodedRequest"
	//   ],
	//   "parameters": {
	//     "clientId": {
	//       "description": "A client ID that uniquely identifies the client implementation of the Safe\nBrowsing API.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "clientVersion": {
	//       "description": "The version of the client implementation.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "encodedRequest": {
	//       "description": "A serialized FetchThreatListUpdatesRequest proto.",
	//       "format": "byte",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/encodedUpdates/{encodedRequest}",
	//   "response": {
	//     "$ref": "FetchThreatListUpdatesResponse"
	//   }
	// }

}

// method id "safebrowsing.fullHashes.find":

type FullHashesFindCall struct {
	s                     *Service
	findfullhashesrequest *FindFullHashesRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Find: Finds the full hashes that match the requested hash prefixes.
func (r *FullHashesService) Find(findfullhashesrequest *FindFullHashesRequest) *FullHashesFindCall {
	c := &FullHashesFindCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.findfullhashesrequest = findfullhashesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FullHashesFindCall) Fields(s ...googleapi.Field) *FullHashesFindCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FullHashesFindCall) Context(ctx context.Context) *FullHashesFindCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FullHashesFindCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FullHashesFindCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.findfullhashesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/fullHashes:find")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.fullHashes.find" call.
// Exactly one of *FindFullHashesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *FindFullHashesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *FullHashesFindCall) Do(opts ...googleapi.CallOption) (*FindFullHashesResponse, error) {
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
	ret := &FindFullHashesResponse{
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
	//   "description": "Finds the full hashes that match the requested hash prefixes.",
	//   "flatPath": "v4/fullHashes:find",
	//   "httpMethod": "POST",
	//   "id": "safebrowsing.fullHashes.find",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/fullHashes:find",
	//   "request": {
	//     "$ref": "FindFullHashesRequest"
	//   },
	//   "response": {
	//     "$ref": "FindFullHashesResponse"
	//   }
	// }

}

// method id "safebrowsing.threatListUpdates.fetch":

type ThreatListUpdatesFetchCall struct {
	s                             *Service
	fetchthreatlistupdatesrequest *FetchThreatListUpdatesRequest
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// Fetch: Fetches the most recent threat list updates. A client can
// request updates
// for multiple lists at once.
func (r *ThreatListUpdatesService) Fetch(fetchthreatlistupdatesrequest *FetchThreatListUpdatesRequest) *ThreatListUpdatesFetchCall {
	c := &ThreatListUpdatesFetchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fetchthreatlistupdatesrequest = fetchthreatlistupdatesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ThreatListUpdatesFetchCall) Fields(s ...googleapi.Field) *ThreatListUpdatesFetchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ThreatListUpdatesFetchCall) Context(ctx context.Context) *ThreatListUpdatesFetchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ThreatListUpdatesFetchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ThreatListUpdatesFetchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.fetchthreatlistupdatesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/threatListUpdates:fetch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.threatListUpdates.fetch" call.
// Exactly one of *FetchThreatListUpdatesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *FetchThreatListUpdatesResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ThreatListUpdatesFetchCall) Do(opts ...googleapi.CallOption) (*FetchThreatListUpdatesResponse, error) {
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
	ret := &FetchThreatListUpdatesResponse{
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
	//   "description": "Fetches the most recent threat list updates. A client can request updates\nfor multiple lists at once.",
	//   "flatPath": "v4/threatListUpdates:fetch",
	//   "httpMethod": "POST",
	//   "id": "safebrowsing.threatListUpdates.fetch",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/threatListUpdates:fetch",
	//   "request": {
	//     "$ref": "FetchThreatListUpdatesRequest"
	//   },
	//   "response": {
	//     "$ref": "FetchThreatListUpdatesResponse"
	//   }
	// }

}

// method id "safebrowsing.threatLists.list":

type ThreatListsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the Safe Browsing threat lists available for download.
func (r *ThreatListsService) List() *ThreatListsListCall {
	c := &ThreatListsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ThreatListsListCall) Fields(s ...googleapi.Field) *ThreatListsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ThreatListsListCall) IfNoneMatch(entityTag string) *ThreatListsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ThreatListsListCall) Context(ctx context.Context) *ThreatListsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ThreatListsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ThreatListsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/threatLists")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.threatLists.list" call.
// Exactly one of *ListThreatListsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListThreatListsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ThreatListsListCall) Do(opts ...googleapi.CallOption) (*ListThreatListsResponse, error) {
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
	ret := &ListThreatListsResponse{
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
	//   "description": "Lists the Safe Browsing threat lists available for download.",
	//   "flatPath": "v4/threatLists",
	//   "httpMethod": "GET",
	//   "id": "safebrowsing.threatLists.list",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/threatLists",
	//   "response": {
	//     "$ref": "ListThreatListsResponse"
	//   }
	// }

}

// method id "safebrowsing.threatMatches.find":

type ThreatMatchesFindCall struct {
	s                        *Service
	findthreatmatchesrequest *FindThreatMatchesRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Find: Finds the threat entries that match the Safe Browsing lists.
func (r *ThreatMatchesService) Find(findthreatmatchesrequest *FindThreatMatchesRequest) *ThreatMatchesFindCall {
	c := &ThreatMatchesFindCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.findthreatmatchesrequest = findthreatmatchesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ThreatMatchesFindCall) Fields(s ...googleapi.Field) *ThreatMatchesFindCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ThreatMatchesFindCall) Context(ctx context.Context) *ThreatMatchesFindCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ThreatMatchesFindCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ThreatMatchesFindCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.findthreatmatchesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/threatMatches:find")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "safebrowsing.threatMatches.find" call.
// Exactly one of *FindThreatMatchesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *FindThreatMatchesResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ThreatMatchesFindCall) Do(opts ...googleapi.CallOption) (*FindThreatMatchesResponse, error) {
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
	ret := &FindThreatMatchesResponse{
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
	//   "description": "Finds the threat entries that match the Safe Browsing lists.",
	//   "flatPath": "v4/threatMatches:find",
	//   "httpMethod": "POST",
	//   "id": "safebrowsing.threatMatches.find",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/threatMatches:find",
	//   "request": {
	//     "$ref": "FindThreatMatchesRequest"
	//   },
	//   "response": {
	//     "$ref": "FindThreatMatchesResponse"
	//   }
	// }

}
