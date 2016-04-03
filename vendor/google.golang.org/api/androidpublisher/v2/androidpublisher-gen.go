// Package androidpublisher provides access to the Google Play Developer API.
//
// See https://developers.google.com/android-publisher
//
// Usage example:
//
//   import "google.golang.org/api/androidpublisher/v2"
//   ...
//   androidpublisherService, err := androidpublisher.New(oauthHttpClient)
package androidpublisher // import "google.golang.org/api/androidpublisher/v2"

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

const apiId = "androidpublisher:v2"
const apiName = "androidpublisher"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/androidpublisher/v2/applications/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Play Developer account
	AndroidpublisherScope = "https://www.googleapis.com/auth/androidpublisher"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Edits = NewEditsService(s)
	s.Entitlements = NewEntitlementsService(s)
	s.Inappproducts = NewInappproductsService(s)
	s.Purchases = NewPurchasesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Edits *EditsService

	Entitlements *EntitlementsService

	Inappproducts *InappproductsService

	Purchases *PurchasesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewEditsService(s *Service) *EditsService {
	rs := &EditsService{s: s}
	rs.Apklistings = NewEditsApklistingsService(s)
	rs.Apks = NewEditsApksService(s)
	rs.Details = NewEditsDetailsService(s)
	rs.Expansionfiles = NewEditsExpansionfilesService(s)
	rs.Images = NewEditsImagesService(s)
	rs.Listings = NewEditsListingsService(s)
	rs.Testers = NewEditsTestersService(s)
	rs.Tracks = NewEditsTracksService(s)
	return rs
}

type EditsService struct {
	s *Service

	Apklistings *EditsApklistingsService

	Apks *EditsApksService

	Details *EditsDetailsService

	Expansionfiles *EditsExpansionfilesService

	Images *EditsImagesService

	Listings *EditsListingsService

	Testers *EditsTestersService

	Tracks *EditsTracksService
}

func NewEditsApklistingsService(s *Service) *EditsApklistingsService {
	rs := &EditsApklistingsService{s: s}
	return rs
}

type EditsApklistingsService struct {
	s *Service
}

func NewEditsApksService(s *Service) *EditsApksService {
	rs := &EditsApksService{s: s}
	return rs
}

type EditsApksService struct {
	s *Service
}

func NewEditsDetailsService(s *Service) *EditsDetailsService {
	rs := &EditsDetailsService{s: s}
	return rs
}

type EditsDetailsService struct {
	s *Service
}

func NewEditsExpansionfilesService(s *Service) *EditsExpansionfilesService {
	rs := &EditsExpansionfilesService{s: s}
	return rs
}

type EditsExpansionfilesService struct {
	s *Service
}

func NewEditsImagesService(s *Service) *EditsImagesService {
	rs := &EditsImagesService{s: s}
	return rs
}

type EditsImagesService struct {
	s *Service
}

func NewEditsListingsService(s *Service) *EditsListingsService {
	rs := &EditsListingsService{s: s}
	return rs
}

type EditsListingsService struct {
	s *Service
}

func NewEditsTestersService(s *Service) *EditsTestersService {
	rs := &EditsTestersService{s: s}
	return rs
}

type EditsTestersService struct {
	s *Service
}

func NewEditsTracksService(s *Service) *EditsTracksService {
	rs := &EditsTracksService{s: s}
	return rs
}

type EditsTracksService struct {
	s *Service
}

func NewEntitlementsService(s *Service) *EntitlementsService {
	rs := &EntitlementsService{s: s}
	return rs
}

type EntitlementsService struct {
	s *Service
}

func NewInappproductsService(s *Service) *InappproductsService {
	rs := &InappproductsService{s: s}
	return rs
}

type InappproductsService struct {
	s *Service
}

func NewPurchasesService(s *Service) *PurchasesService {
	rs := &PurchasesService{s: s}
	rs.Products = NewPurchasesProductsService(s)
	rs.Subscriptions = NewPurchasesSubscriptionsService(s)
	return rs
}

type PurchasesService struct {
	s *Service

	Products *PurchasesProductsService

	Subscriptions *PurchasesSubscriptionsService
}

func NewPurchasesProductsService(s *Service) *PurchasesProductsService {
	rs := &PurchasesProductsService{s: s}
	return rs
}

type PurchasesProductsService struct {
	s *Service
}

func NewPurchasesSubscriptionsService(s *Service) *PurchasesSubscriptionsService {
	rs := &PurchasesSubscriptionsService{s: s}
	return rs
}

type PurchasesSubscriptionsService struct {
	s *Service
}

type Apk struct {
	// Binary: Information about the binary payload of this APK.
	Binary *ApkBinary `json:"binary,omitempty"`

	// VersionCode: The version code of the APK, as specified in the APK's
	// manifest file.
	VersionCode int64 `json:"versionCode,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Binary") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Apk) MarshalJSON() ([]byte, error) {
	type noMethod Apk
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ApkBinary: Represents the binary payload of an APK.
type ApkBinary struct {
	// Sha1: A sha1 hash of the APK payload, encoded as a hex string and
	// matching the output of the sha1sum command.
	Sha1 string `json:"sha1,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Sha1") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApkBinary) MarshalJSON() ([]byte, error) {
	type noMethod ApkBinary
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ApkListing struct {
	// Language: The language code, in BCP 47 format (eg "en-US").
	Language string `json:"language,omitempty"`

	// RecentChanges: Describe what's new in your APK.
	RecentChanges string `json:"recentChanges,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Language") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApkListing) MarshalJSON() ([]byte, error) {
	type noMethod ApkListing
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ApkListingsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#apkListingsListResponse".
	Kind string `json:"kind,omitempty"`

	Listings []*ApkListing `json:"listings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApkListingsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ApkListingsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ApksAddExternallyHostedRequest struct {
	// ExternallyHostedApk: The definition of the externally-hosted APK and
	// where it is located.
	ExternallyHostedApk *ExternallyHostedApk `json:"externallyHostedApk,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExternallyHostedApk")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApksAddExternallyHostedRequest) MarshalJSON() ([]byte, error) {
	type noMethod ApksAddExternallyHostedRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ApksAddExternallyHostedResponse struct {
	// ExternallyHostedApk: The definition of the externally-hosted APK and
	// where it is located.
	ExternallyHostedApk *ExternallyHostedApk `json:"externallyHostedApk,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExternallyHostedApk")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApksAddExternallyHostedResponse) MarshalJSON() ([]byte, error) {
	type noMethod ApksAddExternallyHostedResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ApksListResponse struct {
	Apks []*Apk `json:"apks,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#apksListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Apks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApksListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ApksListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AppDetails struct {
	// ContactEmail: The user-visible support email for this app.
	ContactEmail string `json:"contactEmail,omitempty"`

	// ContactPhone: The user-visible support telephone number for this app.
	ContactPhone string `json:"contactPhone,omitempty"`

	// ContactWebsite: The user-visible website for this app.
	ContactWebsite string `json:"contactWebsite,omitempty"`

	// DefaultLanguage: Default language code, in BCP 47 format (eg
	// "en-US").
	DefaultLanguage string `json:"defaultLanguage,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ContactEmail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AppDetails) MarshalJSON() ([]byte, error) {
	type noMethod AppDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AppEdit: Represents an edit of an app. An edit allows clients to make
// multiple changes before committing them in one operation.
type AppEdit struct {
	// ExpiryTimeSeconds: The time at which the edit will expire and will be
	// no longer valid for use in any subsequent API calls (encoded as
	// seconds since the Epoch).
	ExpiryTimeSeconds string `json:"expiryTimeSeconds,omitempty"`

	// Id: The ID of the edit that can be used in subsequent API calls.
	Id string `json:"id,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExpiryTimeSeconds")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AppEdit) MarshalJSON() ([]byte, error) {
	type noMethod AppEdit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Entitlement: An Entitlement resource indicates a user's current
// entitlement to an inapp item or subscription.
type Entitlement struct {
	// Kind: This kind represents an entitlement object in the
	// androidpublisher service.
	Kind string `json:"kind,omitempty"`

	// ProductId: The SKU of the product.
	ProductId string `json:"productId,omitempty"`

	// ProductType: The type of the inapp product. Possible values are:
	// - In-app item: "inapp"
	// - Subscription: "subs"
	ProductType string `json:"productType,omitempty"`

	// Token: The token which can be verified using the subscriptions or
	// products API.
	Token string `json:"token,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Entitlement) MarshalJSON() ([]byte, error) {
	type noMethod Entitlement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type EntitlementsListResponse struct {
	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	Resources []*Entitlement `json:"resources,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "PageInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EntitlementsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod EntitlementsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ExpansionFile struct {
	// FileSize: If set this field indicates that this APK has an Expansion
	// File uploaded to it: this APK does not reference another APK's
	// Expansion File. The field's value is the size of the uploaded
	// Expansion File in bytes.
	FileSize int64 `json:"fileSize,omitempty,string"`

	// ReferencesVersion: If set this APK's Expansion File references
	// another APK's Expansion File. The file_size field will not be set.
	ReferencesVersion int64 `json:"referencesVersion,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "FileSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ExpansionFile) MarshalJSON() ([]byte, error) {
	type noMethod ExpansionFile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ExpansionFilesUploadResponse struct {
	ExpansionFile *ExpansionFile `json:"expansionFile,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExpansionFile") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ExpansionFilesUploadResponse) MarshalJSON() ([]byte, error) {
	type noMethod ExpansionFilesUploadResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ExternallyHostedApk: Defines an APK available for this application
// that is hosted externally and not uploaded to Google Play. This
// function is only available to enterprises who are using Google Play
// for Work, and whos application is restricted to the enterprise
// private channel
type ExternallyHostedApk struct {
	// ApplicationLabel: The application label.
	ApplicationLabel string `json:"applicationLabel,omitempty"`

	// CertificateBase64s: A certificate (or array of certificates if a
	// certificate-chain is used) used to signed this APK, represented as a
	// base64 encoded byte array.
	CertificateBase64s []string `json:"certificateBase64s,omitempty"`

	// ExternallyHostedUrl: The URL at which the APK is hosted. This must be
	// an https URL.
	ExternallyHostedUrl string `json:"externallyHostedUrl,omitempty"`

	// FileSha1Base64: The SHA1 checksum of this APK, represented as a
	// base64 encoded byte array.
	FileSha1Base64 string `json:"fileSha1Base64,omitempty"`

	// FileSha256Base64: The SHA256 checksum of this APK, represented as a
	// base64 encoded byte array.
	FileSha256Base64 string `json:"fileSha256Base64,omitempty"`

	// FileSize: The file size in bytes of this APK.
	FileSize int64 `json:"fileSize,omitempty,string"`

	// IconBase64: The icon image from the APK, as a base64 encoded byte
	// array.
	IconBase64 string `json:"iconBase64,omitempty"`

	// MaximumSdk: The maximum SDK supported by this APK (optional).
	MaximumSdk int64 `json:"maximumSdk,omitempty"`

	// MinimumSdk: The minimum SDK targeted by this APK.
	MinimumSdk int64 `json:"minimumSdk,omitempty"`

	// NativeCodes: The native code environments supported by this APK
	// (optional).
	NativeCodes []string `json:"nativeCodes,omitempty"`

	// PackageName: The package name.
	PackageName string `json:"packageName,omitempty"`

	// UsesFeatures: The features required by this APK (optional).
	UsesFeatures []string `json:"usesFeatures,omitempty"`

	// UsesPermissions: The permissions requested by this APK.
	UsesPermissions []*ExternallyHostedApkUsesPermission `json:"usesPermissions,omitempty"`

	// VersionCode: The version code of this APK.
	VersionCode int64 `json:"versionCode,omitempty"`

	// VersionName: The version name of this APK.
	VersionName string `json:"versionName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApplicationLabel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ExternallyHostedApk) MarshalJSON() ([]byte, error) {
	type noMethod ExternallyHostedApk
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ExternallyHostedApkUsesPermission: A permission used by this APK.
type ExternallyHostedApkUsesPermission struct {
	// MaxSdkVersion: Optionally, the maximum SDK version for which the
	// permission is required.
	MaxSdkVersion int64 `json:"maxSdkVersion,omitempty"`

	// Name: The name of the permission requested.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxSdkVersion") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ExternallyHostedApkUsesPermission) MarshalJSON() ([]byte, error) {
	type noMethod ExternallyHostedApkUsesPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Image struct {
	// Id: A unique id representing this image.
	Id string `json:"id,omitempty"`

	// Sha1: A sha1 hash of the image that was uploaded.
	Sha1 string `json:"sha1,omitempty"`

	// Url: A URL that will serve a preview of the image.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Image) MarshalJSON() ([]byte, error) {
	type noMethod Image
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ImagesDeleteAllResponse struct {
	Deleted []*Image `json:"deleted,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ImagesDeleteAllResponse) MarshalJSON() ([]byte, error) {
	type noMethod ImagesDeleteAllResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ImagesListResponse struct {
	Images []*Image `json:"images,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Images") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ImagesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ImagesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ImagesUploadResponse struct {
	Image *Image `json:"image,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Image") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ImagesUploadResponse) MarshalJSON() ([]byte, error) {
	type noMethod ImagesUploadResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InAppProduct struct {
	// DefaultLanguage: The default language of the localized data, as
	// defined by BCP 47. e.g. "en-US", "en-GB".
	DefaultLanguage string `json:"defaultLanguage,omitempty"`

	// DefaultPrice: Default price cannot be zero. In-app products can never
	// be free. Default price is always in the developer's Checkout merchant
	// currency.
	DefaultPrice *Price `json:"defaultPrice,omitempty"`

	// Listings: List of localized title and description data.
	Listings map[string]InAppProductListing `json:"listings,omitempty"`

	// PackageName: The package name of the parent app.
	PackageName string `json:"packageName,omitempty"`

	// Prices: Prices per buyer region. None of these prices should be zero.
	// In-app products can never be free.
	Prices map[string]Price `json:"prices,omitempty"`

	// PurchaseType: Purchase type enum value. Unmodifiable after creation.
	PurchaseType string `json:"purchaseType,omitempty"`

	// Season: Definition of a season for a seasonal subscription. Can be
	// defined only for yearly subscriptions.
	Season *Season `json:"season,omitempty"`

	// Sku: The stock-keeping-unit (SKU) of the product, unique within an
	// app.
	Sku string `json:"sku,omitempty"`

	Status string `json:"status,omitempty"`

	// SubscriptionPeriod: Subscription period, specified in ISO 8601
	// format. Acceptable values are "P1W" (one week), "P1M" (one month) and
	// "P1Y" (one year).
	SubscriptionPeriod string `json:"subscriptionPeriod,omitempty"`

	// TrialPeriod: Trial period, specified in ISO 8601 format. Acceptable
	// values are anything between "P7D" (seven days) and "P999D" (999
	// days). Seasonal subscriptions cannot have a trial period.
	TrialPeriod string `json:"trialPeriod,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DefaultLanguage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InAppProduct) MarshalJSON() ([]byte, error) {
	type noMethod InAppProduct
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InAppProductListing struct {
	Description string `json:"description,omitempty"`

	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InAppProductListing) MarshalJSON() ([]byte, error) {
	type noMethod InAppProductListing
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsBatchRequest struct {
	Entrys []*InappproductsBatchRequestEntry `json:"entrys,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entrys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsBatchRequestEntry struct {
	BatchId int64 `json:"batchId,omitempty"`

	Inappproductsinsertrequest *InappproductsInsertRequest `json:"inappproductsinsertrequest,omitempty"`

	Inappproductsupdaterequest *InappproductsUpdateRequest `json:"inappproductsupdaterequest,omitempty"`

	MethodName string `json:"methodName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsBatchResponse struct {
	Entrys []*InappproductsBatchResponseEntry `json:"entrys,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#inappproductsBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entrys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsBatchResponseEntry struct {
	BatchId int64 `json:"batchId,omitempty"`

	Inappproductsinsertresponse *InappproductsInsertResponse `json:"inappproductsinsertresponse,omitempty"`

	Inappproductsupdateresponse *InappproductsUpdateResponse `json:"inappproductsupdateresponse,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsInsertRequest struct {
	Inappproduct *InAppProduct `json:"inappproduct,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Inappproduct") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsInsertRequest) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsInsertRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsInsertResponse struct {
	Inappproduct *InAppProduct `json:"inappproduct,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Inappproduct") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsInsertResponse) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsInsertResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsListResponse struct {
	Inappproduct []*InAppProduct `json:"inappproduct,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#inappproductsListResponse".
	Kind string `json:"kind,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Inappproduct") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsUpdateRequest struct {
	Inappproduct *InAppProduct `json:"inappproduct,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Inappproduct") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsUpdateRequest) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsUpdateRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InappproductsUpdateResponse struct {
	Inappproduct *InAppProduct `json:"inappproduct,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Inappproduct") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InappproductsUpdateResponse) MarshalJSON() ([]byte, error) {
	type noMethod InappproductsUpdateResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Listing struct {
	// FullDescription: Full description of the app; this may be up to 4000
	// characters in length.
	FullDescription string `json:"fullDescription,omitempty"`

	// Language: Language localization code (for example, "de-AT" for
	// Austrian German).
	Language string `json:"language,omitempty"`

	// ShortDescription: Short description of the app (previously known as
	// promo text); this may be up to 80 characters in length.
	ShortDescription string `json:"shortDescription,omitempty"`

	// Title: App's localized title.
	Title string `json:"title,omitempty"`

	// Video: URL of a promotional YouTube video for the app.
	Video string `json:"video,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "FullDescription") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Listing) MarshalJSON() ([]byte, error) {
	type noMethod Listing
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListingsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#listingsListResponse".
	Kind string `json:"kind,omitempty"`

	Listings []*Listing `json:"listings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListingsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListingsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type MonthDay struct {
	// Day: Day of a month, value in [1, 31] range. Valid range depends on
	// the specified month.
	Day int64 `json:"day,omitempty"`

	// Month: Month of a year. e.g. 1 = JAN, 2 = FEB etc.
	Month int64 `json:"month,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MonthDay) MarshalJSON() ([]byte, error) {
	type noMethod MonthDay
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PageInfo struct {
	ResultPerPage int64 `json:"resultPerPage,omitempty"`

	StartIndex int64 `json:"startIndex,omitempty"`

	TotalResults int64 `json:"totalResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResultPerPage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PageInfo) MarshalJSON() ([]byte, error) {
	type noMethod PageInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Price struct {
	// Currency: 3 letter Currency code, as defined by ISO 4217.
	Currency string `json:"currency,omitempty"`

	// PriceMicros: The price in millionths of the currency base unit
	// represented as a string.
	PriceMicros string `json:"priceMicros,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Currency") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Price) MarshalJSON() ([]byte, error) {
	type noMethod Price
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ProductPurchase: A ProductPurchase resource indicates the status of a
// user's inapp product purchase.
type ProductPurchase struct {
	// ConsumptionState: The consumption state of the inapp product.
	// Possible values are:
	// - Yet to be consumed
	// - Consumed
	ConsumptionState int64 `json:"consumptionState,omitempty"`

	// DeveloperPayload: A developer-specified string that contains
	// supplemental information about an order.
	DeveloperPayload string `json:"developerPayload,omitempty"`

	// Kind: This kind represents an inappPurchase object in the
	// androidpublisher service.
	Kind string `json:"kind,omitempty"`

	// PurchaseState: The purchase state of the order. Possible values are:
	//
	// - Purchased
	// - Cancelled
	PurchaseState int64 `json:"purchaseState,omitempty"`

	// PurchaseTimeMillis: The time the product was purchased, in
	// milliseconds since the epoch (Jan 1, 1970).
	PurchaseTimeMillis int64 `json:"purchaseTimeMillis,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ConsumptionState") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ProductPurchase) MarshalJSON() ([]byte, error) {
	type noMethod ProductPurchase
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Season struct {
	// End: Inclusive end date of the recurrence period.
	End *MonthDay `json:"end,omitempty"`

	// Start: Inclusive start date of the recurrence period.
	Start *MonthDay `json:"start,omitempty"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Season) MarshalJSON() ([]byte, error) {
	type noMethod Season
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionDeferralInfo: A SubscriptionDeferralInfo contains the
// data needed to defer a subscription purchase to a future expiry time.
type SubscriptionDeferralInfo struct {
	// DesiredExpiryTimeMillis: The desired next expiry time for the
	// subscription in milliseconds since Epoch. The given time must be
	// after the current expiry time for the subscription.
	DesiredExpiryTimeMillis int64 `json:"desiredExpiryTimeMillis,omitempty,string"`

	// ExpectedExpiryTimeMillis: The expected expiry time for the
	// subscription. If the current expiry time for the subscription is not
	// the value specified here, the deferral will not occur.
	ExpectedExpiryTimeMillis int64 `json:"expectedExpiryTimeMillis,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "DesiredExpiryTimeMillis") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionDeferralInfo) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionDeferralInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionPurchase: A SubscriptionPurchase resource indicates the
// status of a user's subscription purchase.
type SubscriptionPurchase struct {
	// AutoRenewing: Whether the subscription will automatically be renewed
	// when it reaches its current expiry time.
	AutoRenewing bool `json:"autoRenewing,omitempty"`

	// ExpiryTimeMillis: Time at which the subscription will expire, in
	// milliseconds since Epoch.
	ExpiryTimeMillis int64 `json:"expiryTimeMillis,omitempty,string"`

	// Kind: This kind represents a subscriptionPurchase object in the
	// androidpublisher service.
	Kind string `json:"kind,omitempty"`

	// StartTimeMillis: Time at which the subscription was granted, in
	// milliseconds since Epoch.
	StartTimeMillis int64 `json:"startTimeMillis,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AutoRenewing") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionPurchase) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionPurchase
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type SubscriptionPurchasesDeferRequest struct {
	// DeferralInfo: The information about the new desired expiry time for
	// the subscription.
	DeferralInfo *SubscriptionDeferralInfo `json:"deferralInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeferralInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionPurchasesDeferRequest) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionPurchasesDeferRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type SubscriptionPurchasesDeferResponse struct {
	// NewExpiryTimeMillis: The new expiry time for the subscription in
	// milliseconds since the Epoch.
	NewExpiryTimeMillis int64 `json:"newExpiryTimeMillis,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NewExpiryTimeMillis")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionPurchasesDeferResponse) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionPurchasesDeferResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Testers struct {
	GoogleGroups []string `json:"googleGroups,omitempty"`

	GooglePlusCommunities []string `json:"googlePlusCommunities,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "GoogleGroups") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Testers) MarshalJSON() ([]byte, error) {
	type noMethod Testers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TokenPagination struct {
	NextPageToken string `json:"nextPageToken,omitempty"`

	PreviousPageToken string `json:"previousPageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TokenPagination) MarshalJSON() ([]byte, error) {
	type noMethod TokenPagination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Track struct {
	Track string `json:"track,omitempty"`

	UserFraction float64 `json:"userFraction,omitempty"`

	VersionCodes []int64 `json:"versionCodes,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Track") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Track) MarshalJSON() ([]byte, error) {
	type noMethod Track
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TracksListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidpublisher#tracksListResponse".
	Kind string `json:"kind,omitempty"`

	Tracks []*Track `json:"tracks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TracksListResponse) MarshalJSON() ([]byte, error) {
	type noMethod TracksListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "androidpublisher.edits.commit":

type EditsCommitCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Commit: Commits/applies the changes made in this edit back to the
// app.
func (r *EditsService) Commit(packageNameid string, editId string) *EditsCommitCall {
	c := &EditsCommitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsCommitCall) QuotaUser(quotaUser string) *EditsCommitCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsCommitCall) UserIP(userIP string) *EditsCommitCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsCommitCall) Fields(s ...googleapi.Field) *EditsCommitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsCommitCall) Context(ctx context.Context) *EditsCommitCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsCommitCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}:commit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.commit" call.
// Exactly one of *AppEdit or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *AppEdit.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsCommitCall) Do() (*AppEdit, error) {
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
	ret := &AppEdit{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Commits/applies the changes made in this edit back to the app.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.commit",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}:commit",
	//   "response": {
	//     "$ref": "AppEdit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.delete":

type EditsDeleteCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Delete: Deletes an edit for an app. Creating a new edit will
// automatically delete any of your previous edits so this method need
// only be called if you want to preemptively abandon an edit.
func (r *EditsService) Delete(packageNameid string, editId string) *EditsDeleteCall {
	c := &EditsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsDeleteCall) QuotaUser(quotaUser string) *EditsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsDeleteCall) UserIP(userIP string) *EditsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsDeleteCall) Fields(s ...googleapi.Field) *EditsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsDeleteCall) Context(ctx context.Context) *EditsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.delete" call.
func (c *EditsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes an edit for an app. Creating a new edit will automatically delete any of your previous edits so this method need only be called if you want to preemptively abandon an edit.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.delete",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.get":

type EditsGetCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Returns information about the edit specified. Calls will fail if
// the edit is no long active (e.g. has been deleted, superseded or
// expired).
func (r *EditsService) Get(packageNameid string, editId string) *EditsGetCall {
	c := &EditsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsGetCall) QuotaUser(quotaUser string) *EditsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsGetCall) UserIP(userIP string) *EditsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsGetCall) Fields(s ...googleapi.Field) *EditsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsGetCall) IfNoneMatch(entityTag string) *EditsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsGetCall) Context(ctx context.Context) *EditsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.get" call.
// Exactly one of *AppEdit or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *AppEdit.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsGetCall) Do() (*AppEdit, error) {
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
	ret := &AppEdit{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns information about the edit specified. Calls will fail if the edit is no long active (e.g. has been deleted, superseded or expired).",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}",
	//   "response": {
	//     "$ref": "AppEdit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.insert":

type EditsInsertCall struct {
	s             *Service
	packageNameid string
	appedit       *AppEdit
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Insert: Creates a new edit for an app, populated with the app's
// current state.
func (r *EditsService) Insert(packageNameid string, appedit *AppEdit) *EditsInsertCall {
	c := &EditsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.appedit = appedit
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsInsertCall) QuotaUser(quotaUser string) *EditsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsInsertCall) UserIP(userIP string) *EditsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsInsertCall) Fields(s ...googleapi.Field) *EditsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsInsertCall) Context(ctx context.Context) *EditsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.appedit)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.insert" call.
// Exactly one of *AppEdit or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *AppEdit.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsInsertCall) Do() (*AppEdit, error) {
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
	ret := &AppEdit{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new edit for an app, populated with the app's current state.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.insert",
	//   "parameterOrder": [
	//     "packageName"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits",
	//   "request": {
	//     "$ref": "AppEdit"
	//   },
	//   "response": {
	//     "$ref": "AppEdit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.validate":

type EditsValidateCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Validate: Checks that the edit can be successfully committed. The
// edit's changes are not applied to the live app.
func (r *EditsService) Validate(packageNameid string, editId string) *EditsValidateCall {
	c := &EditsValidateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsValidateCall) QuotaUser(quotaUser string) *EditsValidateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsValidateCall) UserIP(userIP string) *EditsValidateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsValidateCall) Fields(s ...googleapi.Field) *EditsValidateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsValidateCall) Context(ctx context.Context) *EditsValidateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsValidateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}:validate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.validate" call.
// Exactly one of *AppEdit or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *AppEdit.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsValidateCall) Do() (*AppEdit, error) {
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
	ret := &AppEdit{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Checks that the edit can be successfully committed. The edit's changes are not applied to the live app.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.validate",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}:validate",
	//   "response": {
	//     "$ref": "AppEdit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.delete":

type EditsApklistingsDeleteCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	language       string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Delete: Deletes the APK-specific localized listing for a specified
// APK and language code.
func (r *EditsApklistingsService) Delete(packageNameid string, editId string, apkVersionCode int64, language string) *EditsApklistingsDeleteCall {
	c := &EditsApklistingsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.language = language
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsDeleteCall) QuotaUser(quotaUser string) *EditsApklistingsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsDeleteCall) UserIP(userIP string) *EditsApklistingsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsDeleteCall) Fields(s ...googleapi.Field) *EditsApklistingsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsDeleteCall) Context(ctx context.Context) *EditsApklistingsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
		"language":       c.language,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.delete" call.
func (c *EditsApklistingsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the APK-specific localized listing for a specified APK and language code.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.apklistings.delete",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "language"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the APK-specific localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.deleteall":

type EditsApklistingsDeleteallCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Deleteall: Deletes all the APK-specific localized listings for a
// specified APK.
func (r *EditsApklistingsService) Deleteall(packageNameid string, editId string, apkVersionCode int64) *EditsApklistingsDeleteallCall {
	c := &EditsApklistingsDeleteallCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsDeleteallCall) QuotaUser(quotaUser string) *EditsApklistingsDeleteallCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsDeleteallCall) UserIP(userIP string) *EditsApklistingsDeleteallCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsDeleteallCall) Fields(s ...googleapi.Field) *EditsApklistingsDeleteallCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsDeleteallCall) Context(ctx context.Context) *EditsApklistingsDeleteallCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsDeleteallCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.deleteall" call.
func (c *EditsApklistingsDeleteallCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes all the APK-specific localized listings for a specified APK.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.apklistings.deleteall",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.get":

type EditsApklistingsGetCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	language       string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
}

// Get: Fetches the APK-specific localized listing for a specified APK
// and language code.
func (r *EditsApklistingsService) Get(packageNameid string, editId string, apkVersionCode int64, language string) *EditsApklistingsGetCall {
	c := &EditsApklistingsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.language = language
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsGetCall) QuotaUser(quotaUser string) *EditsApklistingsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsGetCall) UserIP(userIP string) *EditsApklistingsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsGetCall) Fields(s ...googleapi.Field) *EditsApklistingsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsApklistingsGetCall) IfNoneMatch(entityTag string) *EditsApklistingsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsGetCall) Context(ctx context.Context) *EditsApklistingsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
		"language":       c.language,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.get" call.
// Exactly one of *ApkListing or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ApkListing.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsApklistingsGetCall) Do() (*ApkListing, error) {
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
	ret := &ApkListing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Fetches the APK-specific localized listing for a specified APK and language code.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.apklistings.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "language"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the APK-specific localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}",
	//   "response": {
	//     "$ref": "ApkListing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.list":

type EditsApklistingsListCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
}

// List: Lists all the APK-specific localized listings for a specified
// APK.
func (r *EditsApklistingsService) List(packageNameid string, editId string, apkVersionCode int64) *EditsApklistingsListCall {
	c := &EditsApklistingsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsListCall) QuotaUser(quotaUser string) *EditsApklistingsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsListCall) UserIP(userIP string) *EditsApklistingsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsListCall) Fields(s ...googleapi.Field) *EditsApklistingsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsApklistingsListCall) IfNoneMatch(entityTag string) *EditsApklistingsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsListCall) Context(ctx context.Context) *EditsApklistingsListCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.list" call.
// Exactly one of *ApkListingsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ApkListingsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsApklistingsListCall) Do() (*ApkListingsListResponse, error) {
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
	ret := &ApkListingsListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all the APK-specific localized listings for a specified APK.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.apklistings.list",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings",
	//   "response": {
	//     "$ref": "ApkListingsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.patch":

type EditsApklistingsPatchCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	language       string
	apklisting     *ApkListing
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Patch: Updates or creates the APK-specific localized listing for a
// specified APK and language code. This method supports patch
// semantics.
func (r *EditsApklistingsService) Patch(packageNameid string, editId string, apkVersionCode int64, language string, apklisting *ApkListing) *EditsApklistingsPatchCall {
	c := &EditsApklistingsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.language = language
	c.apklisting = apklisting
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsPatchCall) QuotaUser(quotaUser string) *EditsApklistingsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsPatchCall) UserIP(userIP string) *EditsApklistingsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsPatchCall) Fields(s ...googleapi.Field) *EditsApklistingsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsPatchCall) Context(ctx context.Context) *EditsApklistingsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.apklisting)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
		"language":       c.language,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.patch" call.
// Exactly one of *ApkListing or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ApkListing.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsApklistingsPatchCall) Do() (*ApkListing, error) {
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
	ret := &ApkListing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates or creates the APK-specific localized listing for a specified APK and language code. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.apklistings.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "language"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the APK-specific localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}",
	//   "request": {
	//     "$ref": "ApkListing"
	//   },
	//   "response": {
	//     "$ref": "ApkListing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apklistings.update":

type EditsApklistingsUpdateCall struct {
	s              *Service
	packageNameid  string
	editId         string
	apkVersionCode int64
	language       string
	apklisting     *ApkListing
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Update: Updates or creates the APK-specific localized listing for a
// specified APK and language code.
func (r *EditsApklistingsService) Update(packageNameid string, editId string, apkVersionCode int64, language string, apklisting *ApkListing) *EditsApklistingsUpdateCall {
	c := &EditsApklistingsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.language = language
	c.apklisting = apklisting
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApklistingsUpdateCall) QuotaUser(quotaUser string) *EditsApklistingsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApklistingsUpdateCall) UserIP(userIP string) *EditsApklistingsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApklistingsUpdateCall) Fields(s ...googleapi.Field) *EditsApklistingsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApklistingsUpdateCall) Context(ctx context.Context) *EditsApklistingsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApklistingsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.apklisting)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageNameid,
		"editId":         c.editId,
		"apkVersionCode": strconv.FormatInt(c.apkVersionCode, 10),
		"language":       c.language,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apklistings.update" call.
// Exactly one of *ApkListing or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ApkListing.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsApklistingsUpdateCall) Do() (*ApkListing, error) {
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
	ret := &ApkListing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates or creates the APK-specific localized listing for a specified APK and language code.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.apklistings.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "language"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The APK version code whose APK-specific listings should be read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the APK-specific localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/listings/{language}",
	//   "request": {
	//     "$ref": "ApkListing"
	//   },
	//   "response": {
	//     "$ref": "ApkListing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apks.addexternallyhosted":

type EditsApksAddexternallyhostedCall struct {
	s                              *Service
	packageNameid                  string
	editId                         string
	apksaddexternallyhostedrequest *ApksAddExternallyHostedRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
}

// Addexternallyhosted: Creates a new APK without uploading the APK
// itself to Google Play, instead hosting the APK at a specified URL.
// This function is only available to enterprises using Google Play for
// Work whose application is configured to restrict distribution to the
// enterprise domain.
func (r *EditsApksService) Addexternallyhosted(packageNameid string, editId string, apksaddexternallyhostedrequest *ApksAddExternallyHostedRequest) *EditsApksAddexternallyhostedCall {
	c := &EditsApksAddexternallyhostedCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apksaddexternallyhostedrequest = apksaddexternallyhostedrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApksAddexternallyhostedCall) QuotaUser(quotaUser string) *EditsApksAddexternallyhostedCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApksAddexternallyhostedCall) UserIP(userIP string) *EditsApksAddexternallyhostedCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApksAddexternallyhostedCall) Fields(s ...googleapi.Field) *EditsApksAddexternallyhostedCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApksAddexternallyhostedCall) Context(ctx context.Context) *EditsApksAddexternallyhostedCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApksAddexternallyhostedCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.apksaddexternallyhostedrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/externallyHosted")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apks.addexternallyhosted" call.
// Exactly one of *ApksAddExternallyHostedResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ApksAddExternallyHostedResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsApksAddexternallyhostedCall) Do() (*ApksAddExternallyHostedResponse, error) {
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
	ret := &ApksAddExternallyHostedResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new APK without uploading the APK itself to Google Play, instead hosting the APK at a specified URL. This function is only available to enterprises using Google Play for Work whose application is configured to restrict distribution to the enterprise domain.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.apks.addexternallyhosted",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/externallyHosted",
	//   "request": {
	//     "$ref": "ApksAddExternallyHostedRequest"
	//   },
	//   "response": {
	//     "$ref": "ApksAddExternallyHostedResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apks.list":

type EditsApksListCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List:
func (r *EditsApksService) List(packageNameid string, editId string) *EditsApksListCall {
	c := &EditsApksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApksListCall) QuotaUser(quotaUser string) *EditsApksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApksListCall) UserIP(userIP string) *EditsApksListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApksListCall) Fields(s ...googleapi.Field) *EditsApksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsApksListCall) IfNoneMatch(entityTag string) *EditsApksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsApksListCall) Context(ctx context.Context) *EditsApksListCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apks.list" call.
// Exactly one of *ApksListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ApksListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsApksListCall) Do() (*ApksListResponse, error) {
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
	ret := &ApksListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.apks.list",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks",
	//   "response": {
	//     "$ref": "ApksListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.apks.upload":

type EditsApksUploadCall struct {
	s                *Service
	packageNameid    string
	editId           string
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Upload:
func (r *EditsApksService) Upload(packageNameid string, editId string) *EditsApksUploadCall {
	c := &EditsApksUploadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsApksUploadCall) QuotaUser(quotaUser string) *EditsApksUploadCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsApksUploadCall) UserIP(userIP string) *EditsApksUploadCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *EditsApksUploadCall) Media(r io.Reader) *EditsApksUploadCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *EditsApksUploadCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *EditsApksUploadCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *EditsApksUploadCall) ProgressUpdater(pu googleapi.ProgressUpdater) *EditsApksUploadCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsApksUploadCall) Fields(s ...googleapi.Field) *EditsApksUploadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *EditsApksUploadCall) Context(ctx context.Context) *EditsApksUploadCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsApksUploadCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.apks.upload" call.
// Exactly one of *Apk or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Apk.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *EditsApksUploadCall) Do() (*Apk, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Apk{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.apks.upload",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream",
	//       "application/vnd.android.package-archive"
	//     ],
	//     "maxSize": "1GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/apks"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/apks"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks",
	//   "response": {
	//     "$ref": "Apk"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "androidpublisher.edits.details.get":

type EditsDetailsGetCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Fetches app details for this edit. This includes the default
// language and developer support contact information.
func (r *EditsDetailsService) Get(packageNameid string, editId string) *EditsDetailsGetCall {
	c := &EditsDetailsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsDetailsGetCall) QuotaUser(quotaUser string) *EditsDetailsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsDetailsGetCall) UserIP(userIP string) *EditsDetailsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsDetailsGetCall) Fields(s ...googleapi.Field) *EditsDetailsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsDetailsGetCall) IfNoneMatch(entityTag string) *EditsDetailsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsDetailsGetCall) Context(ctx context.Context) *EditsDetailsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsDetailsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/details")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.details.get" call.
// Exactly one of *AppDetails or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AppDetails.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsDetailsGetCall) Do() (*AppDetails, error) {
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
	ret := &AppDetails{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Fetches app details for this edit. This includes the default language and developer support contact information.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.details.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/details",
	//   "response": {
	//     "$ref": "AppDetails"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.details.patch":

type EditsDetailsPatchCall struct {
	s             *Service
	packageNameid string
	editId        string
	appdetails    *AppDetails
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch: Updates app details for this edit. This method supports patch
// semantics.
func (r *EditsDetailsService) Patch(packageNameid string, editId string, appdetails *AppDetails) *EditsDetailsPatchCall {
	c := &EditsDetailsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.appdetails = appdetails
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsDetailsPatchCall) QuotaUser(quotaUser string) *EditsDetailsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsDetailsPatchCall) UserIP(userIP string) *EditsDetailsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsDetailsPatchCall) Fields(s ...googleapi.Field) *EditsDetailsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsDetailsPatchCall) Context(ctx context.Context) *EditsDetailsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsDetailsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.appdetails)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/details")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.details.patch" call.
// Exactly one of *AppDetails or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AppDetails.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsDetailsPatchCall) Do() (*AppDetails, error) {
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
	ret := &AppDetails{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates app details for this edit. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.details.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/details",
	//   "request": {
	//     "$ref": "AppDetails"
	//   },
	//   "response": {
	//     "$ref": "AppDetails"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.details.update":

type EditsDetailsUpdateCall struct {
	s             *Service
	packageNameid string
	editId        string
	appdetails    *AppDetails
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Updates app details for this edit.
func (r *EditsDetailsService) Update(packageNameid string, editId string, appdetails *AppDetails) *EditsDetailsUpdateCall {
	c := &EditsDetailsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.appdetails = appdetails
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsDetailsUpdateCall) QuotaUser(quotaUser string) *EditsDetailsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsDetailsUpdateCall) UserIP(userIP string) *EditsDetailsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsDetailsUpdateCall) Fields(s ...googleapi.Field) *EditsDetailsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsDetailsUpdateCall) Context(ctx context.Context) *EditsDetailsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsDetailsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.appdetails)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/details")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.details.update" call.
// Exactly one of *AppDetails or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AppDetails.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EditsDetailsUpdateCall) Do() (*AppDetails, error) {
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
	ret := &AppDetails{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates app details for this edit.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.details.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/details",
	//   "request": {
	//     "$ref": "AppDetails"
	//   },
	//   "response": {
	//     "$ref": "AppDetails"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.expansionfiles.get":

type EditsExpansionfilesGetCall struct {
	s                 *Service
	packageNameid     string
	editId            string
	apkVersionCode    int64
	expansionFileType string
	urlParams_        gensupport.URLParams
	ifNoneMatch_      string
	ctx_              context.Context
}

// Get: Fetches the Expansion File configuration for the APK specified.
func (r *EditsExpansionfilesService) Get(packageNameid string, editId string, apkVersionCode int64, expansionFileType string) *EditsExpansionfilesGetCall {
	c := &EditsExpansionfilesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.expansionFileType = expansionFileType
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsExpansionfilesGetCall) QuotaUser(quotaUser string) *EditsExpansionfilesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsExpansionfilesGetCall) UserIP(userIP string) *EditsExpansionfilesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsExpansionfilesGetCall) Fields(s ...googleapi.Field) *EditsExpansionfilesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsExpansionfilesGetCall) IfNoneMatch(entityTag string) *EditsExpansionfilesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsExpansionfilesGetCall) Context(ctx context.Context) *EditsExpansionfilesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsExpansionfilesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":       c.packageNameid,
		"editId":            c.editId,
		"apkVersionCode":    strconv.FormatInt(c.apkVersionCode, 10),
		"expansionFileType": c.expansionFileType,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.expansionfiles.get" call.
// Exactly one of *ExpansionFile or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ExpansionFile.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsExpansionfilesGetCall) Do() (*ExpansionFile, error) {
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
	ret := &ExpansionFile{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Fetches the Expansion File configuration for the APK specified.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.expansionfiles.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "expansionFileType"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The version code of the APK whose Expansion File configuration is being read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "expansionFileType": {
	//       "enum": [
	//         "main",
	//         "patch"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}",
	//   "response": {
	//     "$ref": "ExpansionFile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.expansionfiles.patch":

type EditsExpansionfilesPatchCall struct {
	s                 *Service
	packageNameid     string
	editId            string
	apkVersionCode    int64
	expansionFileType string
	expansionfile     *ExpansionFile
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// Patch: Updates the APK's Expansion File configuration to reference
// another APK's Expansion Files. To add a new Expansion File use the
// Upload method. This method supports patch semantics.
func (r *EditsExpansionfilesService) Patch(packageNameid string, editId string, apkVersionCode int64, expansionFileType string, expansionfile *ExpansionFile) *EditsExpansionfilesPatchCall {
	c := &EditsExpansionfilesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.expansionFileType = expansionFileType
	c.expansionfile = expansionfile
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsExpansionfilesPatchCall) QuotaUser(quotaUser string) *EditsExpansionfilesPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsExpansionfilesPatchCall) UserIP(userIP string) *EditsExpansionfilesPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsExpansionfilesPatchCall) Fields(s ...googleapi.Field) *EditsExpansionfilesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsExpansionfilesPatchCall) Context(ctx context.Context) *EditsExpansionfilesPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsExpansionfilesPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.expansionfile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":       c.packageNameid,
		"editId":            c.editId,
		"apkVersionCode":    strconv.FormatInt(c.apkVersionCode, 10),
		"expansionFileType": c.expansionFileType,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.expansionfiles.patch" call.
// Exactly one of *ExpansionFile or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ExpansionFile.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsExpansionfilesPatchCall) Do() (*ExpansionFile, error) {
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
	ret := &ExpansionFile{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the APK's Expansion File configuration to reference another APK's Expansion Files. To add a new Expansion File use the Upload method. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.expansionfiles.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "expansionFileType"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The version code of the APK whose Expansion File configuration is being read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "expansionFileType": {
	//       "enum": [
	//         "main",
	//         "patch"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}",
	//   "request": {
	//     "$ref": "ExpansionFile"
	//   },
	//   "response": {
	//     "$ref": "ExpansionFile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.expansionfiles.update":

type EditsExpansionfilesUpdateCall struct {
	s                 *Service
	packageNameid     string
	editId            string
	apkVersionCode    int64
	expansionFileType string
	expansionfile     *ExpansionFile
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// Update: Updates the APK's Expansion File configuration to reference
// another APK's Expansion Files. To add a new Expansion File use the
// Upload method.
func (r *EditsExpansionfilesService) Update(packageNameid string, editId string, apkVersionCode int64, expansionFileType string, expansionfile *ExpansionFile) *EditsExpansionfilesUpdateCall {
	c := &EditsExpansionfilesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.expansionFileType = expansionFileType
	c.expansionfile = expansionfile
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsExpansionfilesUpdateCall) QuotaUser(quotaUser string) *EditsExpansionfilesUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsExpansionfilesUpdateCall) UserIP(userIP string) *EditsExpansionfilesUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsExpansionfilesUpdateCall) Fields(s ...googleapi.Field) *EditsExpansionfilesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsExpansionfilesUpdateCall) Context(ctx context.Context) *EditsExpansionfilesUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsExpansionfilesUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.expansionfile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":       c.packageNameid,
		"editId":            c.editId,
		"apkVersionCode":    strconv.FormatInt(c.apkVersionCode, 10),
		"expansionFileType": c.expansionFileType,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.expansionfiles.update" call.
// Exactly one of *ExpansionFile or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ExpansionFile.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsExpansionfilesUpdateCall) Do() (*ExpansionFile, error) {
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
	ret := &ExpansionFile{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the APK's Expansion File configuration to reference another APK's Expansion Files. To add a new Expansion File use the Upload method.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.expansionfiles.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "expansionFileType"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The version code of the APK whose Expansion File configuration is being read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "expansionFileType": {
	//       "enum": [
	//         "main",
	//         "patch"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}",
	//   "request": {
	//     "$ref": "ExpansionFile"
	//   },
	//   "response": {
	//     "$ref": "ExpansionFile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.expansionfiles.upload":

type EditsExpansionfilesUploadCall struct {
	s                 *Service
	packageNameid     string
	editId            string
	apkVersionCode    int64
	expansionFileType string
	urlParams_        gensupport.URLParams
	media_            io.Reader
	resumable_        googleapi.SizeReaderAt
	mediaType_        string
	protocol_         string
	progressUpdater_  googleapi.ProgressUpdater
	ctx_              context.Context
}

// Upload: Uploads and attaches a new Expansion File to the APK
// specified.
func (r *EditsExpansionfilesService) Upload(packageNameid string, editId string, apkVersionCode int64, expansionFileType string) *EditsExpansionfilesUploadCall {
	c := &EditsExpansionfilesUploadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.apkVersionCode = apkVersionCode
	c.expansionFileType = expansionFileType
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsExpansionfilesUploadCall) QuotaUser(quotaUser string) *EditsExpansionfilesUploadCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsExpansionfilesUploadCall) UserIP(userIP string) *EditsExpansionfilesUploadCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *EditsExpansionfilesUploadCall) Media(r io.Reader) *EditsExpansionfilesUploadCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *EditsExpansionfilesUploadCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *EditsExpansionfilesUploadCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *EditsExpansionfilesUploadCall) ProgressUpdater(pu googleapi.ProgressUpdater) *EditsExpansionfilesUploadCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsExpansionfilesUploadCall) Fields(s ...googleapi.Field) *EditsExpansionfilesUploadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *EditsExpansionfilesUploadCall) Context(ctx context.Context) *EditsExpansionfilesUploadCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsExpansionfilesUploadCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":       c.packageNameid,
		"editId":            c.editId,
		"apkVersionCode":    strconv.FormatInt(c.apkVersionCode, 10),
		"expansionFileType": c.expansionFileType,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.expansionfiles.upload" call.
// Exactly one of *ExpansionFilesUploadResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ExpansionFilesUploadResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsExpansionfilesUploadCall) Do() (*ExpansionFilesUploadResponse, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &ExpansionFilesUploadResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads and attaches a new Expansion File to the APK specified.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.expansionfiles.upload",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream"
	//     ],
	//     "maxSize": "2048MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "apkVersionCode",
	//     "expansionFileType"
	//   ],
	//   "parameters": {
	//     "apkVersionCode": {
	//       "description": "The version code of the APK whose Expansion File configuration is being read or modified.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "expansionFileType": {
	//       "enum": [
	//         "main",
	//         "patch"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/apks/{apkVersionCode}/expansionFiles/{expansionFileType}",
	//   "response": {
	//     "$ref": "ExpansionFilesUploadResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "androidpublisher.edits.images.delete":

type EditsImagesDeleteCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	imageType     string
	imageId       string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Delete: Deletes the image (specified by id) from the edit.
func (r *EditsImagesService) Delete(packageNameid string, editId string, language string, imageType string, imageId string) *EditsImagesDeleteCall {
	c := &EditsImagesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.imageType = imageType
	c.imageId = imageId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsImagesDeleteCall) QuotaUser(quotaUser string) *EditsImagesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsImagesDeleteCall) UserIP(userIP string) *EditsImagesDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsImagesDeleteCall) Fields(s ...googleapi.Field) *EditsImagesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsImagesDeleteCall) Context(ctx context.Context) *EditsImagesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsImagesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}/{imageType}/{imageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
		"imageType":   c.imageType,
		"imageId":     c.imageId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.images.delete" call.
func (c *EditsImagesDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the image (specified by id) from the edit.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.images.delete",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language",
	//     "imageType",
	//     "imageId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "imageId": {
	//       "description": "Unique identifier an image within the set of images attached to this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "imageType": {
	//       "enum": [
	//         "featureGraphic",
	//         "icon",
	//         "phoneScreenshots",
	//         "promoGraphic",
	//         "sevenInchScreenshots",
	//         "tenInchScreenshots",
	//         "tvBanner",
	//         "tvScreenshots",
	//         "wearScreenshots"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing whose images are to read or modified. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}/{imageType}/{imageId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.images.deleteall":

type EditsImagesDeleteallCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	imageType     string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Deleteall: Deletes all images for the specified language and image
// type.
func (r *EditsImagesService) Deleteall(packageNameid string, editId string, language string, imageType string) *EditsImagesDeleteallCall {
	c := &EditsImagesDeleteallCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.imageType = imageType
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsImagesDeleteallCall) QuotaUser(quotaUser string) *EditsImagesDeleteallCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsImagesDeleteallCall) UserIP(userIP string) *EditsImagesDeleteallCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsImagesDeleteallCall) Fields(s ...googleapi.Field) *EditsImagesDeleteallCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsImagesDeleteallCall) Context(ctx context.Context) *EditsImagesDeleteallCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsImagesDeleteallCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}/{imageType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
		"imageType":   c.imageType,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.images.deleteall" call.
// Exactly one of *ImagesDeleteAllResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ImagesDeleteAllResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsImagesDeleteallCall) Do() (*ImagesDeleteAllResponse, error) {
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
	ret := &ImagesDeleteAllResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes all images for the specified language and image type.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.images.deleteall",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language",
	//     "imageType"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "imageType": {
	//       "enum": [
	//         "featureGraphic",
	//         "icon",
	//         "phoneScreenshots",
	//         "promoGraphic",
	//         "sevenInchScreenshots",
	//         "tenInchScreenshots",
	//         "tvBanner",
	//         "tvScreenshots",
	//         "wearScreenshots"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing whose images are to read or modified. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}/{imageType}",
	//   "response": {
	//     "$ref": "ImagesDeleteAllResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.images.list":

type EditsImagesListCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	imageType     string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: Lists all images for the specified language and image type.
func (r *EditsImagesService) List(packageNameid string, editId string, language string, imageType string) *EditsImagesListCall {
	c := &EditsImagesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.imageType = imageType
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsImagesListCall) QuotaUser(quotaUser string) *EditsImagesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsImagesListCall) UserIP(userIP string) *EditsImagesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsImagesListCall) Fields(s ...googleapi.Field) *EditsImagesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsImagesListCall) IfNoneMatch(entityTag string) *EditsImagesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsImagesListCall) Context(ctx context.Context) *EditsImagesListCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsImagesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}/{imageType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
		"imageType":   c.imageType,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.images.list" call.
// Exactly one of *ImagesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ImagesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsImagesListCall) Do() (*ImagesListResponse, error) {
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
	ret := &ImagesListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all images for the specified language and image type.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.images.list",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language",
	//     "imageType"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "imageType": {
	//       "enum": [
	//         "featureGraphic",
	//         "icon",
	//         "phoneScreenshots",
	//         "promoGraphic",
	//         "sevenInchScreenshots",
	//         "tenInchScreenshots",
	//         "tvBanner",
	//         "tvScreenshots",
	//         "wearScreenshots"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing whose images are to read or modified. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}/{imageType}",
	//   "response": {
	//     "$ref": "ImagesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.images.upload":

type EditsImagesUploadCall struct {
	s                *Service
	packageNameid    string
	editId           string
	language         string
	imageType        string
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Upload: Uploads a new image and adds it to the list of images for the
// specified language and image type.
func (r *EditsImagesService) Upload(packageNameid string, editId string, language string, imageType string) *EditsImagesUploadCall {
	c := &EditsImagesUploadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.imageType = imageType
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsImagesUploadCall) QuotaUser(quotaUser string) *EditsImagesUploadCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsImagesUploadCall) UserIP(userIP string) *EditsImagesUploadCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *EditsImagesUploadCall) Media(r io.Reader) *EditsImagesUploadCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *EditsImagesUploadCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *EditsImagesUploadCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *EditsImagesUploadCall) ProgressUpdater(pu googleapi.ProgressUpdater) *EditsImagesUploadCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsImagesUploadCall) Fields(s ...googleapi.Field) *EditsImagesUploadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *EditsImagesUploadCall) Context(ctx context.Context) *EditsImagesUploadCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsImagesUploadCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}/{imageType}")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
		"imageType":   c.imageType,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.images.upload" call.
// Exactly one of *ImagesUploadResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ImagesUploadResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsImagesUploadCall) Do() (*ImagesUploadResponse, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &ImagesUploadResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads a new image and adds it to the list of images for the specified language and image type.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.edits.images.upload",
	//   "mediaUpload": {
	//     "accept": [
	//       "image/*"
	//     ],
	//     "maxSize": "15MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/listings/{language}/{imageType}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/androidpublisher/v2/applications/{packageName}/edits/{editId}/listings/{language}/{imageType}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language",
	//     "imageType"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "imageType": {
	//       "enum": [
	//         "featureGraphic",
	//         "icon",
	//         "phoneScreenshots",
	//         "promoGraphic",
	//         "sevenInchScreenshots",
	//         "tenInchScreenshots",
	//         "tvBanner",
	//         "tvScreenshots",
	//         "wearScreenshots"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing whose images are to read or modified. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}/{imageType}",
	//   "response": {
	//     "$ref": "ImagesUploadResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "androidpublisher.edits.listings.delete":

type EditsListingsDeleteCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Delete: Deletes the specified localized store listing from an edit.
func (r *EditsListingsService) Delete(packageNameid string, editId string, language string) *EditsListingsDeleteCall {
	c := &EditsListingsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsDeleteCall) QuotaUser(quotaUser string) *EditsListingsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsDeleteCall) UserIP(userIP string) *EditsListingsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsDeleteCall) Fields(s ...googleapi.Field) *EditsListingsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsDeleteCall) Context(ctx context.Context) *EditsListingsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.delete" call.
func (c *EditsListingsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified localized store listing from an edit.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.listings.delete",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.listings.deleteall":

type EditsListingsDeleteallCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Deleteall: Deletes all localized listings from an edit.
func (r *EditsListingsService) Deleteall(packageNameid string, editId string) *EditsListingsDeleteallCall {
	c := &EditsListingsDeleteallCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsDeleteallCall) QuotaUser(quotaUser string) *EditsListingsDeleteallCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsDeleteallCall) UserIP(userIP string) *EditsListingsDeleteallCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsDeleteallCall) Fields(s ...googleapi.Field) *EditsListingsDeleteallCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsDeleteallCall) Context(ctx context.Context) *EditsListingsDeleteallCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsDeleteallCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.deleteall" call.
func (c *EditsListingsDeleteallCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes all localized listings from an edit.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.edits.listings.deleteall",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.listings.get":

type EditsListingsGetCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Fetches information about a localized store listing.
func (r *EditsListingsService) Get(packageNameid string, editId string, language string) *EditsListingsGetCall {
	c := &EditsListingsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsGetCall) QuotaUser(quotaUser string) *EditsListingsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsGetCall) UserIP(userIP string) *EditsListingsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsGetCall) Fields(s ...googleapi.Field) *EditsListingsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsListingsGetCall) IfNoneMatch(entityTag string) *EditsListingsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsGetCall) Context(ctx context.Context) *EditsListingsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.get" call.
// Exactly one of *Listing or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Listing.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsListingsGetCall) Do() (*Listing, error) {
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
	ret := &Listing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Fetches information about a localized store listing.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.listings.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}",
	//   "response": {
	//     "$ref": "Listing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.listings.list":

type EditsListingsListCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: Returns all of the localized store listings attached to this
// edit.
func (r *EditsListingsService) List(packageNameid string, editId string) *EditsListingsListCall {
	c := &EditsListingsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsListCall) QuotaUser(quotaUser string) *EditsListingsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsListCall) UserIP(userIP string) *EditsListingsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsListCall) Fields(s ...googleapi.Field) *EditsListingsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsListingsListCall) IfNoneMatch(entityTag string) *EditsListingsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsListCall) Context(ctx context.Context) *EditsListingsListCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.list" call.
// Exactly one of *ListingsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListingsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsListingsListCall) Do() (*ListingsListResponse, error) {
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
	ret := &ListingsListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns all of the localized store listings attached to this edit.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.listings.list",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings",
	//   "response": {
	//     "$ref": "ListingsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.listings.patch":

type EditsListingsPatchCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	listing       *Listing
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch: Creates or updates a localized store listing. This method
// supports patch semantics.
func (r *EditsListingsService) Patch(packageNameid string, editId string, language string, listing *Listing) *EditsListingsPatchCall {
	c := &EditsListingsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.listing = listing
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsPatchCall) QuotaUser(quotaUser string) *EditsListingsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsPatchCall) UserIP(userIP string) *EditsListingsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsPatchCall) Fields(s ...googleapi.Field) *EditsListingsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsPatchCall) Context(ctx context.Context) *EditsListingsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.listing)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.patch" call.
// Exactly one of *Listing or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Listing.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsListingsPatchCall) Do() (*Listing, error) {
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
	ret := &Listing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates or updates a localized store listing. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.listings.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}",
	//   "request": {
	//     "$ref": "Listing"
	//   },
	//   "response": {
	//     "$ref": "Listing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.listings.update":

type EditsListingsUpdateCall struct {
	s             *Service
	packageNameid string
	editId        string
	language      string
	listing       *Listing
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Creates or updates a localized store listing.
func (r *EditsListingsService) Update(packageNameid string, editId string, language string, listing *Listing) *EditsListingsUpdateCall {
	c := &EditsListingsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.language = language
	c.listing = listing
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsListingsUpdateCall) QuotaUser(quotaUser string) *EditsListingsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsListingsUpdateCall) UserIP(userIP string) *EditsListingsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsListingsUpdateCall) Fields(s ...googleapi.Field) *EditsListingsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsListingsUpdateCall) Context(ctx context.Context) *EditsListingsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsListingsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.listing)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/listings/{language}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"language":    c.language,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.listings.update" call.
// Exactly one of *Listing or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Listing.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsListingsUpdateCall) Do() (*Listing, error) {
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
	ret := &Listing{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates or updates a localized store listing.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.listings.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "language"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The language code (a BCP-47 language tag) of the localized listing to read or modify. For example, to select Austrian German, pass \"de-AT\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/listings/{language}",
	//   "request": {
	//     "$ref": "Listing"
	//   },
	//   "response": {
	//     "$ref": "Listing"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.testers.get":

type EditsTestersGetCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get:
func (r *EditsTestersService) Get(packageNameid string, editId string, track string) *EditsTestersGetCall {
	c := &EditsTestersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTestersGetCall) QuotaUser(quotaUser string) *EditsTestersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTestersGetCall) UserIP(userIP string) *EditsTestersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTestersGetCall) Fields(s ...googleapi.Field) *EditsTestersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsTestersGetCall) IfNoneMatch(entityTag string) *EditsTestersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTestersGetCall) Context(ctx context.Context) *EditsTestersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTestersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/testers/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.testers.get" call.
// Exactly one of *Testers or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Testers.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTestersGetCall) Do() (*Testers, error) {
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
	ret := &Testers{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.testers.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/testers/{track}",
	//   "response": {
	//     "$ref": "Testers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.testers.patch":

type EditsTestersPatchCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	testers       *Testers
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch:
func (r *EditsTestersService) Patch(packageNameid string, editId string, track string, testers *Testers) *EditsTestersPatchCall {
	c := &EditsTestersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	c.testers = testers
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTestersPatchCall) QuotaUser(quotaUser string) *EditsTestersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTestersPatchCall) UserIP(userIP string) *EditsTestersPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTestersPatchCall) Fields(s ...googleapi.Field) *EditsTestersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTestersPatchCall) Context(ctx context.Context) *EditsTestersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTestersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testers)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/testers/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.testers.patch" call.
// Exactly one of *Testers or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Testers.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTestersPatchCall) Do() (*Testers, error) {
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
	ret := &Testers{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.testers.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/testers/{track}",
	//   "request": {
	//     "$ref": "Testers"
	//   },
	//   "response": {
	//     "$ref": "Testers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.testers.update":

type EditsTestersUpdateCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	testers       *Testers
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update:
func (r *EditsTestersService) Update(packageNameid string, editId string, track string, testers *Testers) *EditsTestersUpdateCall {
	c := &EditsTestersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	c.testers = testers
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTestersUpdateCall) QuotaUser(quotaUser string) *EditsTestersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTestersUpdateCall) UserIP(userIP string) *EditsTestersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTestersUpdateCall) Fields(s ...googleapi.Field) *EditsTestersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTestersUpdateCall) Context(ctx context.Context) *EditsTestersUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTestersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testers)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/testers/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.testers.update" call.
// Exactly one of *Testers or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Testers.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTestersUpdateCall) Do() (*Testers, error) {
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
	ret := &Testers{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.testers.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/testers/{track}",
	//   "request": {
	//     "$ref": "Testers"
	//   },
	//   "response": {
	//     "$ref": "Testers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.tracks.get":

type EditsTracksGetCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Fetches the track configuration for the specified track type.
// Includes the APK version codes that are in this track.
func (r *EditsTracksService) Get(packageNameid string, editId string, track string) *EditsTracksGetCall {
	c := &EditsTracksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTracksGetCall) QuotaUser(quotaUser string) *EditsTracksGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTracksGetCall) UserIP(userIP string) *EditsTracksGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTracksGetCall) Fields(s ...googleapi.Field) *EditsTracksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsTracksGetCall) IfNoneMatch(entityTag string) *EditsTracksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTracksGetCall) Context(ctx context.Context) *EditsTracksGetCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTracksGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/tracks/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.tracks.get" call.
// Exactly one of *Track or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Track.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTracksGetCall) Do() (*Track, error) {
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
	ret := &Track{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Fetches the track configuration for the specified track type. Includes the APK version codes that are in this track.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.tracks.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "description": "The track type to read or modify.",
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/tracks/{track}",
	//   "response": {
	//     "$ref": "Track"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.tracks.list":

type EditsTracksListCall struct {
	s             *Service
	packageNameid string
	editId        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: Lists all the track configurations for this edit.
func (r *EditsTracksService) List(packageNameid string, editId string) *EditsTracksListCall {
	c := &EditsTracksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTracksListCall) QuotaUser(quotaUser string) *EditsTracksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTracksListCall) UserIP(userIP string) *EditsTracksListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTracksListCall) Fields(s ...googleapi.Field) *EditsTracksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EditsTracksListCall) IfNoneMatch(entityTag string) *EditsTracksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTracksListCall) Context(ctx context.Context) *EditsTracksListCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTracksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/tracks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.tracks.list" call.
// Exactly one of *TracksListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TracksListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EditsTracksListCall) Do() (*TracksListResponse, error) {
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
	ret := &TracksListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all the track configurations for this edit.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.edits.tracks.list",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/tracks",
	//   "response": {
	//     "$ref": "TracksListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.tracks.patch":

type EditsTracksPatchCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	track2        *Track
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch: Updates the track configuration for the specified track type.
// When halted, the rollout track cannot be updated without adding new
// APKs, and adding new APKs will cause it to resume. This method
// supports patch semantics.
func (r *EditsTracksService) Patch(packageNameid string, editId string, track string, track2 *Track) *EditsTracksPatchCall {
	c := &EditsTracksPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	c.track2 = track2
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTracksPatchCall) QuotaUser(quotaUser string) *EditsTracksPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTracksPatchCall) UserIP(userIP string) *EditsTracksPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTracksPatchCall) Fields(s ...googleapi.Field) *EditsTracksPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTracksPatchCall) Context(ctx context.Context) *EditsTracksPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTracksPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.track2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/tracks/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.tracks.patch" call.
// Exactly one of *Track or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Track.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTracksPatchCall) Do() (*Track, error) {
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
	ret := &Track{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the track configuration for the specified track type. When halted, the rollout track cannot be updated without adding new APKs, and adding new APKs will cause it to resume. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.edits.tracks.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "description": "The track type to read or modify.",
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/tracks/{track}",
	//   "request": {
	//     "$ref": "Track"
	//   },
	//   "response": {
	//     "$ref": "Track"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.edits.tracks.update":

type EditsTracksUpdateCall struct {
	s             *Service
	packageNameid string
	editId        string
	track         string
	track2        *Track
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Updates the track configuration for the specified track type.
// When halted, the rollout track cannot be updated without adding new
// APKs, and adding new APKs will cause it to resume.
func (r *EditsTracksService) Update(packageNameid string, editId string, track string, track2 *Track) *EditsTracksUpdateCall {
	c := &EditsTracksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.editId = editId
	c.track = track
	c.track2 = track2
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EditsTracksUpdateCall) QuotaUser(quotaUser string) *EditsTracksUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EditsTracksUpdateCall) UserIP(userIP string) *EditsTracksUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EditsTracksUpdateCall) Fields(s ...googleapi.Field) *EditsTracksUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EditsTracksUpdateCall) Context(ctx context.Context) *EditsTracksUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *EditsTracksUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.track2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/edits/{editId}/tracks/{track}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"editId":      c.editId,
		"track":       c.track,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.edits.tracks.update" call.
// Exactly one of *Track or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Track.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EditsTracksUpdateCall) Do() (*Track, error) {
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
	ret := &Track{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the track configuration for the specified track type. When halted, the rollout track cannot be updated without adding new APKs, and adding new APKs will cause it to resume.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.edits.tracks.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "editId",
	//     "track"
	//   ],
	//   "parameters": {
	//     "editId": {
	//       "description": "Unique identifier for this edit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app that is being updated; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "track": {
	//       "description": "The track type to read or modify.",
	//       "enum": [
	//         "alpha",
	//         "beta",
	//         "production",
	//         "rollout"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/edits/{editId}/tracks/{track}",
	//   "request": {
	//     "$ref": "Track"
	//   },
	//   "response": {
	//     "$ref": "Track"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.entitlements.list":

type EntitlementsListCall struct {
	s            *Service
	packageName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the user's current inapp item or subscription
// entitlements
func (r *EntitlementsService) List(packageName string) *EntitlementsListCall {
	c := &EntitlementsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	return c
}

// MaxResults sets the optional parameter "maxResults":
func (c *EntitlementsListCall) MaxResults(maxResults int64) *EntitlementsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// ProductId sets the optional parameter "productId": The product id of
// the inapp product (for example, 'sku1'). This can be used to restrict
// the result set.
func (c *EntitlementsListCall) ProductId(productId string) *EntitlementsListCall {
	c.urlParams_.Set("productId", productId)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsListCall) QuotaUser(quotaUser string) *EntitlementsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartIndex sets the optional parameter "startIndex":
func (c *EntitlementsListCall) StartIndex(startIndex int64) *EntitlementsListCall {
	c.urlParams_.Set("startIndex", fmt.Sprint(startIndex))
	return c
}

// Token sets the optional parameter "token":
func (c *EntitlementsListCall) Token(token string) *EntitlementsListCall {
	c.urlParams_.Set("token", token)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsListCall) UserIP(userIP string) *EntitlementsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsListCall) Fields(s ...googleapi.Field) *EntitlementsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EntitlementsListCall) IfNoneMatch(entityTag string) *EntitlementsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsListCall) Context(ctx context.Context) *EntitlementsListCall {
	c.ctx_ = ctx
	return c
}

func (c *EntitlementsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/entitlements")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.entitlements.list" call.
// Exactly one of *EntitlementsListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *EntitlementsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EntitlementsListCall) Do() (*EntitlementsListResponse, error) {
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
	ret := &EntitlementsListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the user's current inapp item or subscription entitlements",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.entitlements.list",
	//   "parameterOrder": [
	//     "packageName"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "packageName": {
	//       "description": "The package name of the application the inapp product was sold in (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The product id of the inapp product (for example, 'sku1'). This can be used to restrict the result set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "token": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/entitlements",
	//   "response": {
	//     "$ref": "EntitlementsListResponse"
	//   }
	// }

}

// method id "androidpublisher.inappproducts.batch":

type InappproductsBatchCall struct {
	s                         *Service
	inappproductsbatchrequest *InappproductsBatchRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
}

// Batch:
func (r *InappproductsService) Batch(inappproductsbatchrequest *InappproductsBatchRequest) *InappproductsBatchCall {
	c := &InappproductsBatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.inappproductsbatchrequest = inappproductsbatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsBatchCall) QuotaUser(quotaUser string) *InappproductsBatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsBatchCall) UserIP(userIP string) *InappproductsBatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsBatchCall) Fields(s ...googleapi.Field) *InappproductsBatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsBatchCall) Context(ctx context.Context) *InappproductsBatchCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsBatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inappproductsbatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "inappproducts/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.batch" call.
// Exactly one of *InappproductsBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *InappproductsBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InappproductsBatchCall) Do() (*InappproductsBatchResponse, error) {
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
	ret := &InappproductsBatchResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.inappproducts.batch",
	//   "path": "inappproducts/batch",
	//   "request": {
	//     "$ref": "InappproductsBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "InappproductsBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.delete":

type InappproductsDeleteCall struct {
	s             *Service
	packageNameid string
	skuid         string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Delete: Delete an in-app product for an app.
func (r *InappproductsService) Delete(packageNameid string, skuid string) *InappproductsDeleteCall {
	c := &InappproductsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.skuid = skuid
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsDeleteCall) QuotaUser(quotaUser string) *InappproductsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsDeleteCall) UserIP(userIP string) *InappproductsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsDeleteCall) Fields(s ...googleapi.Field) *InappproductsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsDeleteCall) Context(ctx context.Context) *InappproductsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts/{sku}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"sku":         c.skuid,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.delete" call.
func (c *InappproductsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Delete an in-app product for an app.",
	//   "httpMethod": "DELETE",
	//   "id": "androidpublisher.inappproducts.delete",
	//   "parameterOrder": [
	//     "packageName",
	//     "sku"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "Unique identifier for the Android app with the in-app product; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sku": {
	//       "description": "Unique identifier for the in-app product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts/{sku}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.get":

type InappproductsGetCall struct {
	s            *Service
	packageName  string
	skuid        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns information about the in-app product specified.
func (r *InappproductsService) Get(packageName string, skuid string) *InappproductsGetCall {
	c := &InappproductsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.skuid = skuid
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsGetCall) QuotaUser(quotaUser string) *InappproductsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsGetCall) UserIP(userIP string) *InappproductsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsGetCall) Fields(s ...googleapi.Field) *InappproductsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InappproductsGetCall) IfNoneMatch(entityTag string) *InappproductsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsGetCall) Context(ctx context.Context) *InappproductsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts/{sku}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageName,
		"sku":         c.skuid,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.get" call.
// Exactly one of *InAppProduct or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *InAppProduct.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InappproductsGetCall) Do() (*InAppProduct, error) {
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
	ret := &InAppProduct{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns information about the in-app product specified.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.inappproducts.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "sku"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sku": {
	//       "description": "Unique identifier for the in-app product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts/{sku}",
	//   "response": {
	//     "$ref": "InAppProduct"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.insert":

type InappproductsInsertCall struct {
	s             *Service
	packageNameid string
	inappproduct  *InAppProduct
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Insert: Creates a new in-app product for an app.
func (r *InappproductsService) Insert(packageNameid string, inappproduct *InAppProduct) *InappproductsInsertCall {
	c := &InappproductsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.inappproduct = inappproduct
	return c
}

// AutoConvertMissingPrices sets the optional parameter
// "autoConvertMissingPrices": If true the prices for all regions
// targeted by the parent app that don't have a price specified for this
// in-app product will be auto converted to the target currency based on
// the default price. Defaults to false.
func (c *InappproductsInsertCall) AutoConvertMissingPrices(autoConvertMissingPrices bool) *InappproductsInsertCall {
	c.urlParams_.Set("autoConvertMissingPrices", fmt.Sprint(autoConvertMissingPrices))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsInsertCall) QuotaUser(quotaUser string) *InappproductsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsInsertCall) UserIP(userIP string) *InappproductsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsInsertCall) Fields(s ...googleapi.Field) *InappproductsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsInsertCall) Context(ctx context.Context) *InappproductsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inappproduct)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.insert" call.
// Exactly one of *InAppProduct or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *InAppProduct.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InappproductsInsertCall) Do() (*InAppProduct, error) {
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
	ret := &InAppProduct{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new in-app product for an app.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.inappproducts.insert",
	//   "parameterOrder": [
	//     "packageName"
	//   ],
	//   "parameters": {
	//     "autoConvertMissingPrices": {
	//       "description": "If true the prices for all regions targeted by the parent app that don't have a price specified for this in-app product will be auto converted to the target currency based on the default price. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts",
	//   "request": {
	//     "$ref": "InAppProduct"
	//   },
	//   "response": {
	//     "$ref": "InAppProduct"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.list":

type InappproductsListCall struct {
	s             *Service
	packageNameid string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: List all the in-app products for an Android app, both
// subscriptions and managed in-app products..
func (r *InappproductsService) List(packageNameid string) *InappproductsListCall {
	c := &InappproductsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	return c
}

// MaxResults sets the optional parameter "maxResults":
func (c *InappproductsListCall) MaxResults(maxResults int64) *InappproductsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsListCall) QuotaUser(quotaUser string) *InappproductsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartIndex sets the optional parameter "startIndex":
func (c *InappproductsListCall) StartIndex(startIndex int64) *InappproductsListCall {
	c.urlParams_.Set("startIndex", fmt.Sprint(startIndex))
	return c
}

// Token sets the optional parameter "token":
func (c *InappproductsListCall) Token(token string) *InappproductsListCall {
	c.urlParams_.Set("token", token)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsListCall) UserIP(userIP string) *InappproductsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsListCall) Fields(s ...googleapi.Field) *InappproductsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InappproductsListCall) IfNoneMatch(entityTag string) *InappproductsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsListCall) Context(ctx context.Context) *InappproductsListCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.list" call.
// Exactly one of *InappproductsListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *InappproductsListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InappproductsListCall) Do() (*InappproductsListResponse, error) {
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
	ret := &InappproductsListResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all the in-app products for an Android app, both subscriptions and managed in-app products..",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.inappproducts.list",
	//   "parameterOrder": [
	//     "packageName"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app with in-app products; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "token": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts",
	//   "response": {
	//     "$ref": "InappproductsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.patch":

type InappproductsPatchCall struct {
	s             *Service
	packageNameid string
	skuid         string
	inappproduct  *InAppProduct
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch: Updates the details of an in-app product. This method supports
// patch semantics.
func (r *InappproductsService) Patch(packageNameid string, skuid string, inappproduct *InAppProduct) *InappproductsPatchCall {
	c := &InappproductsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.skuid = skuid
	c.inappproduct = inappproduct
	return c
}

// AutoConvertMissingPrices sets the optional parameter
// "autoConvertMissingPrices": If true the prices for all regions
// targeted by the parent app that don't have a price specified for this
// in-app product will be auto converted to the target currency based on
// the default price. Defaults to false.
func (c *InappproductsPatchCall) AutoConvertMissingPrices(autoConvertMissingPrices bool) *InappproductsPatchCall {
	c.urlParams_.Set("autoConvertMissingPrices", fmt.Sprint(autoConvertMissingPrices))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsPatchCall) QuotaUser(quotaUser string) *InappproductsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsPatchCall) UserIP(userIP string) *InappproductsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsPatchCall) Fields(s ...googleapi.Field) *InappproductsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsPatchCall) Context(ctx context.Context) *InappproductsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inappproduct)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts/{sku}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"sku":         c.skuid,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.patch" call.
// Exactly one of *InAppProduct or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *InAppProduct.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InappproductsPatchCall) Do() (*InAppProduct, error) {
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
	ret := &InAppProduct{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the details of an in-app product. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidpublisher.inappproducts.patch",
	//   "parameterOrder": [
	//     "packageName",
	//     "sku"
	//   ],
	//   "parameters": {
	//     "autoConvertMissingPrices": {
	//       "description": "If true the prices for all regions targeted by the parent app that don't have a price specified for this in-app product will be auto converted to the target currency based on the default price. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app with the in-app product; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sku": {
	//       "description": "Unique identifier for the in-app product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts/{sku}",
	//   "request": {
	//     "$ref": "InAppProduct"
	//   },
	//   "response": {
	//     "$ref": "InAppProduct"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.inappproducts.update":

type InappproductsUpdateCall struct {
	s             *Service
	packageNameid string
	skuid         string
	inappproduct  *InAppProduct
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Updates the details of an in-app product.
func (r *InappproductsService) Update(packageNameid string, skuid string, inappproduct *InAppProduct) *InappproductsUpdateCall {
	c := &InappproductsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageNameid = packageNameid
	c.skuid = skuid
	c.inappproduct = inappproduct
	return c
}

// AutoConvertMissingPrices sets the optional parameter
// "autoConvertMissingPrices": If true the prices for all regions
// targeted by the parent app that don't have a price specified for this
// in-app product will be auto converted to the target currency based on
// the default price. Defaults to false.
func (c *InappproductsUpdateCall) AutoConvertMissingPrices(autoConvertMissingPrices bool) *InappproductsUpdateCall {
	c.urlParams_.Set("autoConvertMissingPrices", fmt.Sprint(autoConvertMissingPrices))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InappproductsUpdateCall) QuotaUser(quotaUser string) *InappproductsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InappproductsUpdateCall) UserIP(userIP string) *InappproductsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InappproductsUpdateCall) Fields(s ...googleapi.Field) *InappproductsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InappproductsUpdateCall) Context(ctx context.Context) *InappproductsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *InappproductsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inappproduct)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inappproducts/{sku}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageNameid,
		"sku":         c.skuid,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.inappproducts.update" call.
// Exactly one of *InAppProduct or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *InAppProduct.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InappproductsUpdateCall) Do() (*InAppProduct, error) {
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
	ret := &InAppProduct{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the details of an in-app product.",
	//   "httpMethod": "PUT",
	//   "id": "androidpublisher.inappproducts.update",
	//   "parameterOrder": [
	//     "packageName",
	//     "sku"
	//   ],
	//   "parameters": {
	//     "autoConvertMissingPrices": {
	//       "description": "If true the prices for all regions targeted by the parent app that don't have a price specified for this in-app product will be auto converted to the target currency based on the default price. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "packageName": {
	//       "description": "Unique identifier for the Android app with the in-app product; for example, \"com.spiffygame\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sku": {
	//       "description": "Unique identifier for the in-app product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inappproducts/{sku}",
	//   "request": {
	//     "$ref": "InAppProduct"
	//   },
	//   "response": {
	//     "$ref": "InAppProduct"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.products.get":

type PurchasesProductsGetCall struct {
	s            *Service
	packageName  string
	productId    string
	token        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Checks the purchase and consumption status of an inapp item.
func (r *PurchasesProductsService) Get(packageName string, productId string, token string) *PurchasesProductsGetCall {
	c := &PurchasesProductsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.productId = productId
	c.token = token
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesProductsGetCall) QuotaUser(quotaUser string) *PurchasesProductsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesProductsGetCall) UserIP(userIP string) *PurchasesProductsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesProductsGetCall) Fields(s ...googleapi.Field) *PurchasesProductsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PurchasesProductsGetCall) IfNoneMatch(entityTag string) *PurchasesProductsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesProductsGetCall) Context(ctx context.Context) *PurchasesProductsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesProductsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/products/{productId}/tokens/{token}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName": c.packageName,
		"productId":   c.productId,
		"token":       c.token,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.products.get" call.
// Exactly one of *ProductPurchase or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductPurchase.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PurchasesProductsGetCall) Do() (*ProductPurchase, error) {
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
	ret := &ProductPurchase{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Checks the purchase and consumption status of an inapp item.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.purchases.products.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "productId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application the inapp product was sold in (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The inapp product SKU (for example, 'com.some.thing.inapp1').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the inapp product was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/products/{productId}/tokens/{token}",
	//   "response": {
	//     "$ref": "ProductPurchase"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.subscriptions.cancel":

type PurchasesSubscriptionsCancelCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Cancel: Cancels a user's subscription purchase. The subscription
// remains valid until its expiration time.
func (r *PurchasesSubscriptionsService) Cancel(packageName string, subscriptionId string, token string) *PurchasesSubscriptionsCancelCall {
	c := &PurchasesSubscriptionsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesSubscriptionsCancelCall) QuotaUser(quotaUser string) *PurchasesSubscriptionsCancelCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesSubscriptionsCancelCall) UserIP(userIP string) *PurchasesSubscriptionsCancelCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesSubscriptionsCancelCall) Fields(s ...googleapi.Field) *PurchasesSubscriptionsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesSubscriptionsCancelCall) Context(ctx context.Context) *PurchasesSubscriptionsCancelCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesSubscriptionsCancelCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageName,
		"subscriptionId": c.subscriptionId,
		"token":          c.token,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.subscriptions.cancel" call.
func (c *PurchasesSubscriptionsCancelCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Cancels a user's subscription purchase. The subscription remains valid until its expiration time.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.purchases.subscriptions.cancel",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:cancel",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.subscriptions.defer":

type PurchasesSubscriptionsDeferCall struct {
	s                                 *Service
	packageName                       string
	subscriptionId                    string
	token                             string
	subscriptionpurchasesdeferrequest *SubscriptionPurchasesDeferRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
}

// Defer: Defers a user's subscription purchase until a specified future
// expiration time.
func (r *PurchasesSubscriptionsService) Defer(packageName string, subscriptionId string, token string, subscriptionpurchasesdeferrequest *SubscriptionPurchasesDeferRequest) *PurchasesSubscriptionsDeferCall {
	c := &PurchasesSubscriptionsDeferCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	c.subscriptionpurchasesdeferrequest = subscriptionpurchasesdeferrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesSubscriptionsDeferCall) QuotaUser(quotaUser string) *PurchasesSubscriptionsDeferCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesSubscriptionsDeferCall) UserIP(userIP string) *PurchasesSubscriptionsDeferCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesSubscriptionsDeferCall) Fields(s ...googleapi.Field) *PurchasesSubscriptionsDeferCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesSubscriptionsDeferCall) Context(ctx context.Context) *PurchasesSubscriptionsDeferCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesSubscriptionsDeferCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscriptionpurchasesdeferrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:defer")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageName,
		"subscriptionId": c.subscriptionId,
		"token":          c.token,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.subscriptions.defer" call.
// Exactly one of *SubscriptionPurchasesDeferResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *SubscriptionPurchasesDeferResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *PurchasesSubscriptionsDeferCall) Do() (*SubscriptionPurchasesDeferResponse, error) {
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
	ret := &SubscriptionPurchasesDeferResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Defers a user's subscription purchase until a specified future expiration time.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.purchases.subscriptions.defer",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:defer",
	//   "request": {
	//     "$ref": "SubscriptionPurchasesDeferRequest"
	//   },
	//   "response": {
	//     "$ref": "SubscriptionPurchasesDeferResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.subscriptions.get":

type PurchasesSubscriptionsGetCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
}

// Get: Checks whether a user's subscription purchase is valid and
// returns its expiry time.
func (r *PurchasesSubscriptionsService) Get(packageName string, subscriptionId string, token string) *PurchasesSubscriptionsGetCall {
	c := &PurchasesSubscriptionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesSubscriptionsGetCall) QuotaUser(quotaUser string) *PurchasesSubscriptionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesSubscriptionsGetCall) UserIP(userIP string) *PurchasesSubscriptionsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesSubscriptionsGetCall) Fields(s ...googleapi.Field) *PurchasesSubscriptionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PurchasesSubscriptionsGetCall) IfNoneMatch(entityTag string) *PurchasesSubscriptionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesSubscriptionsGetCall) Context(ctx context.Context) *PurchasesSubscriptionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesSubscriptionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageName,
		"subscriptionId": c.subscriptionId,
		"token":          c.token,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.subscriptions.get" call.
// Exactly one of *SubscriptionPurchase or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SubscriptionPurchase.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PurchasesSubscriptionsGetCall) Do() (*SubscriptionPurchase, error) {
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
	ret := &SubscriptionPurchase{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Checks whether a user's subscription purchase is valid and returns its expiry time.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.purchases.subscriptions.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}",
	//   "response": {
	//     "$ref": "SubscriptionPurchase"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.subscriptions.refund":

type PurchasesSubscriptionsRefundCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Refund: Refunds a user's subscription purchase, but the subscription
// remains valid until its expiration time and it will continue to
// recur.
func (r *PurchasesSubscriptionsService) Refund(packageName string, subscriptionId string, token string) *PurchasesSubscriptionsRefundCall {
	c := &PurchasesSubscriptionsRefundCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesSubscriptionsRefundCall) QuotaUser(quotaUser string) *PurchasesSubscriptionsRefundCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesSubscriptionsRefundCall) UserIP(userIP string) *PurchasesSubscriptionsRefundCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesSubscriptionsRefundCall) Fields(s ...googleapi.Field) *PurchasesSubscriptionsRefundCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesSubscriptionsRefundCall) Context(ctx context.Context) *PurchasesSubscriptionsRefundCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesSubscriptionsRefundCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:refund")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageName,
		"subscriptionId": c.subscriptionId,
		"token":          c.token,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.subscriptions.refund" call.
func (c *PurchasesSubscriptionsRefundCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Refunds a user's subscription purchase, but the subscription remains valid until its expiration time and it will continue to recur.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.purchases.subscriptions.refund",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:refund",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.subscriptions.revoke":

type PurchasesSubscriptionsRevokeCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Revoke: Refunds and immediately revokes a user's subscription
// purchase. Access to the subscription will be terminated immediately
// and it will stop recurring.
func (r *PurchasesSubscriptionsService) Revoke(packageName string, subscriptionId string, token string) *PurchasesSubscriptionsRevokeCall {
	c := &PurchasesSubscriptionsRevokeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PurchasesSubscriptionsRevokeCall) QuotaUser(quotaUser string) *PurchasesSubscriptionsRevokeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PurchasesSubscriptionsRevokeCall) UserIP(userIP string) *PurchasesSubscriptionsRevokeCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PurchasesSubscriptionsRevokeCall) Fields(s ...googleapi.Field) *PurchasesSubscriptionsRevokeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PurchasesSubscriptionsRevokeCall) Context(ctx context.Context) *PurchasesSubscriptionsRevokeCall {
	c.ctx_ = ctx
	return c
}

func (c *PurchasesSubscriptionsRevokeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:revoke")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"packageName":    c.packageName,
		"subscriptionId": c.subscriptionId,
		"token":          c.token,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidpublisher.purchases.subscriptions.revoke" call.
func (c *PurchasesSubscriptionsRevokeCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Refunds and immediately revokes a user's subscription purchase. Access to the subscription will be terminated immediately and it will stop recurring.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.purchases.subscriptions.revoke",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/purchases/subscriptions/{subscriptionId}/tokens/{token}:revoke",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}
