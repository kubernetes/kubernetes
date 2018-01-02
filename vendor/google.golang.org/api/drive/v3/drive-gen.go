// Package drive provides access to the Drive API.
//
// See https://developers.google.com/drive/
//
// Usage example:
//
//   import "google.golang.org/api/drive/v3"
//   ...
//   driveService, err := drive.New(oauthHttpClient)
package drive // import "google.golang.org/api/drive/v3"

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

const apiId = "drive:v3"
const apiName = "drive"
const apiVersion = "v3"
const basePath = "https://www.googleapis.com/drive/v3/"

// OAuth2 scopes used by this API.
const (
	// View and manage the files in your Google Drive
	DriveScope = "https://www.googleapis.com/auth/drive"

	// View and manage its own configuration data in your Google Drive
	DriveAppdataScope = "https://www.googleapis.com/auth/drive.appdata"

	// View and manage Google Drive files and folders that you have opened
	// or created with this app
	DriveFileScope = "https://www.googleapis.com/auth/drive.file"

	// View and manage metadata of files in your Google Drive
	DriveMetadataScope = "https://www.googleapis.com/auth/drive.metadata"

	// View metadata for files in your Google Drive
	DriveMetadataReadonlyScope = "https://www.googleapis.com/auth/drive.metadata.readonly"

	// View the photos, videos and albums in your Google Photos
	DrivePhotosReadonlyScope = "https://www.googleapis.com/auth/drive.photos.readonly"

	// View the files in your Google Drive
	DriveReadonlyScope = "https://www.googleapis.com/auth/drive.readonly"

	// Modify your Google Apps Script scripts' behavior
	DriveScriptsScope = "https://www.googleapis.com/auth/drive.scripts"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.About = NewAboutService(s)
	s.Changes = NewChangesService(s)
	s.Channels = NewChannelsService(s)
	s.Comments = NewCommentsService(s)
	s.Files = NewFilesService(s)
	s.Permissions = NewPermissionsService(s)
	s.Replies = NewRepliesService(s)
	s.Revisions = NewRevisionsService(s)
	s.Teamdrives = NewTeamdrivesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	About *AboutService

	Changes *ChangesService

	Channels *ChannelsService

	Comments *CommentsService

	Files *FilesService

	Permissions *PermissionsService

	Replies *RepliesService

	Revisions *RevisionsService

	Teamdrives *TeamdrivesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAboutService(s *Service) *AboutService {
	rs := &AboutService{s: s}
	return rs
}

type AboutService struct {
	s *Service
}

func NewChangesService(s *Service) *ChangesService {
	rs := &ChangesService{s: s}
	return rs
}

type ChangesService struct {
	s *Service
}

func NewChannelsService(s *Service) *ChannelsService {
	rs := &ChannelsService{s: s}
	return rs
}

type ChannelsService struct {
	s *Service
}

func NewCommentsService(s *Service) *CommentsService {
	rs := &CommentsService{s: s}
	return rs
}

type CommentsService struct {
	s *Service
}

func NewFilesService(s *Service) *FilesService {
	rs := &FilesService{s: s}
	return rs
}

type FilesService struct {
	s *Service
}

func NewPermissionsService(s *Service) *PermissionsService {
	rs := &PermissionsService{s: s}
	return rs
}

type PermissionsService struct {
	s *Service
}

func NewRepliesService(s *Service) *RepliesService {
	rs := &RepliesService{s: s}
	return rs
}

type RepliesService struct {
	s *Service
}

func NewRevisionsService(s *Service) *RevisionsService {
	rs := &RevisionsService{s: s}
	return rs
}

type RevisionsService struct {
	s *Service
}

func NewTeamdrivesService(s *Service) *TeamdrivesService {
	rs := &TeamdrivesService{s: s}
	return rs
}

type TeamdrivesService struct {
	s *Service
}

// About: Information about the user, the user's Drive, and system
// capabilities.
type About struct {
	// AppInstalled: Whether the user has installed the requesting app.
	AppInstalled bool `json:"appInstalled,omitempty"`

	// ExportFormats: A map of source MIME type to possible targets for all
	// supported exports.
	ExportFormats map[string][]string `json:"exportFormats,omitempty"`

	// FolderColorPalette: The currently supported folder colors as RGB hex
	// strings.
	FolderColorPalette []string `json:"folderColorPalette,omitempty"`

	// ImportFormats: A map of source MIME type to possible targets for all
	// supported imports.
	ImportFormats map[string][]string `json:"importFormats,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#about".
	Kind string `json:"kind,omitempty"`

	// MaxImportSizes: A map of maximum import sizes by MIME type, in bytes.
	MaxImportSizes map[string]string `json:"maxImportSizes,omitempty"`

	// MaxUploadSize: The maximum upload size in bytes.
	MaxUploadSize int64 `json:"maxUploadSize,omitempty,string"`

	// StorageQuota: The user's storage quota limits and usage. All fields
	// are measured in bytes.
	StorageQuota *AboutStorageQuota `json:"storageQuota,omitempty"`

	// TeamDriveThemes: A list of themes that are supported for Team Drives.
	TeamDriveThemes []*AboutTeamDriveThemes `json:"teamDriveThemes,omitempty"`

	// User: The authenticated user.
	User *User `json:"user,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AppInstalled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppInstalled") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *About) MarshalJSON() ([]byte, error) {
	type noMethod About
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AboutStorageQuota: The user's storage quota limits and usage. All
// fields are measured in bytes.
type AboutStorageQuota struct {
	// Limit: The usage limit, if applicable. This will not be present if
	// the user has unlimited storage.
	Limit int64 `json:"limit,omitempty,string"`

	// Usage: The total usage across all services.
	Usage int64 `json:"usage,omitempty,string"`

	// UsageInDrive: The usage by all files in Google Drive.
	UsageInDrive int64 `json:"usageInDrive,omitempty,string"`

	// UsageInDriveTrash: The usage by trashed files in Google Drive.
	UsageInDriveTrash int64 `json:"usageInDriveTrash,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Limit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Limit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AboutStorageQuota) MarshalJSON() ([]byte, error) {
	type noMethod AboutStorageQuota
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AboutTeamDriveThemes struct {
	// BackgroundImageLink: A link to this Team Drive theme's background
	// image.
	BackgroundImageLink string `json:"backgroundImageLink,omitempty"`

	// ColorRgb: The color of this Team Drive theme as an RGB hex string.
	ColorRgb string `json:"colorRgb,omitempty"`

	// Id: The ID of the theme.
	Id string `json:"id,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BackgroundImageLink")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BackgroundImageLink") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AboutTeamDriveThemes) MarshalJSON() ([]byte, error) {
	type noMethod AboutTeamDriveThemes
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Change: A change to a file or Team Drive.
type Change struct {
	// File: The updated state of the file. Present if the type is file and
	// the file has not been removed from this list of changes.
	File *File `json:"file,omitempty"`

	// FileId: The ID of the file which has changed.
	FileId string `json:"fileId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#change".
	Kind string `json:"kind,omitempty"`

	// Removed: Whether the file or Team Drive has been removed from this
	// list of changes, for example by deletion or loss of access.
	Removed bool `json:"removed,omitempty"`

	// TeamDrive: The updated state of the Team Drive. Present if the type
	// is teamDrive, the user is still a member of the Team Drive, and the
	// Team Drive has not been removed.
	TeamDrive *TeamDrive `json:"teamDrive,omitempty"`

	// TeamDriveId: The ID of the Team Drive associated with this change.
	TeamDriveId string `json:"teamDriveId,omitempty"`

	// Time: The time of this change (RFC 3339 date-time).
	Time string `json:"time,omitempty"`

	// Type: The type of the change. Possible values are file and teamDrive.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "File") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "File") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Change) MarshalJSON() ([]byte, error) {
	type noMethod Change
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChangeList: A list of changes for a user.
type ChangeList struct {
	// Changes: The list of changes. If nextPageToken is populated, then
	// this list may be incomplete and an additional page of results should
	// be fetched.
	Changes []*Change `json:"changes,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#changeList".
	Kind string `json:"kind,omitempty"`

	// NewStartPageToken: The starting page token for future changes. This
	// will be present only if the end of the current changes list has been
	// reached.
	NewStartPageToken string `json:"newStartPageToken,omitempty"`

	// NextPageToken: The page token for the next page of changes. This will
	// be absent if the end of the changes list has been reached. If the
	// token is rejected for any reason, it should be discarded, and
	// pagination should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Changes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Changes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChangeList) MarshalJSON() ([]byte, error) {
	type noMethod ChangeList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Channel: An notification channel used to watch for resource changes.
type Channel struct {
	// Address: The address where notifications are delivered for this
	// channel.
	Address string `json:"address,omitempty"`

	// Expiration: Date and time of notification channel expiration,
	// expressed as a Unix timestamp, in milliseconds. Optional.
	Expiration int64 `json:"expiration,omitempty,string"`

	// Id: A UUID or similar unique string that identifies this channel.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a notification channel used to watch for
	// changes to a resource. Value: the fixed string "api#channel".
	Kind string `json:"kind,omitempty"`

	// Params: Additional parameters controlling delivery channel behavior.
	// Optional.
	Params map[string]string `json:"params,omitempty"`

	// Payload: A Boolean value to indicate whether payload is wanted.
	// Optional.
	Payload bool `json:"payload,omitempty"`

	// ResourceId: An opaque ID that identifies the resource being watched
	// on this channel. Stable across different API versions.
	ResourceId string `json:"resourceId,omitempty"`

	// ResourceUri: A version-specific identifier for the watched resource.
	ResourceUri string `json:"resourceUri,omitempty"`

	// Token: An arbitrary string delivered to the target address with each
	// notification delivered over this channel. Optional.
	Token string `json:"token,omitempty"`

	// Type: The type of delivery mechanism used for this channel.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Address") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Address") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Channel) MarshalJSON() ([]byte, error) {
	type noMethod Channel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Comment: A comment on a file.
type Comment struct {
	// Anchor: A region of the document represented as a JSON string. See
	// anchor documentation for details on how to define and interpret
	// anchor properties.
	Anchor string `json:"anchor,omitempty"`

	// Author: The user who created the comment.
	Author *User `json:"author,omitempty"`

	// Content: The plain text content of the comment. This field is used
	// for setting the content, while htmlContent should be displayed.
	Content string `json:"content,omitempty"`

	// CreatedTime: The time at which the comment was created (RFC 3339
	// date-time).
	CreatedTime string `json:"createdTime,omitempty"`

	// Deleted: Whether the comment has been deleted. A deleted comment has
	// no content.
	Deleted bool `json:"deleted,omitempty"`

	// HtmlContent: The content of the comment with HTML formatting.
	HtmlContent string `json:"htmlContent,omitempty"`

	// Id: The ID of the comment.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#comment".
	Kind string `json:"kind,omitempty"`

	// ModifiedTime: The last time the comment or any of its replies was
	// modified (RFC 3339 date-time).
	ModifiedTime string `json:"modifiedTime,omitempty"`

	// QuotedFileContent: The file content to which the comment refers,
	// typically within the anchor region. For a text file, for example,
	// this would be the text at the location of the comment.
	QuotedFileContent *CommentQuotedFileContent `json:"quotedFileContent,omitempty"`

	// Replies: The full list of replies to the comment in chronological
	// order.
	Replies []*Reply `json:"replies,omitempty"`

	// Resolved: Whether the comment has been resolved by one of its
	// replies.
	Resolved bool `json:"resolved,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Anchor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Anchor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Comment) MarshalJSON() ([]byte, error) {
	type noMethod Comment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentQuotedFileContent: The file content to which the comment
// refers, typically within the anchor region. For a text file, for
// example, this would be the text at the location of the comment.
type CommentQuotedFileContent struct {
	// MimeType: The MIME type of the quoted content.
	MimeType string `json:"mimeType,omitempty"`

	// Value: The quoted content itself. This is interpreted as plain text
	// if set through the API.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MimeType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MimeType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentQuotedFileContent) MarshalJSON() ([]byte, error) {
	type noMethod CommentQuotedFileContent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentList: A list of comments on a file.
type CommentList struct {
	// Comments: The list of comments. If nextPageToken is populated, then
	// this list may be incomplete and an additional page of results should
	// be fetched.
	Comments []*Comment `json:"comments,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#commentList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of comments. This
	// will be absent if the end of the comments list has been reached. If
	// the token is rejected for any reason, it should be discarded, and
	// pagination should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Comments") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Comments") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentList) MarshalJSON() ([]byte, error) {
	type noMethod CommentList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// File: The metadata for a file.
type File struct {
	// AppProperties: A collection of arbitrary key-value pairs which are
	// private to the requesting app.
	// Entries with null values are cleared in update and copy requests.
	AppProperties map[string]string `json:"appProperties,omitempty"`

	// Capabilities: Capabilities the current user has on this file. Each
	// capability corresponds to a fine-grained action that a user may take.
	Capabilities *FileCapabilities `json:"capabilities,omitempty"`

	// ContentHints: Additional information about the content of the file.
	// These fields are never populated in responses.
	ContentHints *FileContentHints `json:"contentHints,omitempty"`

	// CreatedTime: The time at which the file was created (RFC 3339
	// date-time).
	CreatedTime string `json:"createdTime,omitempty"`

	// Description: A short description of the file.
	Description string `json:"description,omitempty"`

	// ExplicitlyTrashed: Whether the file has been explicitly trashed, as
	// opposed to recursively trashed from a parent folder.
	ExplicitlyTrashed bool `json:"explicitlyTrashed,omitempty"`

	// FileExtension: The final component of fullFileExtension. This is only
	// available for files with binary content in Drive.
	FileExtension string `json:"fileExtension,omitempty"`

	// FolderColorRgb: The color for a folder as an RGB hex string. The
	// supported colors are published in the folderColorPalette field of the
	// About resource.
	// If an unsupported color is specified, the closest color in the
	// palette will be used instead.
	FolderColorRgb string `json:"folderColorRgb,omitempty"`

	// FullFileExtension: The full file extension extracted from the name
	// field. May contain multiple concatenated extensions, such as
	// "tar.gz". This is only available for files with binary content in
	// Drive.
	// This is automatically updated when the name field changes, however it
	// is not cleared if the new name does not contain a valid extension.
	FullFileExtension string `json:"fullFileExtension,omitempty"`

	// HasAugmentedPermissions: Whether any users are granted file access
	// directly on this file. This field is only populated for Team Drive
	// files.
	HasAugmentedPermissions bool `json:"hasAugmentedPermissions,omitempty"`

	// HasThumbnail: Whether this file has a thumbnail. This does not
	// indicate whether the requesting app has access to the thumbnail. To
	// check access, look for the presence of the thumbnailLink field.
	HasThumbnail bool `json:"hasThumbnail,omitempty"`

	// HeadRevisionId: The ID of the file's head revision. This is currently
	// only available for files with binary content in Drive.
	HeadRevisionId string `json:"headRevisionId,omitempty"`

	// IconLink: A static, unauthenticated link to the file's icon.
	IconLink string `json:"iconLink,omitempty"`

	// Id: The ID of the file.
	Id string `json:"id,omitempty"`

	// ImageMediaMetadata: Additional metadata about image media, if
	// available.
	ImageMediaMetadata *FileImageMediaMetadata `json:"imageMediaMetadata,omitempty"`

	// IsAppAuthorized: Whether the file was created or opened by the
	// requesting app.
	IsAppAuthorized bool `json:"isAppAuthorized,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#file".
	Kind string `json:"kind,omitempty"`

	// LastModifyingUser: The last user to modify the file.
	LastModifyingUser *User `json:"lastModifyingUser,omitempty"`

	// Md5Checksum: The MD5 checksum for the content of the file. This is
	// only applicable to files with binary content in Drive.
	Md5Checksum string `json:"md5Checksum,omitempty"`

	// MimeType: The MIME type of the file.
	// Drive will attempt to automatically detect an appropriate value from
	// uploaded content if no value is provided. The value cannot be changed
	// unless a new revision is uploaded.
	// If a file is created with a Google Doc MIME type, the uploaded
	// content will be imported if possible. The supported import formats
	// are published in the About resource.
	MimeType string `json:"mimeType,omitempty"`

	// ModifiedByMe: Whether the file has been modified by this user.
	ModifiedByMe bool `json:"modifiedByMe,omitempty"`

	// ModifiedByMeTime: The last time the file was modified by the user
	// (RFC 3339 date-time).
	ModifiedByMeTime string `json:"modifiedByMeTime,omitempty"`

	// ModifiedTime: The last time the file was modified by anyone (RFC 3339
	// date-time).
	// Note that setting modifiedTime will also update modifiedByMeTime for
	// the user.
	ModifiedTime string `json:"modifiedTime,omitempty"`

	// Name: The name of the file. This is not necessarily unique within a
	// folder. Note that for immutable items such as the top level folders
	// of Team Drives, My Drive root folder, and Application Data folder the
	// name is constant.
	Name string `json:"name,omitempty"`

	// OriginalFilename: The original filename of the uploaded content if
	// available, or else the original value of the name field. This is only
	// available for files with binary content in Drive.
	OriginalFilename string `json:"originalFilename,omitempty"`

	// OwnedByMe: Whether the user owns the file. Not populated for Team
	// Drive files.
	OwnedByMe bool `json:"ownedByMe,omitempty"`

	// Owners: The owners of the file. Currently, only certain legacy files
	// may have more than one owner. Not populated for Team Drive files.
	Owners []*User `json:"owners,omitempty"`

	// Parents: The IDs of the parent folders which contain the file.
	// If not specified as part of a create request, the file will be placed
	// directly in the My Drive folder. Update requests must use the
	// addParents and removeParents parameters to modify the values.
	Parents []string `json:"parents,omitempty"`

	// PermissionIds: List of permission IDs for users with access to this
	// file.
	PermissionIds []string `json:"permissionIds,omitempty"`

	// Permissions: The full list of permissions for the file. This is only
	// available if the requesting user can share the file. Not populated
	// for Team Drive files.
	Permissions []*Permission `json:"permissions,omitempty"`

	// Properties: A collection of arbitrary key-value pairs which are
	// visible to all apps.
	// Entries with null values are cleared in update and copy requests.
	Properties map[string]string `json:"properties,omitempty"`

	// QuotaBytesUsed: The number of storage quota bytes used by the file.
	// This includes the head revision as well as previous revisions with
	// keepForever enabled.
	QuotaBytesUsed int64 `json:"quotaBytesUsed,omitempty,string"`

	// Shared: Whether the file has been shared. Not populated for Team
	// Drive files.
	Shared bool `json:"shared,omitempty"`

	// SharedWithMeTime: The time at which the file was shared with the
	// user, if applicable (RFC 3339 date-time).
	SharedWithMeTime string `json:"sharedWithMeTime,omitempty"`

	// SharingUser: The user who shared the file with the requesting user,
	// if applicable.
	SharingUser *User `json:"sharingUser,omitempty"`

	// Size: The size of the file's content in bytes. This is only
	// applicable to files with binary content in Drive.
	Size int64 `json:"size,omitempty,string"`

	// Spaces: The list of spaces which contain the file. The currently
	// supported values are 'drive', 'appDataFolder' and 'photos'.
	Spaces []string `json:"spaces,omitempty"`

	// Starred: Whether the user has starred the file.
	Starred bool `json:"starred,omitempty"`

	// TeamDriveId: ID of the Team Drive the file resides in.
	TeamDriveId string `json:"teamDriveId,omitempty"`

	// ThumbnailLink: A short-lived link to the file's thumbnail, if
	// available. Typically lasts on the order of hours. Only populated when
	// the requesting app can access the file's content.
	ThumbnailLink string `json:"thumbnailLink,omitempty"`

	// ThumbnailVersion: The thumbnail version for use in thumbnail cache
	// invalidation.
	ThumbnailVersion int64 `json:"thumbnailVersion,omitempty,string"`

	// Trashed: Whether the file has been trashed, either explicitly or from
	// a trashed parent folder. Only the owner may trash a file, and other
	// users cannot see files in the owner's trash.
	Trashed bool `json:"trashed,omitempty"`

	// TrashedTime: The time that the item was trashed (RFC 3339 date-time).
	// Only populated for Team Drive files.
	TrashedTime string `json:"trashedTime,omitempty"`

	// TrashingUser: If the file has been explicitly trashed, the user who
	// trashed it. Only populated for Team Drive files.
	TrashingUser *User `json:"trashingUser,omitempty"`

	// Version: A monotonically increasing version number for the file. This
	// reflects every change made to the file on the server, even those not
	// visible to the user.
	Version int64 `json:"version,omitempty,string"`

	// VideoMediaMetadata: Additional metadata about video media. This may
	// not be available immediately upon upload.
	VideoMediaMetadata *FileVideoMediaMetadata `json:"videoMediaMetadata,omitempty"`

	// ViewedByMe: Whether the file has been viewed by this user.
	ViewedByMe bool `json:"viewedByMe,omitempty"`

	// ViewedByMeTime: The last time the file was viewed by the user (RFC
	// 3339 date-time).
	ViewedByMeTime string `json:"viewedByMeTime,omitempty"`

	// ViewersCanCopyContent: Whether users with only reader or commenter
	// permission can copy the file's content. This affects copy, download,
	// and print operations.
	ViewersCanCopyContent bool `json:"viewersCanCopyContent,omitempty"`

	// WebContentLink: A link for downloading the content of the file in a
	// browser. This is only available for files with binary content in
	// Drive.
	WebContentLink string `json:"webContentLink,omitempty"`

	// WebViewLink: A link for opening the file in a relevant Google editor
	// or viewer in a browser.
	WebViewLink string `json:"webViewLink,omitempty"`

	// WritersCanShare: Whether users with only writer permission can modify
	// the file's permissions. Not populated for Team Drive files.
	WritersCanShare bool `json:"writersCanShare,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AppProperties") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppProperties") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *File) MarshalJSON() ([]byte, error) {
	type noMethod File
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FileCapabilities: Capabilities the current user has on this file.
// Each capability corresponds to a fine-grained action that a user may
// take.
type FileCapabilities struct {
	// CanAddChildren: Whether the current user can add children to this
	// folder. This is always false when the item is not a folder.
	CanAddChildren bool `json:"canAddChildren,omitempty"`

	// CanChangeViewersCanCopyContent: Whether the current user can change
	// whether viewers can copy the contents of this file.
	CanChangeViewersCanCopyContent bool `json:"canChangeViewersCanCopyContent,omitempty"`

	// CanComment: Whether the current user can comment on this file.
	CanComment bool `json:"canComment,omitempty"`

	// CanCopy: Whether the current user can copy this file. For a Team
	// Drive item, whether the current user can copy non-folder descendants
	// of this item, or this item itself if it is not a folder.
	CanCopy bool `json:"canCopy,omitempty"`

	// CanDelete: Whether the current user can delete this file.
	CanDelete bool `json:"canDelete,omitempty"`

	// CanDownload: Whether the current user can download this file.
	CanDownload bool `json:"canDownload,omitempty"`

	// CanEdit: Whether the current user can edit this file.
	CanEdit bool `json:"canEdit,omitempty"`

	// CanListChildren: Whether the current user can list the children of
	// this folder. This is always false when the item is not a folder.
	CanListChildren bool `json:"canListChildren,omitempty"`

	// CanMoveItemIntoTeamDrive: Whether the current user can move this item
	// into a Team Drive. If the item is in a Team Drive, this field is
	// equivalent to canMoveTeamDriveItem.
	CanMoveItemIntoTeamDrive bool `json:"canMoveItemIntoTeamDrive,omitempty"`

	// CanMoveTeamDriveItem: Whether the current user can move this Team
	// Drive item by changing its parent. Note that a request to change the
	// parent for this item may still fail depending on the new parent that
	// is being added. Only populated for Team Drive files.
	CanMoveTeamDriveItem bool `json:"canMoveTeamDriveItem,omitempty"`

	// CanReadRevisions: Whether the current user can read the revisions
	// resource of this file. For a Team Drive item, whether revisions of
	// non-folder descendants of this item, or this item itself if it is not
	// a folder, can be read.
	CanReadRevisions bool `json:"canReadRevisions,omitempty"`

	// CanReadTeamDrive: Whether the current user can read the Team Drive to
	// which this file belongs. Only populated for Team Drive files.
	CanReadTeamDrive bool `json:"canReadTeamDrive,omitempty"`

	// CanRemoveChildren: Whether the current user can remove children from
	// this folder. This is always false when the item is not a folder.
	CanRemoveChildren bool `json:"canRemoveChildren,omitempty"`

	// CanRename: Whether the current user can rename this file.
	CanRename bool `json:"canRename,omitempty"`

	// CanShare: Whether the current user can modify the sharing settings
	// for this file.
	CanShare bool `json:"canShare,omitempty"`

	// CanTrash: Whether the current user can move this file to trash.
	CanTrash bool `json:"canTrash,omitempty"`

	// CanUntrash: Whether the current user can restore this file from
	// trash.
	CanUntrash bool `json:"canUntrash,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CanAddChildren") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CanAddChildren") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FileCapabilities) MarshalJSON() ([]byte, error) {
	type noMethod FileCapabilities
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FileContentHints: Additional information about the content of the
// file. These fields are never populated in responses.
type FileContentHints struct {
	// IndexableText: Text to be indexed for the file to improve fullText
	// queries. This is limited to 128KB in length and may contain HTML
	// elements.
	IndexableText string `json:"indexableText,omitempty"`

	// Thumbnail: A thumbnail for the file. This will only be used if Drive
	// cannot generate a standard thumbnail.
	Thumbnail *FileContentHintsThumbnail `json:"thumbnail,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IndexableText") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IndexableText") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FileContentHints) MarshalJSON() ([]byte, error) {
	type noMethod FileContentHints
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FileContentHintsThumbnail: A thumbnail for the file. This will only
// be used if Drive cannot generate a standard thumbnail.
type FileContentHintsThumbnail struct {
	// Image: The thumbnail data encoded with URL-safe Base64 (RFC 4648
	// section 5).
	Image string `json:"image,omitempty"`

	// MimeType: The MIME type of the thumbnail.
	MimeType string `json:"mimeType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Image") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Image") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FileContentHintsThumbnail) MarshalJSON() ([]byte, error) {
	type noMethod FileContentHintsThumbnail
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FileImageMediaMetadata: Additional metadata about image media, if
// available.
type FileImageMediaMetadata struct {
	// Aperture: The aperture used to create the photo (f-number).
	Aperture float64 `json:"aperture,omitempty"`

	// CameraMake: The make of the camera used to create the photo.
	CameraMake string `json:"cameraMake,omitempty"`

	// CameraModel: The model of the camera used to create the photo.
	CameraModel string `json:"cameraModel,omitempty"`

	// ColorSpace: The color space of the photo.
	ColorSpace string `json:"colorSpace,omitempty"`

	// ExposureBias: The exposure bias of the photo (APEX value).
	ExposureBias float64 `json:"exposureBias,omitempty"`

	// ExposureMode: The exposure mode used to create the photo.
	ExposureMode string `json:"exposureMode,omitempty"`

	// ExposureTime: The length of the exposure, in seconds.
	ExposureTime float64 `json:"exposureTime,omitempty"`

	// FlashUsed: Whether a flash was used to create the photo.
	FlashUsed bool `json:"flashUsed,omitempty"`

	// FocalLength: The focal length used to create the photo, in
	// millimeters.
	FocalLength float64 `json:"focalLength,omitempty"`

	// Height: The height of the image in pixels.
	Height int64 `json:"height,omitempty"`

	// IsoSpeed: The ISO speed used to create the photo.
	IsoSpeed int64 `json:"isoSpeed,omitempty"`

	// Lens: The lens used to create the photo.
	Lens string `json:"lens,omitempty"`

	// Location: Geographic location information stored in the image.
	Location *FileImageMediaMetadataLocation `json:"location,omitempty"`

	// MaxApertureValue: The smallest f-number of the lens at the focal
	// length used to create the photo (APEX value).
	MaxApertureValue float64 `json:"maxApertureValue,omitempty"`

	// MeteringMode: The metering mode used to create the photo.
	MeteringMode string `json:"meteringMode,omitempty"`

	// Rotation: The rotation in clockwise degrees from the image's original
	// orientation.
	Rotation int64 `json:"rotation,omitempty"`

	// Sensor: The type of sensor used to create the photo.
	Sensor string `json:"sensor,omitempty"`

	// SubjectDistance: The distance to the subject of the photo, in meters.
	SubjectDistance int64 `json:"subjectDistance,omitempty"`

	// Time: The date and time the photo was taken (EXIF DateTime).
	Time string `json:"time,omitempty"`

	// WhiteBalance: The white balance mode used to create the photo.
	WhiteBalance string `json:"whiteBalance,omitempty"`

	// Width: The width of the image in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Aperture") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Aperture") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FileImageMediaMetadata) MarshalJSON() ([]byte, error) {
	type noMethod FileImageMediaMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *FileImageMediaMetadata) UnmarshalJSON(data []byte) error {
	type noMethod FileImageMediaMetadata
	var s1 struct {
		Aperture         gensupport.JSONFloat64 `json:"aperture"`
		ExposureBias     gensupport.JSONFloat64 `json:"exposureBias"`
		ExposureTime     gensupport.JSONFloat64 `json:"exposureTime"`
		FocalLength      gensupport.JSONFloat64 `json:"focalLength"`
		MaxApertureValue gensupport.JSONFloat64 `json:"maxApertureValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Aperture = float64(s1.Aperture)
	s.ExposureBias = float64(s1.ExposureBias)
	s.ExposureTime = float64(s1.ExposureTime)
	s.FocalLength = float64(s1.FocalLength)
	s.MaxApertureValue = float64(s1.MaxApertureValue)
	return nil
}

// FileImageMediaMetadataLocation: Geographic location information
// stored in the image.
type FileImageMediaMetadataLocation struct {
	// Altitude: The altitude stored in the image.
	Altitude float64 `json:"altitude,omitempty"`

	// Latitude: The latitude stored in the image.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude stored in the image.
	Longitude float64 `json:"longitude,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Altitude") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Altitude") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FileImageMediaMetadataLocation) MarshalJSON() ([]byte, error) {
	type noMethod FileImageMediaMetadataLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *FileImageMediaMetadataLocation) UnmarshalJSON(data []byte) error {
	type noMethod FileImageMediaMetadataLocation
	var s1 struct {
		Altitude  gensupport.JSONFloat64 `json:"altitude"`
		Latitude  gensupport.JSONFloat64 `json:"latitude"`
		Longitude gensupport.JSONFloat64 `json:"longitude"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Altitude = float64(s1.Altitude)
	s.Latitude = float64(s1.Latitude)
	s.Longitude = float64(s1.Longitude)
	return nil
}

// FileVideoMediaMetadata: Additional metadata about video media. This
// may not be available immediately upon upload.
type FileVideoMediaMetadata struct {
	// DurationMillis: The duration of the video in milliseconds.
	DurationMillis int64 `json:"durationMillis,omitempty,string"`

	// Height: The height of the video in pixels.
	Height int64 `json:"height,omitempty"`

	// Width: The width of the video in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DurationMillis") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DurationMillis") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FileVideoMediaMetadata) MarshalJSON() ([]byte, error) {
	type noMethod FileVideoMediaMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FileList: A list of files.
type FileList struct {
	// Files: The list of files. If nextPageToken is populated, then this
	// list may be incomplete and an additional page of results should be
	// fetched.
	Files []*File `json:"files,omitempty"`

	// IncompleteSearch: Whether the search process was incomplete. If true,
	// then some search results may be missing, since all documents were not
	// searched. This may occur when searching multiple Team Drives with the
	// "user,allTeamDrives" corpora, but all corpora could not be searched.
	// When this happens, it is suggested that clients narrow their query by
	// choosing a different corpus such as "user" or "teamDrive".
	IncompleteSearch bool `json:"incompleteSearch,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#fileList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of files. This will
	// be absent if the end of the files list has been reached. If the token
	// is rejected for any reason, it should be discarded, and pagination
	// should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Files") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Files") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FileList) MarshalJSON() ([]byte, error) {
	type noMethod FileList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GeneratedIds: A list of generated file IDs which can be provided in
// create requests.
type GeneratedIds struct {
	// Ids: The IDs generated for the requesting user in the specified
	// space.
	Ids []string `json:"ids,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#generatedIds".
	Kind string `json:"kind,omitempty"`

	// Space: The type of file that can be created with these IDs.
	Space string `json:"space,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Ids") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Ids") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GeneratedIds) MarshalJSON() ([]byte, error) {
	type noMethod GeneratedIds
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Permission: A permission for a file. A permission grants a user,
// group, domain or the world access to a file or a folder hierarchy.
type Permission struct {
	// AllowFileDiscovery: Whether the permission allows the file to be
	// discovered through search. This is only applicable for permissions of
	// type domain or anyone.
	AllowFileDiscovery bool `json:"allowFileDiscovery,omitempty"`

	// Deleted: Whether the account associated with this permission has been
	// deleted. This field only pertains to user and group permissions.
	Deleted bool `json:"deleted,omitempty"`

	// DisplayName: A displayable name for users, groups or domains.
	DisplayName string `json:"displayName,omitempty"`

	// Domain: The domain to which this permission refers.
	Domain string `json:"domain,omitempty"`

	// EmailAddress: The email address of the user or group to which this
	// permission refers.
	EmailAddress string `json:"emailAddress,omitempty"`

	// ExpirationTime: The time at which this permission will expire (RFC
	// 3339 date-time). Expiration times have the following restrictions:
	//
	// - They can only be set on user and group permissions
	// - The time must be in the future
	// - The time cannot be more than a year in the future
	ExpirationTime string `json:"expirationTime,omitempty"`

	// Id: The ID of this permission. This is a unique identifier for the
	// grantee, and is published in User resources as permissionId.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#permission".
	Kind string `json:"kind,omitempty"`

	// PhotoLink: A link to the user's profile photo, if available.
	PhotoLink string `json:"photoLink,omitempty"`

	// Role: The role granted by this permission. While new values may be
	// supported in the future, the following are currently allowed:
	// - organizer
	// - owner
	// - writer
	// - commenter
	// - reader
	Role string `json:"role,omitempty"`

	// TeamDrivePermissionDetails: Details of whether the permissions on
	// this Team Drive item are inherited or directly on this item. This is
	// an output-only field which is present only for Team Drive items.
	TeamDrivePermissionDetails []*PermissionTeamDrivePermissionDetails `json:"teamDrivePermissionDetails,omitempty"`

	// Type: The type of the grantee. Valid values are:
	// - user
	// - group
	// - domain
	// - anyone
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AllowFileDiscovery")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowFileDiscovery") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Permission) MarshalJSON() ([]byte, error) {
	type noMethod Permission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type PermissionTeamDrivePermissionDetails struct {
	// Inherited: Whether this permission is inherited. This field is always
	// populated. This is an output-only field.
	Inherited bool `json:"inherited,omitempty"`

	// InheritedFrom: The ID of the item from which this permission is
	// inherited. This is an output-only field and is only populated for
	// members of the Team Drive.
	InheritedFrom string `json:"inheritedFrom,omitempty"`

	// Role: The primary role for this user. While new values may be added
	// in the future, the following are currently possible:
	// - organizer
	// - writer
	// - commenter
	// - reader
	Role string `json:"role,omitempty"`

	// TeamDrivePermissionType: The Team Drive permission type for this
	// user. While new values may be added in future, the following are
	// currently possible:
	// - file
	// - member
	TeamDrivePermissionType string `json:"teamDrivePermissionType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Inherited") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Inherited") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PermissionTeamDrivePermissionDetails) MarshalJSON() ([]byte, error) {
	type noMethod PermissionTeamDrivePermissionDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PermissionList: A list of permissions for a file.
type PermissionList struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#permissionList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of permissions. This
	// field will be absent if the end of the permissions list has been
	// reached. If the token is rejected for any reason, it should be
	// discarded, and pagination should be restarted from the first page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Permissions: The list of permissions. If nextPageToken is populated,
	// then this list may be incomplete and an additional page of results
	// should be fetched.
	Permissions []*Permission `json:"permissions,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PermissionList) MarshalJSON() ([]byte, error) {
	type noMethod PermissionList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Reply: A reply to a comment on a file.
type Reply struct {
	// Action: The action the reply performed to the parent comment. Valid
	// values are:
	// - resolve
	// - reopen
	Action string `json:"action,omitempty"`

	// Author: The user who created the reply.
	Author *User `json:"author,omitempty"`

	// Content: The plain text content of the reply. This field is used for
	// setting the content, while htmlContent should be displayed. This is
	// required on creates if no action is specified.
	Content string `json:"content,omitempty"`

	// CreatedTime: The time at which the reply was created (RFC 3339
	// date-time).
	CreatedTime string `json:"createdTime,omitempty"`

	// Deleted: Whether the reply has been deleted. A deleted reply has no
	// content.
	Deleted bool `json:"deleted,omitempty"`

	// HtmlContent: The content of the reply with HTML formatting.
	HtmlContent string `json:"htmlContent,omitempty"`

	// Id: The ID of the reply.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#reply".
	Kind string `json:"kind,omitempty"`

	// ModifiedTime: The last time the reply was modified (RFC 3339
	// date-time).
	ModifiedTime string `json:"modifiedTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Action") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Action") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Reply) MarshalJSON() ([]byte, error) {
	type noMethod Reply
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReplyList: A list of replies to a comment on a file.
type ReplyList struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#replyList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of replies. This will
	// be absent if the end of the replies list has been reached. If the
	// token is rejected for any reason, it should be discarded, and
	// pagination should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Replies: The list of replies. If nextPageToken is populated, then
	// this list may be incomplete and an additional page of results should
	// be fetched.
	Replies []*Reply `json:"replies,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReplyList) MarshalJSON() ([]byte, error) {
	type noMethod ReplyList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Revision: The metadata for a revision to a file.
type Revision struct {
	// Id: The ID of the revision.
	Id string `json:"id,omitempty"`

	// KeepForever: Whether to keep this revision forever, even if it is no
	// longer the head revision. If not set, the revision will be
	// automatically purged 30 days after newer content is uploaded. This
	// can be set on a maximum of 200 revisions for a file.
	// This field is only applicable to files with binary content in Drive.
	KeepForever bool `json:"keepForever,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#revision".
	Kind string `json:"kind,omitempty"`

	// LastModifyingUser: The last user to modify this revision.
	LastModifyingUser *User `json:"lastModifyingUser,omitempty"`

	// Md5Checksum: The MD5 checksum of the revision's content. This is only
	// applicable to files with binary content in Drive.
	Md5Checksum string `json:"md5Checksum,omitempty"`

	// MimeType: The MIME type of the revision.
	MimeType string `json:"mimeType,omitempty"`

	// ModifiedTime: The last time the revision was modified (RFC 3339
	// date-time).
	ModifiedTime string `json:"modifiedTime,omitempty"`

	// OriginalFilename: The original filename used to create this revision.
	// This is only applicable to files with binary content in Drive.
	OriginalFilename string `json:"originalFilename,omitempty"`

	// PublishAuto: Whether subsequent revisions will be automatically
	// republished. This is only applicable to Google Docs.
	PublishAuto bool `json:"publishAuto,omitempty"`

	// Published: Whether this revision is published. This is only
	// applicable to Google Docs.
	Published bool `json:"published,omitempty"`

	// PublishedOutsideDomain: Whether this revision is published outside
	// the domain. This is only applicable to Google Docs.
	PublishedOutsideDomain bool `json:"publishedOutsideDomain,omitempty"`

	// Size: The size of the revision's content in bytes. This is only
	// applicable to files with binary content in Drive.
	Size int64 `json:"size,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Revision) MarshalJSON() ([]byte, error) {
	type noMethod Revision
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevisionList: A list of revisions of a file.
type RevisionList struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#revisionList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of revisions. This
	// will be absent if the end of the revisions list has been reached. If
	// the token is rejected for any reason, it should be discarded, and
	// pagination should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Revisions: The list of revisions. If nextPageToken is populated, then
	// this list may be incomplete and an additional page of results should
	// be fetched.
	Revisions []*Revision `json:"revisions,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevisionList) MarshalJSON() ([]byte, error) {
	type noMethod RevisionList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type StartPageToken struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#startPageToken".
	Kind string `json:"kind,omitempty"`

	// StartPageToken: The starting page token for listing changes.
	StartPageToken string `json:"startPageToken,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *StartPageToken) MarshalJSON() ([]byte, error) {
	type noMethod StartPageToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TeamDrive: Representation of a Team Drive.
type TeamDrive struct {
	// BackgroundImageFile: An image file and cropping parameters from which
	// a background image for this Team Drive is set. This is a write only
	// field; it can only be set on drive.teamdrives.update requests that
	// don't set themeId. When specified, all fields of the
	// backgroundImageFile must be set.
	BackgroundImageFile *TeamDriveBackgroundImageFile `json:"backgroundImageFile,omitempty"`

	// BackgroundImageLink: A short-lived link to this Team Drive's
	// background image.
	BackgroundImageLink string `json:"backgroundImageLink,omitempty"`

	// Capabilities: Capabilities the current user has on this Team Drive.
	Capabilities *TeamDriveCapabilities `json:"capabilities,omitempty"`

	// ColorRgb: The color of this Team Drive as an RGB hex string. It can
	// only be set on a drive.teamdrives.update request that does not set
	// themeId.
	ColorRgb string `json:"colorRgb,omitempty"`

	// Id: The ID of this Team Drive which is also the ID of the top level
	// folder for this Team Drive.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#teamDrive".
	Kind string `json:"kind,omitempty"`

	// Name: The name of this Team Drive.
	Name string `json:"name,omitempty"`

	// ThemeId: The ID of the theme from which the background image and
	// color will be set. The set of possible teamDriveThemes can be
	// retrieved from a drive.about.get response. When not specified on a
	// drive.teamdrives.create request, a random theme is chosen from which
	// the background image and color are set. This is a write-only field;
	// it can only be set on requests that don't set colorRgb or
	// backgroundImageFile.
	ThemeId string `json:"themeId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BackgroundImageFile")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BackgroundImageFile") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TeamDrive) MarshalJSON() ([]byte, error) {
	type noMethod TeamDrive
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TeamDriveBackgroundImageFile: An image file and cropping parameters
// from which a background image for this Team Drive is set. This is a
// write only field; it can only be set on drive.teamdrives.update
// requests that don't set themeId. When specified, all fields of the
// backgroundImageFile must be set.
type TeamDriveBackgroundImageFile struct {
	// Id: The ID of an image file in Drive to use for the background image.
	Id string `json:"id,omitempty"`

	// Width: The width of the cropped image in the closed range of 0 to 1.
	// This value represents the width of the cropped image divided by the
	// width of the entire image. The height is computed by applying a width
	// to height aspect ratio of 80 to 9. The resulting image must be at
	// least 1280 pixels wide and 144 pixels high.
	Width float64 `json:"width,omitempty"`

	// XCoordinate: The X coordinate of the upper left corner of the
	// cropping area in the background image. This is a value in the closed
	// range of 0 to 1. This value represents the horizontal distance from
	// the left side of the entire image to the left side of the cropping
	// area divided by the width of the entire image.
	XCoordinate float64 `json:"xCoordinate,omitempty"`

	// YCoordinate: The Y coordinate of the upper left corner of the
	// cropping area in the background image. This is a value in the closed
	// range of 0 to 1. This value represents the vertical distance from the
	// top side of the entire image to the top side of the cropping area
	// divided by the height of the entire image.
	YCoordinate float64 `json:"yCoordinate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TeamDriveBackgroundImageFile) MarshalJSON() ([]byte, error) {
	type noMethod TeamDriveBackgroundImageFile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *TeamDriveBackgroundImageFile) UnmarshalJSON(data []byte) error {
	type noMethod TeamDriveBackgroundImageFile
	var s1 struct {
		Width       gensupport.JSONFloat64 `json:"width"`
		XCoordinate gensupport.JSONFloat64 `json:"xCoordinate"`
		YCoordinate gensupport.JSONFloat64 `json:"yCoordinate"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Width = float64(s1.Width)
	s.XCoordinate = float64(s1.XCoordinate)
	s.YCoordinate = float64(s1.YCoordinate)
	return nil
}

// TeamDriveCapabilities: Capabilities the current user has on this Team
// Drive.
type TeamDriveCapabilities struct {
	// CanAddChildren: Whether the current user can add children to folders
	// in this Team Drive.
	CanAddChildren bool `json:"canAddChildren,omitempty"`

	// CanChangeTeamDriveBackground: Whether the current user can change the
	// background of this Team Drive.
	CanChangeTeamDriveBackground bool `json:"canChangeTeamDriveBackground,omitempty"`

	// CanComment: Whether the current user can comment on files in this
	// Team Drive.
	CanComment bool `json:"canComment,omitempty"`

	// CanCopy: Whether the current user can copy files in this Team Drive.
	CanCopy bool `json:"canCopy,omitempty"`

	// CanDeleteTeamDrive: Whether the current user can delete this Team
	// Drive. Attempting to delete the Team Drive may still fail if there
	// are untrashed items inside the Team Drive.
	CanDeleteTeamDrive bool `json:"canDeleteTeamDrive,omitempty"`

	// CanDownload: Whether the current user can download files in this Team
	// Drive.
	CanDownload bool `json:"canDownload,omitempty"`

	// CanEdit: Whether the current user can edit files in this Team Drive
	CanEdit bool `json:"canEdit,omitempty"`

	// CanListChildren: Whether the current user can list the children of
	// folders in this Team Drive.
	CanListChildren bool `json:"canListChildren,omitempty"`

	// CanManageMembers: Whether the current user can add members to this
	// Team Drive or remove them or change their role.
	CanManageMembers bool `json:"canManageMembers,omitempty"`

	// CanReadRevisions: Whether the current user can read the revisions
	// resource of files in this Team Drive.
	CanReadRevisions bool `json:"canReadRevisions,omitempty"`

	// CanRemoveChildren: Whether the current user can remove children from
	// folders in this Team Drive.
	CanRemoveChildren bool `json:"canRemoveChildren,omitempty"`

	// CanRename: Whether the current user can rename files or folders in
	// this Team Drive.
	CanRename bool `json:"canRename,omitempty"`

	// CanRenameTeamDrive: Whether the current user can rename this Team
	// Drive.
	CanRenameTeamDrive bool `json:"canRenameTeamDrive,omitempty"`

	// CanShare: Whether the current user can share files or folders in this
	// Team Drive.
	CanShare bool `json:"canShare,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CanAddChildren") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CanAddChildren") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TeamDriveCapabilities) MarshalJSON() ([]byte, error) {
	type noMethod TeamDriveCapabilities
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TeamDriveList: A list of Team Drives.
type TeamDriveList struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#teamDriveList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The page token for the next page of Team Drives. This
	// will be absent if the end of the Team Drives list has been reached.
	// If the token is rejected for any reason, it should be discarded, and
	// pagination should be restarted from the first page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TeamDrives: The list of Team Drives. If nextPageToken is populated,
	// then this list may be incomplete and an additional page of results
	// should be fetched.
	TeamDrives []*TeamDrive `json:"teamDrives,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TeamDriveList) MarshalJSON() ([]byte, error) {
	type noMethod TeamDriveList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// User: Information about a Drive user.
type User struct {
	// DisplayName: A plain text displayable name for this user.
	DisplayName string `json:"displayName,omitempty"`

	// EmailAddress: The email address of the user. This may not be present
	// in certain contexts if the user has not made their email address
	// visible to the requester.
	EmailAddress string `json:"emailAddress,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "drive#user".
	Kind string `json:"kind,omitempty"`

	// Me: Whether this user is the requesting user.
	Me bool `json:"me,omitempty"`

	// PermissionId: The user's ID as visible in Permission resources.
	PermissionId string `json:"permissionId,omitempty"`

	// PhotoLink: A link to the user's profile photo, if available.
	PhotoLink string `json:"photoLink,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "drive.about.get":

type AboutGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets information about the user, the user's Drive, and system
// capabilities.
func (r *AboutService) Get() *AboutGetCall {
	c := &AboutGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AboutGetCall) Fields(s ...googleapi.Field) *AboutGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AboutGetCall) IfNoneMatch(entityTag string) *AboutGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AboutGetCall) Context(ctx context.Context) *AboutGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AboutGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AboutGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "about")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.about.get" call.
// Exactly one of *About or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *About.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AboutGetCall) Do(opts ...googleapi.CallOption) (*About, error) {
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
	ret := &About{
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
	//   "description": "Gets information about the user, the user's Drive, and system capabilities.",
	//   "httpMethod": "GET",
	//   "id": "drive.about.get",
	//   "path": "about",
	//   "response": {
	//     "$ref": "About"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.changes.getStartPageToken":

type ChangesGetStartPageTokenCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetStartPageToken: Gets the starting pageToken for listing future
// changes.
func (r *ChangesService) GetStartPageToken() *ChangesGetStartPageTokenCall {
	c := &ChangesGetStartPageTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *ChangesGetStartPageTokenCall) SupportsTeamDrives(supportsTeamDrives bool) *ChangesGetStartPageTokenCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TeamDriveId sets the optional parameter "teamDriveId": The ID of the
// Team Drive for which the starting pageToken for listing future
// changes from that Team Drive will be returned.
func (c *ChangesGetStartPageTokenCall) TeamDriveId(teamDriveId string) *ChangesGetStartPageTokenCall {
	c.urlParams_.Set("teamDriveId", teamDriveId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesGetStartPageTokenCall) Fields(s ...googleapi.Field) *ChangesGetStartPageTokenCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ChangesGetStartPageTokenCall) IfNoneMatch(entityTag string) *ChangesGetStartPageTokenCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesGetStartPageTokenCall) Context(ctx context.Context) *ChangesGetStartPageTokenCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ChangesGetStartPageTokenCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ChangesGetStartPageTokenCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "changes/startPageToken")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.changes.getStartPageToken" call.
// Exactly one of *StartPageToken or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StartPageToken.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ChangesGetStartPageTokenCall) Do(opts ...googleapi.CallOption) (*StartPageToken, error) {
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
	ret := &StartPageToken{
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
	//   "description": "Gets the starting pageToken for listing future changes.",
	//   "httpMethod": "GET",
	//   "id": "drive.changes.getStartPageToken",
	//   "parameters": {
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "teamDriveId": {
	//       "description": "The ID of the Team Drive for which the starting pageToken for listing future changes from that Team Drive will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "changes/startPageToken",
	//   "response": {
	//     "$ref": "StartPageToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.changes.list":

type ChangesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the changes for a user or Team Drive.
func (r *ChangesService) List(pageToken string) *ChangesListCall {
	c := &ChangesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// IncludeCorpusRemovals sets the optional parameter
// "includeCorpusRemovals": Whether changes should include the file
// resource if the file is still accessible by the user at the time of
// the request, even when a file was removed from the list of changes
// and there will be no further change entries for this file.
func (c *ChangesListCall) IncludeCorpusRemovals(includeCorpusRemovals bool) *ChangesListCall {
	c.urlParams_.Set("includeCorpusRemovals", fmt.Sprint(includeCorpusRemovals))
	return c
}

// IncludeRemoved sets the optional parameter "includeRemoved": Whether
// to include changes indicating that items have been removed from the
// list of changes, for example by deletion or loss of access.
func (c *ChangesListCall) IncludeRemoved(includeRemoved bool) *ChangesListCall {
	c.urlParams_.Set("includeRemoved", fmt.Sprint(includeRemoved))
	return c
}

// IncludeTeamDriveItems sets the optional parameter
// "includeTeamDriveItems": Whether Team Drive files or changes should
// be included in results.
func (c *ChangesListCall) IncludeTeamDriveItems(includeTeamDriveItems bool) *ChangesListCall {
	c.urlParams_.Set("includeTeamDriveItems", fmt.Sprint(includeTeamDriveItems))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of changes to return per page.
func (c *ChangesListCall) PageSize(pageSize int64) *ChangesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// RestrictToMyDrive sets the optional parameter "restrictToMyDrive":
// Whether to restrict the results to changes inside the My Drive
// hierarchy. This omits changes to files such as those in the
// Application Data folder or shared files which have not been added to
// My Drive.
func (c *ChangesListCall) RestrictToMyDrive(restrictToMyDrive bool) *ChangesListCall {
	c.urlParams_.Set("restrictToMyDrive", fmt.Sprint(restrictToMyDrive))
	return c
}

// Spaces sets the optional parameter "spaces": A comma-separated list
// of spaces to query within the user corpus. Supported values are
// 'drive', 'appDataFolder' and 'photos'.
func (c *ChangesListCall) Spaces(spaces string) *ChangesListCall {
	c.urlParams_.Set("spaces", spaces)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *ChangesListCall) SupportsTeamDrives(supportsTeamDrives bool) *ChangesListCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TeamDriveId sets the optional parameter "teamDriveId": The Team Drive
// from which changes will be returned. If specified the change IDs will
// be reflective of the Team Drive; use the combined Team Drive ID and
// change ID as an identifier.
func (c *ChangesListCall) TeamDriveId(teamDriveId string) *ChangesListCall {
	c.urlParams_.Set("teamDriveId", teamDriveId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesListCall) Fields(s ...googleapi.Field) *ChangesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ChangesListCall) IfNoneMatch(entityTag string) *ChangesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesListCall) Context(ctx context.Context) *ChangesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ChangesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ChangesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "changes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.changes.list" call.
// Exactly one of *ChangeList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ChangeList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ChangesListCall) Do(opts ...googleapi.CallOption) (*ChangeList, error) {
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
	ret := &ChangeList{
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
	//   "description": "Lists the changes for a user or Team Drive.",
	//   "httpMethod": "GET",
	//   "id": "drive.changes.list",
	//   "parameterOrder": [
	//     "pageToken"
	//   ],
	//   "parameters": {
	//     "includeCorpusRemovals": {
	//       "default": "false",
	//       "description": "Whether changes should include the file resource if the file is still accessible by the user at the time of the request, even when a file was removed from the list of changes and there will be no further change entries for this file.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "includeRemoved": {
	//       "default": "true",
	//       "description": "Whether to include changes indicating that items have been removed from the list of changes, for example by deletion or loss of access.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "includeTeamDriveItems": {
	//       "default": "false",
	//       "description": "Whether Team Drive files or changes should be included in results.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageSize": {
	//       "default": "100",
	//       "description": "The maximum number of changes to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response or to the response from the getStartPageToken method.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "restrictToMyDrive": {
	//       "default": "false",
	//       "description": "Whether to restrict the results to changes inside the My Drive hierarchy. This omits changes to files such as those in the Application Data folder or shared files which have not been added to My Drive.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "spaces": {
	//       "default": "drive",
	//       "description": "A comma-separated list of spaces to query within the user corpus. Supported values are 'drive', 'appDataFolder' and 'photos'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "teamDriveId": {
	//       "description": "The Team Drive from which changes will be returned. If specified the change IDs will be reflective of the Team Drive; use the combined Team Drive ID and change ID as an identifier.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "changes",
	//   "response": {
	//     "$ref": "ChangeList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsSubscription": true
	// }

}

// method id "drive.changes.watch":

type ChangesWatchCall struct {
	s          *Service
	channel    *Channel
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Watch: Subscribes to changes for a user.
func (r *ChangesService) Watch(pageToken string, channel *Channel) *ChangesWatchCall {
	c := &ChangesWatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("pageToken", pageToken)
	c.channel = channel
	return c
}

// IncludeCorpusRemovals sets the optional parameter
// "includeCorpusRemovals": Whether changes should include the file
// resource if the file is still accessible by the user at the time of
// the request, even when a file was removed from the list of changes
// and there will be no further change entries for this file.
func (c *ChangesWatchCall) IncludeCorpusRemovals(includeCorpusRemovals bool) *ChangesWatchCall {
	c.urlParams_.Set("includeCorpusRemovals", fmt.Sprint(includeCorpusRemovals))
	return c
}

// IncludeRemoved sets the optional parameter "includeRemoved": Whether
// to include changes indicating that items have been removed from the
// list of changes, for example by deletion or loss of access.
func (c *ChangesWatchCall) IncludeRemoved(includeRemoved bool) *ChangesWatchCall {
	c.urlParams_.Set("includeRemoved", fmt.Sprint(includeRemoved))
	return c
}

// IncludeTeamDriveItems sets the optional parameter
// "includeTeamDriveItems": Whether Team Drive files or changes should
// be included in results.
func (c *ChangesWatchCall) IncludeTeamDriveItems(includeTeamDriveItems bool) *ChangesWatchCall {
	c.urlParams_.Set("includeTeamDriveItems", fmt.Sprint(includeTeamDriveItems))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of changes to return per page.
func (c *ChangesWatchCall) PageSize(pageSize int64) *ChangesWatchCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// RestrictToMyDrive sets the optional parameter "restrictToMyDrive":
// Whether to restrict the results to changes inside the My Drive
// hierarchy. This omits changes to files such as those in the
// Application Data folder or shared files which have not been added to
// My Drive.
func (c *ChangesWatchCall) RestrictToMyDrive(restrictToMyDrive bool) *ChangesWatchCall {
	c.urlParams_.Set("restrictToMyDrive", fmt.Sprint(restrictToMyDrive))
	return c
}

// Spaces sets the optional parameter "spaces": A comma-separated list
// of spaces to query within the user corpus. Supported values are
// 'drive', 'appDataFolder' and 'photos'.
func (c *ChangesWatchCall) Spaces(spaces string) *ChangesWatchCall {
	c.urlParams_.Set("spaces", spaces)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *ChangesWatchCall) SupportsTeamDrives(supportsTeamDrives bool) *ChangesWatchCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TeamDriveId sets the optional parameter "teamDriveId": The Team Drive
// from which changes will be returned. If specified the change IDs will
// be reflective of the Team Drive; use the combined Team Drive ID and
// change ID as an identifier.
func (c *ChangesWatchCall) TeamDriveId(teamDriveId string) *ChangesWatchCall {
	c.urlParams_.Set("teamDriveId", teamDriveId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesWatchCall) Fields(s ...googleapi.Field) *ChangesWatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesWatchCall) Context(ctx context.Context) *ChangesWatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ChangesWatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ChangesWatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "changes/watch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.changes.watch" call.
// Exactly one of *Channel or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Channel.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ChangesWatchCall) Do(opts ...googleapi.CallOption) (*Channel, error) {
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
	ret := &Channel{
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
	//   "description": "Subscribes to changes for a user.",
	//   "httpMethod": "POST",
	//   "id": "drive.changes.watch",
	//   "parameterOrder": [
	//     "pageToken"
	//   ],
	//   "parameters": {
	//     "includeCorpusRemovals": {
	//       "default": "false",
	//       "description": "Whether changes should include the file resource if the file is still accessible by the user at the time of the request, even when a file was removed from the list of changes and there will be no further change entries for this file.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "includeRemoved": {
	//       "default": "true",
	//       "description": "Whether to include changes indicating that items have been removed from the list of changes, for example by deletion or loss of access.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "includeTeamDriveItems": {
	//       "default": "false",
	//       "description": "Whether Team Drive files or changes should be included in results.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageSize": {
	//       "default": "100",
	//       "description": "The maximum number of changes to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response or to the response from the getStartPageToken method.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "restrictToMyDrive": {
	//       "default": "false",
	//       "description": "Whether to restrict the results to changes inside the My Drive hierarchy. This omits changes to files such as those in the Application Data folder or shared files which have not been added to My Drive.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "spaces": {
	//       "default": "drive",
	//       "description": "A comma-separated list of spaces to query within the user corpus. Supported values are 'drive', 'appDataFolder' and 'photos'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "teamDriveId": {
	//       "description": "The Team Drive from which changes will be returned. If specified the change IDs will be reflective of the Team Drive; use the combined Team Drive ID and change ID as an identifier.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "changes/watch",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "response": {
	//     "$ref": "Channel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsSubscription": true
	// }

}

// method id "drive.channels.stop":

type ChannelsStopCall struct {
	s          *Service
	channel    *Channel
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Stop: Stop watching resources through this channel
func (r *ChannelsService) Stop(channel *Channel) *ChannelsStopCall {
	c := &ChannelsStopCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.channel = channel
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChannelsStopCall) Fields(s ...googleapi.Field) *ChannelsStopCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChannelsStopCall) Context(ctx context.Context) *ChannelsStopCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ChannelsStopCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ChannelsStopCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "channels/stop")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.channels.stop" call.
func (c *ChannelsStopCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Stop watching resources through this channel",
	//   "httpMethod": "POST",
	//   "id": "drive.channels.stop",
	//   "path": "channels/stop",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.comments.create":

type CommentsCreateCall struct {
	s          *Service
	fileId     string
	comment    *Comment
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new comment on a file.
func (r *CommentsService) Create(fileId string, comment *Comment) *CommentsCreateCall {
	c := &CommentsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.comment = comment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsCreateCall) Fields(s ...googleapi.Field) *CommentsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsCreateCall) Context(ctx context.Context) *CommentsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.comment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.comments.create" call.
// Exactly one of *Comment or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Comment.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CommentsCreateCall) Do(opts ...googleapi.CallOption) (*Comment, error) {
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
	ret := &Comment{
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
	//   "description": "Creates a new comment on a file.",
	//   "httpMethod": "POST",
	//   "id": "drive.comments.create",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments",
	//   "request": {
	//     "$ref": "Comment"
	//   },
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.comments.delete":

type CommentsDeleteCall struct {
	s          *Service
	fileId     string
	commentId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a comment.
func (r *CommentsService) Delete(fileId string, commentId string) *CommentsDeleteCall {
	c := &CommentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsDeleteCall) Fields(s ...googleapi.Field) *CommentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsDeleteCall) Context(ctx context.Context) *CommentsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.comments.delete" call.
func (c *CommentsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a comment.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.comments.delete",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.comments.get":

type CommentsGetCall struct {
	s            *Service
	fileId       string
	commentId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a comment by ID.
func (r *CommentsService) Get(fileId string, commentId string) *CommentsGetCall {
	c := &CommentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": Whether
// to return deleted comments. Deleted comments will not include their
// original content.
func (c *CommentsGetCall) IncludeDeleted(includeDeleted bool) *CommentsGetCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsGetCall) Fields(s ...googleapi.Field) *CommentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CommentsGetCall) IfNoneMatch(entityTag string) *CommentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsGetCall) Context(ctx context.Context) *CommentsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.comments.get" call.
// Exactly one of *Comment or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Comment.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CommentsGetCall) Do(opts ...googleapi.CallOption) (*Comment, error) {
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
	ret := &Comment{
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
	//   "description": "Gets a comment by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.comments.get",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "default": "false",
	//       "description": "Whether to return deleted comments. Deleted comments will not include their original content.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.comments.list":

type CommentsListCall struct {
	s            *Service
	fileId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists a file's comments.
func (r *CommentsService) List(fileId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": Whether
// to include deleted comments. Deleted comments will not include their
// original content.
func (c *CommentsListCall) IncludeDeleted(includeDeleted bool) *CommentsListCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of comments to return per page.
func (c *CommentsListCall) PageSize(pageSize int64) *CommentsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token for
// continuing a previous list request on the next page. This should be
// set to the value of 'nextPageToken' from the previous response.
func (c *CommentsListCall) PageToken(pageToken string) *CommentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartModifiedTime sets the optional parameter "startModifiedTime":
// The minimum value of 'modifiedTime' for the result comments (RFC 3339
// date-time).
func (c *CommentsListCall) StartModifiedTime(startModifiedTime string) *CommentsListCall {
	c.urlParams_.Set("startModifiedTime", startModifiedTime)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsListCall) Fields(s ...googleapi.Field) *CommentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CommentsListCall) IfNoneMatch(entityTag string) *CommentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsListCall) Context(ctx context.Context) *CommentsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.comments.list" call.
// Exactly one of *CommentList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CommentList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CommentsListCall) Do(opts ...googleapi.CallOption) (*CommentList, error) {
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
	ret := &CommentList{
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
	//   "description": "Lists a file's comments.",
	//   "httpMethod": "GET",
	//   "id": "drive.comments.list",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "default": "false",
	//       "description": "Whether to include deleted comments. Deleted comments will not include their original content.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageSize": {
	//       "default": "20",
	//       "description": "The maximum number of comments to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startModifiedTime": {
	//       "description": "The minimum value of 'modifiedTime' for the result comments (RFC 3339 date-time).",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments",
	//   "response": {
	//     "$ref": "CommentList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *CommentsListCall) Pages(ctx context.Context, f func(*CommentList) error) error {
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

// method id "drive.comments.update":

type CommentsUpdateCall struct {
	s          *Service
	fileId     string
	commentId  string
	comment    *Comment
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a comment with patch semantics.
func (r *CommentsService) Update(fileId string, commentId string, comment *Comment) *CommentsUpdateCall {
	c := &CommentsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	c.comment = comment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsUpdateCall) Fields(s ...googleapi.Field) *CommentsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsUpdateCall) Context(ctx context.Context) *CommentsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.comment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.comments.update" call.
// Exactly one of *Comment or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Comment.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CommentsUpdateCall) Do(opts ...googleapi.CallOption) (*Comment, error) {
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
	ret := &Comment{
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
	//   "description": "Updates a comment with patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.comments.update",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}",
	//   "request": {
	//     "$ref": "Comment"
	//   },
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.files.copy":

type FilesCopyCall struct {
	s          *Service
	fileId     string
	file       *File
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Copy: Creates a copy of a file and applies any requested updates with
// patch semantics.
func (r *FilesService) Copy(fileId string, file *File) *FilesCopyCall {
	c := &FilesCopyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.file = file
	return c
}

// IgnoreDefaultVisibility sets the optional parameter
// "ignoreDefaultVisibility": Whether to ignore the domain's default
// visibility settings for the created file. Domain administrators can
// choose to make all uploaded files visible to the domain by default;
// this parameter bypasses that behavior for the request. Permissions
// are still inherited from parent folders.
func (c *FilesCopyCall) IgnoreDefaultVisibility(ignoreDefaultVisibility bool) *FilesCopyCall {
	c.urlParams_.Set("ignoreDefaultVisibility", fmt.Sprint(ignoreDefaultVisibility))
	return c
}

// KeepRevisionForever sets the optional parameter
// "keepRevisionForever": Whether to set the 'keepForever' field in the
// new head revision. This is only applicable to files with binary
// content in Drive.
func (c *FilesCopyCall) KeepRevisionForever(keepRevisionForever bool) *FilesCopyCall {
	c.urlParams_.Set("keepRevisionForever", fmt.Sprint(keepRevisionForever))
	return c
}

// OcrLanguage sets the optional parameter "ocrLanguage": A language
// hint for OCR processing during image import (ISO 639-1 code).
func (c *FilesCopyCall) OcrLanguage(ocrLanguage string) *FilesCopyCall {
	c.urlParams_.Set("ocrLanguage", ocrLanguage)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesCopyCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesCopyCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesCopyCall) Fields(s ...googleapi.Field) *FilesCopyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesCopyCall) Context(ctx context.Context) *FilesCopyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesCopyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesCopyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/copy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.copy" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *FilesCopyCall) Do(opts ...googleapi.CallOption) (*File, error) {
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
	ret := &File{
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
	//   "description": "Creates a copy of a file and applies any requested updates with patch semantics.",
	//   "httpMethod": "POST",
	//   "id": "drive.files.copy",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "ignoreDefaultVisibility": {
	//       "default": "false",
	//       "description": "Whether to ignore the domain's default visibility settings for the created file. Domain administrators can choose to make all uploaded files visible to the domain by default; this parameter bypasses that behavior for the request. Permissions are still inherited from parent folders.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "keepRevisionForever": {
	//       "default": "false",
	//       "description": "Whether to set the 'keepForever' field in the new head revision. This is only applicable to files with binary content in Drive.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "ocrLanguage": {
	//       "description": "A language hint for OCR processing during image import (ISO 639-1 code).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/copy",
	//   "request": {
	//     "$ref": "File"
	//   },
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.photos.readonly"
	//   ]
	// }

}

// method id "drive.files.create":

type FilesCreateCall struct {
	s          *Service
	file       *File
	urlParams_ gensupport.URLParams
	mediaInfo_ *gensupport.MediaInfo
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new file.
func (r *FilesService) Create(file *File) *FilesCreateCall {
	c := &FilesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.file = file
	return c
}

// IgnoreDefaultVisibility sets the optional parameter
// "ignoreDefaultVisibility": Whether to ignore the domain's default
// visibility settings for the created file. Domain administrators can
// choose to make all uploaded files visible to the domain by default;
// this parameter bypasses that behavior for the request. Permissions
// are still inherited from parent folders.
func (c *FilesCreateCall) IgnoreDefaultVisibility(ignoreDefaultVisibility bool) *FilesCreateCall {
	c.urlParams_.Set("ignoreDefaultVisibility", fmt.Sprint(ignoreDefaultVisibility))
	return c
}

// KeepRevisionForever sets the optional parameter
// "keepRevisionForever": Whether to set the 'keepForever' field in the
// new head revision. This is only applicable to files with binary
// content in Drive.
func (c *FilesCreateCall) KeepRevisionForever(keepRevisionForever bool) *FilesCreateCall {
	c.urlParams_.Set("keepRevisionForever", fmt.Sprint(keepRevisionForever))
	return c
}

// OcrLanguage sets the optional parameter "ocrLanguage": A language
// hint for OCR processing during image import (ISO 639-1 code).
func (c *FilesCreateCall) OcrLanguage(ocrLanguage string) *FilesCreateCall {
	c.urlParams_.Set("ocrLanguage", ocrLanguage)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesCreateCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesCreateCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// UseContentAsIndexableText sets the optional parameter
// "useContentAsIndexableText": Whether to use the uploaded content as
// indexable text.
func (c *FilesCreateCall) UseContentAsIndexableText(useContentAsIndexableText bool) *FilesCreateCall {
	c.urlParams_.Set("useContentAsIndexableText", fmt.Sprint(useContentAsIndexableText))
	return c
}

// Media specifies the media to upload in one or more chunks. The chunk
// size may be controlled by supplying a MediaOption generated by
// googleapi.ChunkSize. The chunk size defaults to
// googleapi.DefaultUploadChunkSize.The Content-Type header used in the
// upload request will be determined by sniffing the contents of r,
// unless a MediaOption generated by googleapi.ContentType is
// supplied.
// At most one of Media and ResumableMedia may be set.
func (c *FilesCreateCall) Media(r io.Reader, options ...googleapi.MediaOption) *FilesCreateCall {
	c.mediaInfo_ = gensupport.NewInfoFromMedia(r, options)
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx.
//
// Deprecated: use Media instead.
//
// At most one of Media and ResumableMedia may be set. mediaType
// identifies the MIME media type of the upload, such as "image/png". If
// mediaType is "", it will be auto-detected. The provided ctx will
// supersede any context previously provided to the Context method.
func (c *FilesCreateCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *FilesCreateCall {
	c.ctx_ = ctx
	c.mediaInfo_ = gensupport.NewInfoFromResumableMedia(r, size, mediaType)
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *FilesCreateCall) ProgressUpdater(pu googleapi.ProgressUpdater) *FilesCreateCall {
	c.mediaInfo_.SetProgressUpdater(pu)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesCreateCall) Fields(s ...googleapi.Field) *FilesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *FilesCreateCall) Context(ctx context.Context) *FilesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files")
	if c.mediaInfo_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.mediaInfo_.UploadType())
	}
	if body == nil {
		body = new(bytes.Buffer)
		reqHeaders.Set("Content-Type", "application/json")
	}
	body, cleanup := c.mediaInfo_.UploadRequest(reqHeaders, body)
	defer cleanup()
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.create" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *FilesCreateCall) Do(opts ...googleapi.CallOption) (*File, error) {
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
	rx := c.mediaInfo_.ResumableUpload(res.Header.Get("Location"))
	if rx != nil {
		rx.Client = c.s.client
		rx.UserAgent = c.s.userAgent()
		ctx := c.ctx_
		if ctx == nil {
			ctx = context.TODO()
		}
		res, err = rx.Upload(ctx)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
		if err := googleapi.CheckResponse(res); err != nil {
			return nil, err
		}
	}
	ret := &File{
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
	//   "description": "Creates a new file.",
	//   "httpMethod": "POST",
	//   "id": "drive.files.create",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "maxSize": "5120GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/drive/v3/files"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/drive/v3/files"
	//       }
	//     }
	//   },
	//   "parameters": {
	//     "ignoreDefaultVisibility": {
	//       "default": "false",
	//       "description": "Whether to ignore the domain's default visibility settings for the created file. Domain administrators can choose to make all uploaded files visible to the domain by default; this parameter bypasses that behavior for the request. Permissions are still inherited from parent folders.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "keepRevisionForever": {
	//       "default": "false",
	//       "description": "Whether to set the 'keepForever' field in the new head revision. This is only applicable to files with binary content in Drive.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "ocrLanguage": {
	//       "description": "A language hint for OCR processing during image import (ISO 639-1 code).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "useContentAsIndexableText": {
	//       "default": "false",
	//       "description": "Whether to use the uploaded content as indexable text.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files",
	//   "request": {
	//     "$ref": "File"
	//   },
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ],
	//   "supportsMediaUpload": true,
	//   "supportsSubscription": true
	// }

}

// method id "drive.files.delete":

type FilesDeleteCall struct {
	s          *Service
	fileId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Permanently deletes a file owned by the user without moving
// it to the trash. If the file belongs to a Team Drive the user must be
// an organizer on the parent. If the target is a folder, all
// descendants owned by the user are also deleted.
func (r *FilesService) Delete(fileId string) *FilesDeleteCall {
	c := &FilesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesDeleteCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesDeleteCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesDeleteCall) Fields(s ...googleapi.Field) *FilesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesDeleteCall) Context(ctx context.Context) *FilesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.delete" call.
func (c *FilesDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Permanently deletes a file owned by the user without moving it to the trash. If the file belongs to a Team Drive the user must be an organizer on the parent. If the target is a folder, all descendants owned by the user are also deleted.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.files.delete",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.files.emptyTrash":

type FilesEmptyTrashCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// EmptyTrash: Permanently deletes all of the user's trashed files.
func (r *FilesService) EmptyTrash() *FilesEmptyTrashCall {
	c := &FilesEmptyTrashCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesEmptyTrashCall) Fields(s ...googleapi.Field) *FilesEmptyTrashCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesEmptyTrashCall) Context(ctx context.Context) *FilesEmptyTrashCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesEmptyTrashCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesEmptyTrashCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/trash")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.emptyTrash" call.
func (c *FilesEmptyTrashCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Permanently deletes all of the user's trashed files.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.files.emptyTrash",
	//   "path": "files/trash",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive"
	//   ]
	// }

}

// method id "drive.files.export":

type FilesExportCall struct {
	s            *Service
	fileId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Export: Exports a Google Doc to the requested MIME type and returns
// the exported content. Please note that the exported content is
// limited to 10MB.
func (r *FilesService) Export(fileId string, mimeType string) *FilesExportCall {
	c := &FilesExportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.urlParams_.Set("mimeType", mimeType)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesExportCall) Fields(s ...googleapi.Field) *FilesExportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FilesExportCall) IfNoneMatch(entityTag string) *FilesExportCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do and Download
// methods. Any pending HTTP request will be aborted if the provided
// context is canceled.
func (c *FilesExportCall) Context(ctx context.Context) *FilesExportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesExportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesExportCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/export")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Download fetches the API endpoint's "media" value, instead of the normal
// API response value. If the returned error is nil, the Response is guaranteed to
// have a 2xx status code. Callers must close the Response.Body as usual.
func (c *FilesExportCall) Download(opts ...googleapi.CallOption) (*http.Response, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("media")
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckMediaResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

// Do executes the "drive.files.export" call.
func (c *FilesExportCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Exports a Google Doc to the requested MIME type and returns the exported content. Please note that the exported content is limited to 10MB.",
	//   "httpMethod": "GET",
	//   "id": "drive.files.export",
	//   "parameterOrder": [
	//     "fileId",
	//     "mimeType"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "mimeType": {
	//       "description": "The MIME type of the format requested for this export.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/export",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "drive.files.generateIds":

type FilesGenerateIdsCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GenerateIds: Generates a set of file IDs which can be provided in
// create requests.
func (r *FilesService) GenerateIds() *FilesGenerateIdsCall {
	c := &FilesGenerateIdsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Count sets the optional parameter "count": The number of IDs to
// return.
func (c *FilesGenerateIdsCall) Count(count int64) *FilesGenerateIdsCall {
	c.urlParams_.Set("count", fmt.Sprint(count))
	return c
}

// Space sets the optional parameter "space": The space in which the IDs
// can be used to create new files. Supported values are 'drive' and
// 'appDataFolder'.
func (c *FilesGenerateIdsCall) Space(space string) *FilesGenerateIdsCall {
	c.urlParams_.Set("space", space)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesGenerateIdsCall) Fields(s ...googleapi.Field) *FilesGenerateIdsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FilesGenerateIdsCall) IfNoneMatch(entityTag string) *FilesGenerateIdsCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesGenerateIdsCall) Context(ctx context.Context) *FilesGenerateIdsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesGenerateIdsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesGenerateIdsCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/generateIds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.generateIds" call.
// Exactly one of *GeneratedIds or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *GeneratedIds.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *FilesGenerateIdsCall) Do(opts ...googleapi.CallOption) (*GeneratedIds, error) {
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
	ret := &GeneratedIds{
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
	//   "description": "Generates a set of file IDs which can be provided in create requests.",
	//   "httpMethod": "GET",
	//   "id": "drive.files.generateIds",
	//   "parameters": {
	//     "count": {
	//       "default": "10",
	//       "description": "The number of IDs to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "space": {
	//       "default": "drive",
	//       "description": "The space in which the IDs can be used to create new files. Supported values are 'drive' and 'appDataFolder'.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/generateIds",
	//   "response": {
	//     "$ref": "GeneratedIds"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.files.get":

type FilesGetCall struct {
	s            *Service
	fileId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a file's metadata or content by ID.
func (r *FilesService) Get(fileId string) *FilesGetCall {
	c := &FilesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	return c
}

// AcknowledgeAbuse sets the optional parameter "acknowledgeAbuse":
// Whether the user is acknowledging the risk of downloading known
// malware or other abusive files. This is only applicable when
// alt=media.
func (c *FilesGetCall) AcknowledgeAbuse(acknowledgeAbuse bool) *FilesGetCall {
	c.urlParams_.Set("acknowledgeAbuse", fmt.Sprint(acknowledgeAbuse))
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesGetCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesGetCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesGetCall) Fields(s ...googleapi.Field) *FilesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FilesGetCall) IfNoneMatch(entityTag string) *FilesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do and Download
// methods. Any pending HTTP request will be aborted if the provided
// context is canceled.
func (c *FilesGetCall) Context(ctx context.Context) *FilesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Download fetches the API endpoint's "media" value, instead of the normal
// API response value. If the returned error is nil, the Response is guaranteed to
// have a 2xx status code. Callers must close the Response.Body as usual.
func (c *FilesGetCall) Download(opts ...googleapi.CallOption) (*http.Response, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("media")
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckMediaResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

// Do executes the "drive.files.get" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *FilesGetCall) Do(opts ...googleapi.CallOption) (*File, error) {
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
	ret := &File{
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
	//   "description": "Gets a file's metadata or content by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.files.get",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "acknowledgeAbuse": {
	//       "default": "false",
	//       "description": "Whether the user is acknowledging the risk of downloading known malware or other abusive files. This is only applicable when alt=media.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}",
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsMediaDownload": true,
	//   "supportsSubscription": true,
	//   "useMediaDownloadService": true
	// }

}

// method id "drive.files.list":

type FilesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists or searches files.
func (r *FilesService) List() *FilesListCall {
	c := &FilesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Corpora sets the optional parameter "corpora": Comma-separated list
// of bodies of items (files/documents) to which the query applies.
// Supported bodies are 'user', 'domain', 'teamDrive' and
// 'allTeamDrives'. 'allTeamDrives' must be combined with 'user'; all
// other values must be used in isolation. Prefer 'user' or 'teamDrive'
// to 'allTeamDrives' for efficiency.
func (c *FilesListCall) Corpora(corpora string) *FilesListCall {
	c.urlParams_.Set("corpora", corpora)
	return c
}

// Corpus sets the optional parameter "corpus": The source of files to
// list. Deprecated: use 'corpora' instead.
//
// Possible values:
//   "domain" - Files shared to the user's domain.
//   "user" - Files owned by or shared to the user.
func (c *FilesListCall) Corpus(corpus string) *FilesListCall {
	c.urlParams_.Set("corpus", corpus)
	return c
}

// IncludeTeamDriveItems sets the optional parameter
// "includeTeamDriveItems": Whether Team Drive items should be included
// in results.
func (c *FilesListCall) IncludeTeamDriveItems(includeTeamDriveItems bool) *FilesListCall {
	c.urlParams_.Set("includeTeamDriveItems", fmt.Sprint(includeTeamDriveItems))
	return c
}

// OrderBy sets the optional parameter "orderBy": A comma-separated list
// of sort keys. Valid keys are 'createdTime', 'folder',
// 'modifiedByMeTime', 'modifiedTime', 'name', 'name_natural',
// 'quotaBytesUsed', 'recency', 'sharedWithMeTime', 'starred', and
// 'viewedByMeTime'. Each key sorts ascending by default, but may be
// reversed with the 'desc' modifier. Example usage:
// ?orderBy=folder,modifiedTime desc,name. Please note that there is a
// current limitation for users with approximately one million files in
// which the requested sort order is ignored.
func (c *FilesListCall) OrderBy(orderBy string) *FilesListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of files to return per page. Partial or empty result pages are
// possible even before the end of the files list has been reached.
func (c *FilesListCall) PageSize(pageSize int64) *FilesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token for
// continuing a previous list request on the next page. This should be
// set to the value of 'nextPageToken' from the previous response.
func (c *FilesListCall) PageToken(pageToken string) *FilesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Q sets the optional parameter "q": A query for filtering the file
// results. See the "Search for Files" guide for supported syntax.
func (c *FilesListCall) Q(q string) *FilesListCall {
	c.urlParams_.Set("q", q)
	return c
}

// Spaces sets the optional parameter "spaces": A comma-separated list
// of spaces to query within the corpus. Supported values are 'drive',
// 'appDataFolder' and 'photos'.
func (c *FilesListCall) Spaces(spaces string) *FilesListCall {
	c.urlParams_.Set("spaces", spaces)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesListCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesListCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TeamDriveId sets the optional parameter "teamDriveId": ID of Team
// Drive to search.
func (c *FilesListCall) TeamDriveId(teamDriveId string) *FilesListCall {
	c.urlParams_.Set("teamDriveId", teamDriveId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesListCall) Fields(s ...googleapi.Field) *FilesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FilesListCall) IfNoneMatch(entityTag string) *FilesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesListCall) Context(ctx context.Context) *FilesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.list" call.
// Exactly one of *FileList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *FileList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *FilesListCall) Do(opts ...googleapi.CallOption) (*FileList, error) {
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
	ret := &FileList{
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
	//   "description": "Lists or searches files.",
	//   "httpMethod": "GET",
	//   "id": "drive.files.list",
	//   "parameters": {
	//     "corpora": {
	//       "description": "Comma-separated list of bodies of items (files/documents) to which the query applies. Supported bodies are 'user', 'domain', 'teamDrive' and 'allTeamDrives'. 'allTeamDrives' must be combined with 'user'; all other values must be used in isolation. Prefer 'user' or 'teamDrive' to 'allTeamDrives' for efficiency.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "corpus": {
	//       "description": "The source of files to list. Deprecated: use 'corpora' instead.",
	//       "enum": [
	//         "domain",
	//         "user"
	//       ],
	//       "enumDescriptions": [
	//         "Files shared to the user's domain.",
	//         "Files owned by or shared to the user."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "includeTeamDriveItems": {
	//       "default": "false",
	//       "description": "Whether Team Drive items should be included in results.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "orderBy": {
	//       "description": "A comma-separated list of sort keys. Valid keys are 'createdTime', 'folder', 'modifiedByMeTime', 'modifiedTime', 'name', 'name_natural', 'quotaBytesUsed', 'recency', 'sharedWithMeTime', 'starred', and 'viewedByMeTime'. Each key sorts ascending by default, but may be reversed with the 'desc' modifier. Example usage: ?orderBy=folder,modifiedTime desc,name. Please note that there is a current limitation for users with approximately one million files in which the requested sort order is ignored.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "default": "100",
	//       "description": "The maximum number of files to return per page. Partial or empty result pages are possible even before the end of the files list has been reached.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "A query for filtering the file results. See the \"Search for Files\" guide for supported syntax.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "spaces": {
	//       "default": "drive",
	//       "description": "A comma-separated list of spaces to query within the corpus. Supported values are 'drive', 'appDataFolder' and 'photos'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "teamDriveId": {
	//       "description": "ID of Team Drive to search.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "files",
	//   "response": {
	//     "$ref": "FileList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *FilesListCall) Pages(ctx context.Context, f func(*FileList) error) error {
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

// method id "drive.files.update":

type FilesUpdateCall struct {
	s          *Service
	fileId     string
	file       *File
	urlParams_ gensupport.URLParams
	mediaInfo_ *gensupport.MediaInfo
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a file's metadata and/or content with patch
// semantics.
func (r *FilesService) Update(fileId string, file *File) *FilesUpdateCall {
	c := &FilesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.file = file
	return c
}

// AddParents sets the optional parameter "addParents": A
// comma-separated list of parent IDs to add.
func (c *FilesUpdateCall) AddParents(addParents string) *FilesUpdateCall {
	c.urlParams_.Set("addParents", addParents)
	return c
}

// KeepRevisionForever sets the optional parameter
// "keepRevisionForever": Whether to set the 'keepForever' field in the
// new head revision. This is only applicable to files with binary
// content in Drive.
func (c *FilesUpdateCall) KeepRevisionForever(keepRevisionForever bool) *FilesUpdateCall {
	c.urlParams_.Set("keepRevisionForever", fmt.Sprint(keepRevisionForever))
	return c
}

// OcrLanguage sets the optional parameter "ocrLanguage": A language
// hint for OCR processing during image import (ISO 639-1 code).
func (c *FilesUpdateCall) OcrLanguage(ocrLanguage string) *FilesUpdateCall {
	c.urlParams_.Set("ocrLanguage", ocrLanguage)
	return c
}

// RemoveParents sets the optional parameter "removeParents": A
// comma-separated list of parent IDs to remove.
func (c *FilesUpdateCall) RemoveParents(removeParents string) *FilesUpdateCall {
	c.urlParams_.Set("removeParents", removeParents)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesUpdateCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesUpdateCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// UseContentAsIndexableText sets the optional parameter
// "useContentAsIndexableText": Whether to use the uploaded content as
// indexable text.
func (c *FilesUpdateCall) UseContentAsIndexableText(useContentAsIndexableText bool) *FilesUpdateCall {
	c.urlParams_.Set("useContentAsIndexableText", fmt.Sprint(useContentAsIndexableText))
	return c
}

// Media specifies the media to upload in one or more chunks. The chunk
// size may be controlled by supplying a MediaOption generated by
// googleapi.ChunkSize. The chunk size defaults to
// googleapi.DefaultUploadChunkSize.The Content-Type header used in the
// upload request will be determined by sniffing the contents of r,
// unless a MediaOption generated by googleapi.ContentType is
// supplied.
// At most one of Media and ResumableMedia may be set.
func (c *FilesUpdateCall) Media(r io.Reader, options ...googleapi.MediaOption) *FilesUpdateCall {
	c.mediaInfo_ = gensupport.NewInfoFromMedia(r, options)
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx.
//
// Deprecated: use Media instead.
//
// At most one of Media and ResumableMedia may be set. mediaType
// identifies the MIME media type of the upload, such as "image/png". If
// mediaType is "", it will be auto-detected. The provided ctx will
// supersede any context previously provided to the Context method.
func (c *FilesUpdateCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *FilesUpdateCall {
	c.ctx_ = ctx
	c.mediaInfo_ = gensupport.NewInfoFromResumableMedia(r, size, mediaType)
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *FilesUpdateCall) ProgressUpdater(pu googleapi.ProgressUpdater) *FilesUpdateCall {
	c.mediaInfo_.SetProgressUpdater(pu)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesUpdateCall) Fields(s ...googleapi.Field) *FilesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *FilesUpdateCall) Context(ctx context.Context) *FilesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}")
	if c.mediaInfo_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.mediaInfo_.UploadType())
	}
	if body == nil {
		body = new(bytes.Buffer)
		reqHeaders.Set("Content-Type", "application/json")
	}
	body, cleanup := c.mediaInfo_.UploadRequest(reqHeaders, body)
	defer cleanup()
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.files.update" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *FilesUpdateCall) Do(opts ...googleapi.CallOption) (*File, error) {
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
	rx := c.mediaInfo_.ResumableUpload(res.Header.Get("Location"))
	if rx != nil {
		rx.Client = c.s.client
		rx.UserAgent = c.s.userAgent()
		ctx := c.ctx_
		if ctx == nil {
			ctx = context.TODO()
		}
		res, err = rx.Upload(ctx)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
		if err := googleapi.CheckResponse(res); err != nil {
			return nil, err
		}
	}
	ret := &File{
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
	//   "description": "Updates a file's metadata and/or content with patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.files.update",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "maxSize": "5120GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/drive/v3/files/{fileId}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/drive/v3/files/{fileId}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "addParents": {
	//       "description": "A comma-separated list of parent IDs to add.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "keepRevisionForever": {
	//       "default": "false",
	//       "description": "Whether to set the 'keepForever' field in the new head revision. This is only applicable to files with binary content in Drive.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "ocrLanguage": {
	//       "description": "A language hint for OCR processing during image import (ISO 639-1 code).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "removeParents": {
	//       "description": "A comma-separated list of parent IDs to remove.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "useContentAsIndexableText": {
	//       "default": "false",
	//       "description": "Whether to use the uploaded content as indexable text.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}",
	//   "request": {
	//     "$ref": "File"
	//   },
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.scripts"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "drive.files.watch":

type FilesWatchCall struct {
	s          *Service
	fileId     string
	channel    *Channel
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Watch: Subscribes to changes to a file
func (r *FilesService) Watch(fileId string, channel *Channel) *FilesWatchCall {
	c := &FilesWatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.channel = channel
	return c
}

// AcknowledgeAbuse sets the optional parameter "acknowledgeAbuse":
// Whether the user is acknowledging the risk of downloading known
// malware or other abusive files. This is only applicable when
// alt=media.
func (c *FilesWatchCall) AcknowledgeAbuse(acknowledgeAbuse bool) *FilesWatchCall {
	c.urlParams_.Set("acknowledgeAbuse", fmt.Sprint(acknowledgeAbuse))
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *FilesWatchCall) SupportsTeamDrives(supportsTeamDrives bool) *FilesWatchCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesWatchCall) Fields(s ...googleapi.Field) *FilesWatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do and Download
// methods. Any pending HTTP request will be aborted if the provided
// context is canceled.
func (c *FilesWatchCall) Context(ctx context.Context) *FilesWatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FilesWatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FilesWatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/watch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Download fetches the API endpoint's "media" value, instead of the normal
// API response value. If the returned error is nil, the Response is guaranteed to
// have a 2xx status code. Callers must close the Response.Body as usual.
func (c *FilesWatchCall) Download(opts ...googleapi.CallOption) (*http.Response, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("media")
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckMediaResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

// Do executes the "drive.files.watch" call.
// Exactly one of *Channel or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Channel.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FilesWatchCall) Do(opts ...googleapi.CallOption) (*Channel, error) {
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
	ret := &Channel{
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
	//   "description": "Subscribes to changes to a file",
	//   "httpMethod": "POST",
	//   "id": "drive.files.watch",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "acknowledgeAbuse": {
	//       "default": "false",
	//       "description": "Whether the user is acknowledging the risk of downloading known malware or other abusive files. This is only applicable when alt=media.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/watch",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "response": {
	//     "$ref": "Channel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsMediaDownload": true,
	//   "supportsSubscription": true,
	//   "useMediaDownloadService": true
	// }

}

// method id "drive.permissions.create":

type PermissionsCreateCall struct {
	s          *Service
	fileId     string
	permission *Permission
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a permission for a file or Team Drive.
func (r *PermissionsService) Create(fileId string, permission *Permission) *PermissionsCreateCall {
	c := &PermissionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.permission = permission
	return c
}

// EmailMessage sets the optional parameter "emailMessage": A custom
// message to include in the notification email.
func (c *PermissionsCreateCall) EmailMessage(emailMessage string) *PermissionsCreateCall {
	c.urlParams_.Set("emailMessage", emailMessage)
	return c
}

// SendNotificationEmail sets the optional parameter
// "sendNotificationEmail": Whether to send a notification email when
// sharing to users or groups. This defaults to true for users and
// groups, and is not allowed for other requests. It must not be
// disabled for ownership transfers.
func (c *PermissionsCreateCall) SendNotificationEmail(sendNotificationEmail bool) *PermissionsCreateCall {
	c.urlParams_.Set("sendNotificationEmail", fmt.Sprint(sendNotificationEmail))
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *PermissionsCreateCall) SupportsTeamDrives(supportsTeamDrives bool) *PermissionsCreateCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TransferOwnership sets the optional parameter "transferOwnership":
// Whether to transfer ownership to the specified user and downgrade the
// current owner to a writer. This parameter is required as an
// acknowledgement of the side effect.
func (c *PermissionsCreateCall) TransferOwnership(transferOwnership bool) *PermissionsCreateCall {
	c.urlParams_.Set("transferOwnership", fmt.Sprint(transferOwnership))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsCreateCall) Fields(s ...googleapi.Field) *PermissionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsCreateCall) Context(ctx context.Context) *PermissionsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.permission)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.permissions.create" call.
// Exactly one of *Permission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Permission.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PermissionsCreateCall) Do(opts ...googleapi.CallOption) (*Permission, error) {
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
	ret := &Permission{
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
	//   "description": "Creates a permission for a file or Team Drive.",
	//   "httpMethod": "POST",
	//   "id": "drive.permissions.create",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "emailMessage": {
	//       "description": "A custom message to include in the notification email.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file or Team Drive.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sendNotificationEmail": {
	//       "description": "Whether to send a notification email when sharing to users or groups. This defaults to true for users and groups, and is not allowed for other requests. It must not be disabled for ownership transfers.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "transferOwnership": {
	//       "default": "false",
	//       "description": "Whether to transfer ownership to the specified user and downgrade the current owner to a writer. This parameter is required as an acknowledgement of the side effect.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/permissions",
	//   "request": {
	//     "$ref": "Permission"
	//   },
	//   "response": {
	//     "$ref": "Permission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.permissions.delete":

type PermissionsDeleteCall struct {
	s            *Service
	fileId       string
	permissionId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes a permission.
func (r *PermissionsService) Delete(fileId string, permissionId string) *PermissionsDeleteCall {
	c := &PermissionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.permissionId = permissionId
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *PermissionsDeleteCall) SupportsTeamDrives(supportsTeamDrives bool) *PermissionsDeleteCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsDeleteCall) Fields(s ...googleapi.Field) *PermissionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsDeleteCall) Context(ctx context.Context) *PermissionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/permissions/{permissionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":       c.fileId,
		"permissionId": c.permissionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.permissions.delete" call.
func (c *PermissionsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a permission.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.permissions.delete",
	//   "parameterOrder": [
	//     "fileId",
	//     "permissionId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file or Team Drive.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "permissionId": {
	//       "description": "The ID of the permission.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/permissions/{permissionId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.permissions.get":

type PermissionsGetCall struct {
	s            *Service
	fileId       string
	permissionId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a permission by ID.
func (r *PermissionsService) Get(fileId string, permissionId string) *PermissionsGetCall {
	c := &PermissionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.permissionId = permissionId
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *PermissionsGetCall) SupportsTeamDrives(supportsTeamDrives bool) *PermissionsGetCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsGetCall) Fields(s ...googleapi.Field) *PermissionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PermissionsGetCall) IfNoneMatch(entityTag string) *PermissionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsGetCall) Context(ctx context.Context) *PermissionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/permissions/{permissionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":       c.fileId,
		"permissionId": c.permissionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.permissions.get" call.
// Exactly one of *Permission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Permission.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PermissionsGetCall) Do(opts ...googleapi.CallOption) (*Permission, error) {
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
	ret := &Permission{
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
	//   "description": "Gets a permission by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.permissions.get",
	//   "parameterOrder": [
	//     "fileId",
	//     "permissionId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "permissionId": {
	//       "description": "The ID of the permission.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/permissions/{permissionId}",
	//   "response": {
	//     "$ref": "Permission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.permissions.list":

type PermissionsListCall struct {
	s            *Service
	fileId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists a file's or Team Drive's permissions.
func (r *PermissionsService) List(fileId string) *PermissionsListCall {
	c := &PermissionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of permissions to return per page. When not set for files in a Team
// Drive, at most 100 results will be returned. When not set for files
// that are not in a Team Drive, the entire list will be returned.
func (c *PermissionsListCall) PageSize(pageSize int64) *PermissionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token for
// continuing a previous list request on the next page. This should be
// set to the value of 'nextPageToken' from the previous response.
func (c *PermissionsListCall) PageToken(pageToken string) *PermissionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *PermissionsListCall) SupportsTeamDrives(supportsTeamDrives bool) *PermissionsListCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsListCall) Fields(s ...googleapi.Field) *PermissionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PermissionsListCall) IfNoneMatch(entityTag string) *PermissionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsListCall) Context(ctx context.Context) *PermissionsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.permissions.list" call.
// Exactly one of *PermissionList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *PermissionList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PermissionsListCall) Do(opts ...googleapi.CallOption) (*PermissionList, error) {
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
	ret := &PermissionList{
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
	//   "description": "Lists a file's or Team Drive's permissions.",
	//   "httpMethod": "GET",
	//   "id": "drive.permissions.list",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file or Team Drive.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of permissions to return per page. When not set for files in a Team Drive, at most 100 results will be returned. When not set for files that are not in a Team Drive, the entire list will be returned.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/permissions",
	//   "response": {
	//     "$ref": "PermissionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PermissionsListCall) Pages(ctx context.Context, f func(*PermissionList) error) error {
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

// method id "drive.permissions.update":

type PermissionsUpdateCall struct {
	s            *Service
	fileId       string
	permissionId string
	permission   *Permission
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates a permission with patch semantics.
func (r *PermissionsService) Update(fileId string, permissionId string, permission *Permission) *PermissionsUpdateCall {
	c := &PermissionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.permissionId = permissionId
	c.permission = permission
	return c
}

// RemoveExpiration sets the optional parameter "removeExpiration":
// Whether to remove the expiration date.
func (c *PermissionsUpdateCall) RemoveExpiration(removeExpiration bool) *PermissionsUpdateCall {
	c.urlParams_.Set("removeExpiration", fmt.Sprint(removeExpiration))
	return c
}

// SupportsTeamDrives sets the optional parameter "supportsTeamDrives":
// Whether the requesting application supports Team Drives.
func (c *PermissionsUpdateCall) SupportsTeamDrives(supportsTeamDrives bool) *PermissionsUpdateCall {
	c.urlParams_.Set("supportsTeamDrives", fmt.Sprint(supportsTeamDrives))
	return c
}

// TransferOwnership sets the optional parameter "transferOwnership":
// Whether to transfer ownership to the specified user and downgrade the
// current owner to a writer. This parameter is required as an
// acknowledgement of the side effect.
func (c *PermissionsUpdateCall) TransferOwnership(transferOwnership bool) *PermissionsUpdateCall {
	c.urlParams_.Set("transferOwnership", fmt.Sprint(transferOwnership))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsUpdateCall) Fields(s ...googleapi.Field) *PermissionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsUpdateCall) Context(ctx context.Context) *PermissionsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.permission)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/permissions/{permissionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":       c.fileId,
		"permissionId": c.permissionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.permissions.update" call.
// Exactly one of *Permission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Permission.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PermissionsUpdateCall) Do(opts ...googleapi.CallOption) (*Permission, error) {
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
	ret := &Permission{
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
	//   "description": "Updates a permission with patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.permissions.update",
	//   "parameterOrder": [
	//     "fileId",
	//     "permissionId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file or Team Drive.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "permissionId": {
	//       "description": "The ID of the permission.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "removeExpiration": {
	//       "default": "false",
	//       "description": "Whether to remove the expiration date.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "supportsTeamDrives": {
	//       "default": "false",
	//       "description": "Whether the requesting application supports Team Drives.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "transferOwnership": {
	//       "default": "false",
	//       "description": "Whether to transfer ownership to the specified user and downgrade the current owner to a writer. This parameter is required as an acknowledgement of the side effect.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{fileId}/permissions/{permissionId}",
	//   "request": {
	//     "$ref": "Permission"
	//   },
	//   "response": {
	//     "$ref": "Permission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.replies.create":

type RepliesCreateCall struct {
	s          *Service
	fileId     string
	commentId  string
	reply      *Reply
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new reply to a comment.
func (r *RepliesService) Create(fileId string, commentId string, reply *Reply) *RepliesCreateCall {
	c := &RepliesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	c.reply = reply
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RepliesCreateCall) Fields(s ...googleapi.Field) *RepliesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RepliesCreateCall) Context(ctx context.Context) *RepliesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RepliesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RepliesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reply)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}/replies")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.replies.create" call.
// Exactly one of *Reply or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Reply.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *RepliesCreateCall) Do(opts ...googleapi.CallOption) (*Reply, error) {
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
	ret := &Reply{
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
	//   "description": "Creates a new reply to a comment.",
	//   "httpMethod": "POST",
	//   "id": "drive.replies.create",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}/replies",
	//   "request": {
	//     "$ref": "Reply"
	//   },
	//   "response": {
	//     "$ref": "Reply"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.replies.delete":

type RepliesDeleteCall struct {
	s          *Service
	fileId     string
	commentId  string
	replyId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a reply.
func (r *RepliesService) Delete(fileId string, commentId string, replyId string) *RepliesDeleteCall {
	c := &RepliesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	c.replyId = replyId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RepliesDeleteCall) Fields(s ...googleapi.Field) *RepliesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RepliesDeleteCall) Context(ctx context.Context) *RepliesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RepliesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RepliesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}/replies/{replyId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
		"replyId":   c.replyId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.replies.delete" call.
func (c *RepliesDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a reply.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.replies.delete",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId",
	//     "replyId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "replyId": {
	//       "description": "The ID of the reply.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}/replies/{replyId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.replies.get":

type RepliesGetCall struct {
	s            *Service
	fileId       string
	commentId    string
	replyId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a reply by ID.
func (r *RepliesService) Get(fileId string, commentId string, replyId string) *RepliesGetCall {
	c := &RepliesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	c.replyId = replyId
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": Whether
// to return deleted replies. Deleted replies will not include their
// original content.
func (c *RepliesGetCall) IncludeDeleted(includeDeleted bool) *RepliesGetCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RepliesGetCall) Fields(s ...googleapi.Field) *RepliesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RepliesGetCall) IfNoneMatch(entityTag string) *RepliesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RepliesGetCall) Context(ctx context.Context) *RepliesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RepliesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RepliesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}/replies/{replyId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
		"replyId":   c.replyId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.replies.get" call.
// Exactly one of *Reply or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Reply.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *RepliesGetCall) Do(opts ...googleapi.CallOption) (*Reply, error) {
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
	ret := &Reply{
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
	//   "description": "Gets a reply by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.replies.get",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId",
	//     "replyId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "default": "false",
	//       "description": "Whether to return deleted replies. Deleted replies will not include their original content.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "replyId": {
	//       "description": "The ID of the reply.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}/replies/{replyId}",
	//   "response": {
	//     "$ref": "Reply"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.replies.list":

type RepliesListCall struct {
	s            *Service
	fileId       string
	commentId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists a comment's replies.
func (r *RepliesService) List(fileId string, commentId string) *RepliesListCall {
	c := &RepliesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": Whether
// to include deleted replies. Deleted replies will not include their
// original content.
func (c *RepliesListCall) IncludeDeleted(includeDeleted bool) *RepliesListCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of replies to return per page.
func (c *RepliesListCall) PageSize(pageSize int64) *RepliesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token for
// continuing a previous list request on the next page. This should be
// set to the value of 'nextPageToken' from the previous response.
func (c *RepliesListCall) PageToken(pageToken string) *RepliesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RepliesListCall) Fields(s ...googleapi.Field) *RepliesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RepliesListCall) IfNoneMatch(entityTag string) *RepliesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RepliesListCall) Context(ctx context.Context) *RepliesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RepliesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RepliesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}/replies")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.replies.list" call.
// Exactly one of *ReplyList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReplyList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *RepliesListCall) Do(opts ...googleapi.CallOption) (*ReplyList, error) {
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
	ret := &ReplyList{
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
	//   "description": "Lists a comment's replies.",
	//   "httpMethod": "GET",
	//   "id": "drive.replies.list",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "default": "false",
	//       "description": "Whether to include deleted replies. Deleted replies will not include their original content.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageSize": {
	//       "default": "20",
	//       "description": "The maximum number of replies to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}/replies",
	//   "response": {
	//     "$ref": "ReplyList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RepliesListCall) Pages(ctx context.Context, f func(*ReplyList) error) error {
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

// method id "drive.replies.update":

type RepliesUpdateCall struct {
	s          *Service
	fileId     string
	commentId  string
	replyId    string
	reply      *Reply
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a reply with patch semantics.
func (r *RepliesService) Update(fileId string, commentId string, replyId string, reply *Reply) *RepliesUpdateCall {
	c := &RepliesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.commentId = commentId
	c.replyId = replyId
	c.reply = reply
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RepliesUpdateCall) Fields(s ...googleapi.Field) *RepliesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RepliesUpdateCall) Context(ctx context.Context) *RepliesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RepliesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RepliesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reply)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/comments/{commentId}/replies/{replyId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":    c.fileId,
		"commentId": c.commentId,
		"replyId":   c.replyId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.replies.update" call.
// Exactly one of *Reply or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Reply.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *RepliesUpdateCall) Do(opts ...googleapi.CallOption) (*Reply, error) {
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
	ret := &Reply{
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
	//   "description": "Updates a reply with patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.replies.update",
	//   "parameterOrder": [
	//     "fileId",
	//     "commentId",
	//     "replyId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "replyId": {
	//       "description": "The ID of the reply.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/comments/{commentId}/replies/{replyId}",
	//   "request": {
	//     "$ref": "Reply"
	//   },
	//   "response": {
	//     "$ref": "Reply"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.revisions.delete":

type RevisionsDeleteCall struct {
	s          *Service
	fileId     string
	revisionId string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Permanently deletes a revision. This method is only
// applicable to files with binary content in Drive.
func (r *RevisionsService) Delete(fileId string, revisionId string) *RevisionsDeleteCall {
	c := &RevisionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.revisionId = revisionId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RevisionsDeleteCall) Fields(s ...googleapi.Field) *RevisionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RevisionsDeleteCall) Context(ctx context.Context) *RevisionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RevisionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RevisionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/revisions/{revisionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":     c.fileId,
		"revisionId": c.revisionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.revisions.delete" call.
func (c *RevisionsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Permanently deletes a revision. This method is only applicable to files with binary content in Drive.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.revisions.delete",
	//   "parameterOrder": [
	//     "fileId",
	//     "revisionId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "revisionId": {
	//       "description": "The ID of the revision.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/revisions/{revisionId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.revisions.get":

type RevisionsGetCall struct {
	s            *Service
	fileId       string
	revisionId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a revision's metadata or content by ID.
func (r *RevisionsService) Get(fileId string, revisionId string) *RevisionsGetCall {
	c := &RevisionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.revisionId = revisionId
	return c
}

// AcknowledgeAbuse sets the optional parameter "acknowledgeAbuse":
// Whether the user is acknowledging the risk of downloading known
// malware or other abusive files. This is only applicable when
// alt=media.
func (c *RevisionsGetCall) AcknowledgeAbuse(acknowledgeAbuse bool) *RevisionsGetCall {
	c.urlParams_.Set("acknowledgeAbuse", fmt.Sprint(acknowledgeAbuse))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RevisionsGetCall) Fields(s ...googleapi.Field) *RevisionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RevisionsGetCall) IfNoneMatch(entityTag string) *RevisionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do and Download
// methods. Any pending HTTP request will be aborted if the provided
// context is canceled.
func (c *RevisionsGetCall) Context(ctx context.Context) *RevisionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RevisionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RevisionsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/revisions/{revisionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":     c.fileId,
		"revisionId": c.revisionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Download fetches the API endpoint's "media" value, instead of the normal
// API response value. If the returned error is nil, the Response is guaranteed to
// have a 2xx status code. Callers must close the Response.Body as usual.
func (c *RevisionsGetCall) Download(opts ...googleapi.CallOption) (*http.Response, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("media")
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckMediaResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

// Do executes the "drive.revisions.get" call.
// Exactly one of *Revision or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Revision.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *RevisionsGetCall) Do(opts ...googleapi.CallOption) (*Revision, error) {
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
	ret := &Revision{
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
	//   "description": "Gets a revision's metadata or content by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.revisions.get",
	//   "parameterOrder": [
	//     "fileId",
	//     "revisionId"
	//   ],
	//   "parameters": {
	//     "acknowledgeAbuse": {
	//       "default": "false",
	//       "description": "Whether the user is acknowledging the risk of downloading known malware or other abusive files. This is only applicable when alt=media.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "revisionId": {
	//       "description": "The ID of the revision.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/revisions/{revisionId}",
	//   "response": {
	//     "$ref": "Revision"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ],
	//   "supportsMediaDownload": true,
	//   "useMediaDownloadService": true
	// }

}

// method id "drive.revisions.list":

type RevisionsListCall struct {
	s            *Service
	fileId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists a file's revisions.
func (r *RevisionsService) List(fileId string) *RevisionsListCall {
	c := &RevisionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of revisions to return per page.
func (c *RevisionsListCall) PageSize(pageSize int64) *RevisionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token for
// continuing a previous list request on the next page. This should be
// set to the value of 'nextPageToken' from the previous response.
func (c *RevisionsListCall) PageToken(pageToken string) *RevisionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RevisionsListCall) Fields(s ...googleapi.Field) *RevisionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RevisionsListCall) IfNoneMatch(entityTag string) *RevisionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RevisionsListCall) Context(ctx context.Context) *RevisionsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RevisionsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RevisionsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/revisions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId": c.fileId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.revisions.list" call.
// Exactly one of *RevisionList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *RevisionList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *RevisionsListCall) Do(opts ...googleapi.CallOption) (*RevisionList, error) {
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
	ret := &RevisionList{
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
	//   "description": "Lists a file's revisions.",
	//   "httpMethod": "GET",
	//   "id": "drive.revisions.list",
	//   "parameterOrder": [
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "default": "200",
	//       "description": "The maximum number of revisions to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token for continuing a previous list request on the next page. This should be set to the value of 'nextPageToken' from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/revisions",
	//   "response": {
	//     "$ref": "RevisionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.metadata",
	//     "https://www.googleapis.com/auth/drive.metadata.readonly",
	//     "https://www.googleapis.com/auth/drive.photos.readonly",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RevisionsListCall) Pages(ctx context.Context, f func(*RevisionList) error) error {
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

// method id "drive.revisions.update":

type RevisionsUpdateCall struct {
	s          *Service
	fileId     string
	revisionId string
	revision   *Revision
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a revision with patch semantics.
func (r *RevisionsService) Update(fileId string, revisionId string, revision *Revision) *RevisionsUpdateCall {
	c := &RevisionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.fileId = fileId
	c.revisionId = revisionId
	c.revision = revision
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RevisionsUpdateCall) Fields(s ...googleapi.Field) *RevisionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RevisionsUpdateCall) Context(ctx context.Context) *RevisionsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RevisionsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RevisionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.revision)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{fileId}/revisions/{revisionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"fileId":     c.fileId,
		"revisionId": c.revisionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.revisions.update" call.
// Exactly one of *Revision or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Revision.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *RevisionsUpdateCall) Do(opts ...googleapi.CallOption) (*Revision, error) {
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
	ret := &Revision{
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
	//   "description": "Updates a revision with patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.revisions.update",
	//   "parameterOrder": [
	//     "fileId",
	//     "revisionId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the file.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "revisionId": {
	//       "description": "The ID of the revision.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "files/{fileId}/revisions/{revisionId}",
	//   "request": {
	//     "$ref": "Revision"
	//   },
	//   "response": {
	//     "$ref": "Revision"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.appdata",
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.teamdrives.create":

type TeamdrivesCreateCall struct {
	s          *Service
	teamdrive  *TeamDrive
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new Team Drive.
func (r *TeamdrivesService) Create(requestId string, teamdrive *TeamDrive) *TeamdrivesCreateCall {
	c := &TeamdrivesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("requestId", requestId)
	c.teamdrive = teamdrive
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamdrivesCreateCall) Fields(s ...googleapi.Field) *TeamdrivesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamdrivesCreateCall) Context(ctx context.Context) *TeamdrivesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TeamdrivesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TeamdrivesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.teamdrive)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teamdrives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.teamdrives.create" call.
// Exactly one of *TeamDrive or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TeamDrive.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TeamdrivesCreateCall) Do(opts ...googleapi.CallOption) (*TeamDrive, error) {
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
	ret := &TeamDrive{
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
	//   "description": "Creates a new Team Drive.",
	//   "httpMethod": "POST",
	//   "id": "drive.teamdrives.create",
	//   "parameterOrder": [
	//     "requestId"
	//   ],
	//   "parameters": {
	//     "requestId": {
	//       "description": "An ID, such as a random UUID, which uniquely identifies this user's request for idempotent creation of a Team Drive. A repeated request by the same user and with the same request ID will avoid creating duplicates by attempting to create the same Team Drive. If the Team Drive already exists a 409 error will be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "teamdrives",
	//   "request": {
	//     "$ref": "TeamDrive"
	//   },
	//   "response": {
	//     "$ref": "TeamDrive"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive"
	//   ]
	// }

}

// method id "drive.teamdrives.delete":

type TeamdrivesDeleteCall struct {
	s           *Service
	teamDriveId string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Delete: Permanently deletes a Team Drive for which the user is an
// organizer. The Team Drive cannot contain any untrashed items.
func (r *TeamdrivesService) Delete(teamDriveId string) *TeamdrivesDeleteCall {
	c := &TeamdrivesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamDriveId = teamDriveId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamdrivesDeleteCall) Fields(s ...googleapi.Field) *TeamdrivesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamdrivesDeleteCall) Context(ctx context.Context) *TeamdrivesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TeamdrivesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TeamdrivesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teamdrives/{teamDriveId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"teamDriveId": c.teamDriveId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.teamdrives.delete" call.
func (c *TeamdrivesDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Permanently deletes a Team Drive for which the user is an organizer. The Team Drive cannot contain any untrashed items.",
	//   "httpMethod": "DELETE",
	//   "id": "drive.teamdrives.delete",
	//   "parameterOrder": [
	//     "teamDriveId"
	//   ],
	//   "parameters": {
	//     "teamDriveId": {
	//       "description": "The ID of the Team Drive",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "teamdrives/{teamDriveId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive"
	//   ]
	// }

}

// method id "drive.teamdrives.get":

type TeamdrivesGetCall struct {
	s            *Service
	teamDriveId  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Team Drive's metadata by ID.
func (r *TeamdrivesService) Get(teamDriveId string) *TeamdrivesGetCall {
	c := &TeamdrivesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamDriveId = teamDriveId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamdrivesGetCall) Fields(s ...googleapi.Field) *TeamdrivesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TeamdrivesGetCall) IfNoneMatch(entityTag string) *TeamdrivesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamdrivesGetCall) Context(ctx context.Context) *TeamdrivesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TeamdrivesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TeamdrivesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "teamdrives/{teamDriveId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"teamDriveId": c.teamDriveId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.teamdrives.get" call.
// Exactly one of *TeamDrive or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TeamDrive.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TeamdrivesGetCall) Do(opts ...googleapi.CallOption) (*TeamDrive, error) {
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
	ret := &TeamDrive{
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
	//   "description": "Gets a Team Drive's metadata by ID.",
	//   "httpMethod": "GET",
	//   "id": "drive.teamdrives.get",
	//   "parameterOrder": [
	//     "teamDriveId"
	//   ],
	//   "parameters": {
	//     "teamDriveId": {
	//       "description": "The ID of the Team Drive",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "teamdrives/{teamDriveId}",
	//   "response": {
	//     "$ref": "TeamDrive"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// method id "drive.teamdrives.list":

type TeamdrivesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the user's Team Drives.
func (r *TeamdrivesService) List() *TeamdrivesListCall {
	c := &TeamdrivesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// Team Drives to return.
func (c *TeamdrivesListCall) PageSize(pageSize int64) *TeamdrivesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token for
// Team Drives.
func (c *TeamdrivesListCall) PageToken(pageToken string) *TeamdrivesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamdrivesListCall) Fields(s ...googleapi.Field) *TeamdrivesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TeamdrivesListCall) IfNoneMatch(entityTag string) *TeamdrivesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamdrivesListCall) Context(ctx context.Context) *TeamdrivesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TeamdrivesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TeamdrivesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "teamdrives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.teamdrives.list" call.
// Exactly one of *TeamDriveList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TeamDriveList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TeamdrivesListCall) Do(opts ...googleapi.CallOption) (*TeamDriveList, error) {
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
	ret := &TeamDriveList{
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
	//   "description": "Lists the user's Team Drives.",
	//   "httpMethod": "GET",
	//   "id": "drive.teamdrives.list",
	//   "parameters": {
	//     "pageSize": {
	//       "default": "10",
	//       "description": "Maximum number of Team Drives to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token for Team Drives.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "teamdrives",
	//   "response": {
	//     "$ref": "TeamDriveList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TeamdrivesListCall) Pages(ctx context.Context, f func(*TeamDriveList) error) error {
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

// method id "drive.teamdrives.update":

type TeamdrivesUpdateCall struct {
	s           *Service
	teamDriveId string
	teamdrive   *TeamDrive
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Update: Updates a Team Drive's metadata
func (r *TeamdrivesService) Update(teamDriveId string, teamdrive *TeamDrive) *TeamdrivesUpdateCall {
	c := &TeamdrivesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamDriveId = teamDriveId
	c.teamdrive = teamdrive
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamdrivesUpdateCall) Fields(s ...googleapi.Field) *TeamdrivesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamdrivesUpdateCall) Context(ctx context.Context) *TeamdrivesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TeamdrivesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TeamdrivesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.teamdrive)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teamdrives/{teamDriveId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"teamDriveId": c.teamDriveId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "drive.teamdrives.update" call.
// Exactly one of *TeamDrive or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TeamDrive.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TeamdrivesUpdateCall) Do(opts ...googleapi.CallOption) (*TeamDrive, error) {
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
	ret := &TeamDrive{
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
	//   "description": "Updates a Team Drive's metadata",
	//   "httpMethod": "PATCH",
	//   "id": "drive.teamdrives.update",
	//   "parameterOrder": [
	//     "teamDriveId"
	//   ],
	//   "parameters": {
	//     "teamDriveId": {
	//       "description": "The ID of the Team Drive",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "teamdrives/{teamDriveId}",
	//   "request": {
	//     "$ref": "TeamDrive"
	//   },
	//   "response": {
	//     "$ref": "TeamDrive"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive"
	//   ]
	// }

}
