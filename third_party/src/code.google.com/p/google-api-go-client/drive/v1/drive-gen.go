// Package drive provides access to the Drive API.
//
// See https://developers.google.com/drive/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/drive/v1"
//   ...
//   driveService, err := drive.New(oauthHttpClient)
package drive

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "drive:v1"
const apiName = "drive"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/drive/v1/"

// OAuth2 scopes used by this API.
const (
	// View and manage Google Drive files that you have opened or created
	// with this app
	DriveFileScope = "https://www.googleapis.com/auth/drive.file"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Files = NewFilesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Files *FilesService
}

func NewFilesService(s *Service) *FilesService {
	rs := &FilesService{s: s}
	return rs
}

type FilesService struct {
	s *Service
}

type File struct {
	// CreatedDate: Create time for this file (formatted ISO8601 timestamp).
	CreatedDate string `json:"createdDate,omitempty"`

	// Description: A short description of the file
	Description string `json:"description,omitempty"`

	// DownloadUrl: Short term download URL for the file. This will only be
	// populated on files with content stored in Drive.
	DownloadUrl string `json:"downloadUrl,omitempty"`

	// Etag: ETag of the file.
	Etag string `json:"etag,omitempty"`

	// FileExtension: The file extension used when downloading this file.
	// This field is read only. To set the extension, include it on title
	// when creating the file. This will only be populated on files with
	// content stored in Drive.
	FileExtension string `json:"fileExtension,omitempty"`

	// FileSize: The size of the file in bytes. This will only be populated
	// on files with content stored in Drive.
	FileSize int64 `json:"fileSize,omitempty,string"`

	// Id: The id of the file.
	Id string `json:"id,omitempty"`

	// IndexableText: Indexable text attributes for the file (can only be
	// written)
	IndexableText *FileIndexableText `json:"indexableText,omitempty"`

	// Kind: The type of file. This is always drive#file
	Kind string `json:"kind,omitempty"`

	// Labels: Labels for the file.
	Labels *FileLabels `json:"labels,omitempty"`

	// LastViewedDate: Last time this file was viewed by the user (formatted
	// RFC 3339 timestamp).
	LastViewedDate string `json:"lastViewedDate,omitempty"`

	// Md5Checksum: An MD5 checksum for the content of this file. This will
	// only be populated on files with content stored in Drive.
	Md5Checksum string `json:"md5Checksum,omitempty"`

	// MimeType: The mimetype of the file
	MimeType string `json:"mimeType,omitempty"`

	// ModifiedByMeDate: Last time this file was modified by the user
	// (formatted RFC 3339 timestamp).
	ModifiedByMeDate string `json:"modifiedByMeDate,omitempty"`

	// ModifiedDate: Last time this file was modified by anyone (formatted
	// RFC 3339 timestamp).
	ModifiedDate string `json:"modifiedDate,omitempty"`

	// ParentsCollection: Collection of parent folders which contain this
	// file.
	// On insert, setting this field will put the file in all of the
	// provided folders. If no folders are provided, the file will be placed
	// in the default root folder. On update, this field is ignored.
	ParentsCollection []*FileParentsCollection `json:"parentsCollection,omitempty"`

	// SelfLink: A link back to this file.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: The title of this file.
	Title string `json:"title,omitempty"`

	// UserPermission: The permissions for the authenticated user on this
	// file.
	UserPermission *Permission `json:"userPermission,omitempty"`
}

type FileIndexableText struct {
	// Text: The text to be indexed for this file
	Text string `json:"text,omitempty"`
}

type FileLabels struct {
	// Hidden: Whether this file is hidden from the user
	Hidden bool `json:"hidden,omitempty"`

	// Starred: Whether this file is starred by the user.
	Starred bool `json:"starred,omitempty"`

	// Trashed: Whether this file has been trashed.
	Trashed bool `json:"trashed,omitempty"`
}

type FileParentsCollection struct {
	// Id: The id of this parent
	Id string `json:"id,omitempty"`

	// ParentLink: A link to get the metadata for this parent
	ParentLink string `json:"parentLink,omitempty"`
}

type Permission struct {
	// AdditionalRoles: Any additional roles that this permission describes.
	AdditionalRoles []string `json:"additionalRoles,omitempty"`

	// Etag: An etag for this permission.
	Etag string `json:"etag,omitempty"`

	// Kind: The kind of this permission. This is always drive#permission
	Kind string `json:"kind,omitempty"`

	// Role: The role that this permission describes. (For example: reader,
	// writer, owner)
	Role string `json:"role,omitempty"`

	// Type: The type of permission (For example: user, group etc).
	Type string `json:"type,omitempty"`
}

// method id "drive.files.get":

type FilesGetCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Get: Gets a file's metadata by id.
func (r *FilesService) Get(id string) *FilesGetCall {
	c := &FilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// Projection sets the optional parameter "projection": This parameter
// is deprecated and has no function.
func (c *FilesGetCall) Projection(projection string) *FilesGetCall {
	c.opt_["projection"] = projection
	return c
}

// UpdateViewedDate sets the optional parameter "updateViewedDate":
// Whether to update the view date after successfully retrieving the
// file.
func (c *FilesGetCall) UpdateViewedDate(updateViewedDate bool) *FilesGetCall {
	c.opt_["updateViewedDate"] = updateViewedDate
	return c
}

func (c *FilesGetCall) Do() (*File, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updateViewedDate"]; ok {
		params.Set("updateViewedDate", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a file's metadata by id.",
	//   "httpMethod": "GET",
	//   "id": "drive.files.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id for the file in question.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "This parameter is deprecated and has no function.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "enumDescriptions": [
	//         "Deprecated",
	//         "Deprecated"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updateViewedDate": {
	//       "default": "true",
	//       "description": "Whether to update the view date after successfully retrieving the file.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{id}",
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.files.insert":

type FilesInsertCall struct {
	s      *Service
	file   *File
	opt_   map[string]interface{}
	media_ io.Reader
}

// Insert: Inserts a file, and any settable metadata or blob content
// sent with the request.
func (r *FilesService) Insert(file *File) *FilesInsertCall {
	c := &FilesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.file = file
	return c
}
func (c *FilesInsertCall) Media(r io.Reader) *FilesInsertCall {
	c.media_ = r
	return c
}

func (c *FilesInsertCall) Do() (*File, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "files")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a file, and any settable metadata or blob content sent with the request.",
	//   "httpMethod": "POST",
	//   "id": "drive.files.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "maxSize": "1024GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/drive/v1/files"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/drive/v1/files"
	//       }
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
	//     "https://www.googleapis.com/auth/drive.file"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "drive.files.patch":

type FilesPatchCall struct {
	s    *Service
	id   string
	file *File
	opt_ map[string]interface{}
}

// Patch: Updates file metadata and/or content. This method supports
// patch semantics.
func (r *FilesService) Patch(id string, file *File) *FilesPatchCall {
	c := &FilesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.file = file
	return c
}

// NewRevision sets the optional parameter "newRevision": Whether a blob
// upload should create a new revision. If false, the blob data in the
// current head revision is replaced. If not set or true, a new blob is
// created as head revision, and previous revisions are preserved
// (causing increased use of the user's data storage quota).
func (c *FilesPatchCall) NewRevision(newRevision bool) *FilesPatchCall {
	c.opt_["newRevision"] = newRevision
	return c
}

// UpdateModifiedDate sets the optional parameter "updateModifiedDate":
// Controls updating the modified date of the file. If true, the
// modified date will be updated to the current time, regardless of
// whether other changes are being made. If false, the modified date
// will only be updated to the current time if other changes are also
// being made (changing the title, for example).
func (c *FilesPatchCall) UpdateModifiedDate(updateModifiedDate bool) *FilesPatchCall {
	c.opt_["updateModifiedDate"] = updateModifiedDate
	return c
}

// UpdateViewedDate sets the optional parameter "updateViewedDate":
// Whether to update the view date after successfully updating the file.
func (c *FilesPatchCall) UpdateViewedDate(updateViewedDate bool) *FilesPatchCall {
	c.opt_["updateViewedDate"] = updateViewedDate
	return c
}

func (c *FilesPatchCall) Do() (*File, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["newRevision"]; ok {
		params.Set("newRevision", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updateModifiedDate"]; ok {
		params.Set("updateModifiedDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updateViewedDate"]; ok {
		params.Set("updateViewedDate", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates file metadata and/or content. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "drive.files.patch",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id for the file in question.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "newRevision": {
	//       "default": "true",
	//       "description": "Whether a blob upload should create a new revision. If false, the blob data in the current head revision is replaced. If not set or true, a new blob is created as head revision, and previous revisions are preserved (causing increased use of the user's data storage quota).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "updateModifiedDate": {
	//       "default": "false",
	//       "description": "Controls updating the modified date of the file. If true, the modified date will be updated to the current time, regardless of whether other changes are being made. If false, the modified date will only be updated to the current time if other changes are also being made (changing the title, for example).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "updateViewedDate": {
	//       "default": "true",
	//       "description": "Whether to update the view date after successfully updating the file.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{id}",
	//   "request": {
	//     "$ref": "File"
	//   },
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive.file"
	//   ]
	// }

}

// method id "drive.files.update":

type FilesUpdateCall struct {
	s      *Service
	id     string
	file   *File
	opt_   map[string]interface{}
	media_ io.Reader
}

// Update: Updates file metadata and/or content
func (r *FilesService) Update(id string, file *File) *FilesUpdateCall {
	c := &FilesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.file = file
	return c
}

// NewRevision sets the optional parameter "newRevision": Whether a blob
// upload should create a new revision. If false, the blob data in the
// current head revision is replaced. If not set or true, a new blob is
// created as head revision, and previous revisions are preserved
// (causing increased use of the user's data storage quota).
func (c *FilesUpdateCall) NewRevision(newRevision bool) *FilesUpdateCall {
	c.opt_["newRevision"] = newRevision
	return c
}

// UpdateModifiedDate sets the optional parameter "updateModifiedDate":
// Controls updating the modified date of the file. If true, the
// modified date will be updated to the current time, regardless of
// whether other changes are being made. If false, the modified date
// will only be updated to the current time if other changes are also
// being made (changing the title, for example).
func (c *FilesUpdateCall) UpdateModifiedDate(updateModifiedDate bool) *FilesUpdateCall {
	c.opt_["updateModifiedDate"] = updateModifiedDate
	return c
}

// UpdateViewedDate sets the optional parameter "updateViewedDate":
// Whether to update the view date after successfully updating the file.
func (c *FilesUpdateCall) UpdateViewedDate(updateViewedDate bool) *FilesUpdateCall {
	c.opt_["updateViewedDate"] = updateViewedDate
	return c
}
func (c *FilesUpdateCall) Media(r io.Reader) *FilesUpdateCall {
	c.media_ = r
	return c
}

func (c *FilesUpdateCall) Do() (*File, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.file)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["newRevision"]; ok {
		params.Set("newRevision", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updateModifiedDate"]; ok {
		params.Set("updateModifiedDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updateViewedDate"]; ok {
		params.Set("updateViewedDate", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "files/{id}")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates file metadata and/or content",
	//   "httpMethod": "PUT",
	//   "id": "drive.files.update",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "maxSize": "1024GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/drive/v1/files/{id}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/drive/v1/files/{id}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id for the file in question.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "newRevision": {
	//       "default": "true",
	//       "description": "Whether a blob upload should create a new revision. If false, the blob data in the current head revision is replaced. If not set or true, a new blob is created as head revision, and previous revisions are preserved (causing increased use of the user's data storage quota).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "updateModifiedDate": {
	//       "default": "false",
	//       "description": "Controls updating the modified date of the file. If true, the modified date will be updated to the current time, regardless of whether other changes are being made. If false, the modified date will only be updated to the current time if other changes are also being made (changing the title, for example).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "updateViewedDate": {
	//       "default": "true",
	//       "description": "Whether to update the view date after successfully updating the file.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "files/{id}",
	//   "request": {
	//     "$ref": "File"
	//   },
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive.file"
	//   ],
	//   "supportsMediaUpload": true
	// }

}
