// Package storage provides access to the Cloud Storage API.
//
// See https://developers.google.com/storage/docs/json_api/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/storage/v1beta1"
//   ...
//   storageService, err := storage.New(oauthHttpClient)
package storage

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

const apiId = "storage:v1beta1"
const apiName = "storage"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/storage/v1beta1/"

// OAuth2 scopes used by this API.
const (
	// Manage your data and permissions in Google Cloud Storage
	DevstorageFull_controlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageRead_onlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageRead_writeScope = "https://www.googleapis.com/auth/devstorage.read_write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.BucketAccessControls = NewBucketAccessControlsService(s)
	s.Buckets = NewBucketsService(s)
	s.ObjectAccessControls = NewObjectAccessControlsService(s)
	s.Objects = NewObjectsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	BucketAccessControls *BucketAccessControlsService

	Buckets *BucketsService

	ObjectAccessControls *ObjectAccessControlsService

	Objects *ObjectsService
}

func NewBucketAccessControlsService(s *Service) *BucketAccessControlsService {
	rs := &BucketAccessControlsService{s: s}
	return rs
}

type BucketAccessControlsService struct {
	s *Service
}

func NewBucketsService(s *Service) *BucketsService {
	rs := &BucketsService{s: s}
	return rs
}

type BucketsService struct {
	s *Service
}

func NewObjectAccessControlsService(s *Service) *ObjectAccessControlsService {
	rs := &ObjectAccessControlsService{s: s}
	return rs
}

type ObjectAccessControlsService struct {
	s *Service
}

func NewObjectsService(s *Service) *ObjectsService {
	rs := &ObjectsService{s: s}
	return rs
}

type ObjectsService struct {
	s *Service
}

type Bucket struct {
	// Acl: Access controls on the bucket.
	Acl []*BucketAccessControl `json:"acl,omitempty"`

	// DefaultObjectAcl: Default access controls to apply to new objects
	// when no ACL is provided.
	DefaultObjectAcl []*ObjectAccessControl `json:"defaultObjectAcl,omitempty"`

	// Id: The name of the bucket.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For buckets, this is always
	// storage#bucket.
	Kind string `json:"kind,omitempty"`

	// Location: The location of the bucket. Object data for objects in the
	// bucket resides in physical storage in this location. Can be US or EU.
	// Defaults to US.
	Location string `json:"location,omitempty"`

	// Owner: The owner of the bucket. This will always be the project
	// team's owner group.
	Owner *BucketOwner `json:"owner,omitempty"`

	// ProjectId: The project the bucket belongs to.
	ProjectId uint64 `json:"projectId,omitempty,string"`

	// SelfLink: The URI of this bucket.
	SelfLink string `json:"selfLink,omitempty"`

	// TimeCreated: Creation time of the bucket in RFC 3339 format.
	TimeCreated string `json:"timeCreated,omitempty"`

	// Website: The bucket's website configuration.
	Website *BucketWebsite `json:"website,omitempty"`
}

type BucketOwner struct {
	// Entity: The entity, in the form group-groupId.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity.
	EntityId string `json:"entityId,omitempty"`
}

type BucketWebsite struct {
	// MainPageSuffix: Behaves as the bucket's directory index where missing
	// objects are treated as potential directories.
	MainPageSuffix string `json:"mainPageSuffix,omitempty"`

	// NotFoundPage: The custom object to return when a requested resource
	// is not found.
	NotFoundPage string `json:"notFoundPage,omitempty"`
}

type BucketAccessControl struct {
	// Bucket: The name of the bucket.
	Bucket string `json:"bucket,omitempty"`

	// Domain: The domain associated with the entity, if any.
	Domain string `json:"domain,omitempty"`

	// Email: The email address associated with the entity, if any.
	Email string `json:"email,omitempty"`

	// Entity: The entity holding the permission, in one of the following
	// forms:
	// - user-userId
	// - user-email
	// - group-groupId
	// - group-email
	//
	// - domain-domain
	// - allUsers
	// - allAuthenticatedUsers Examples:
	// -
	// The user liz@example.com would be user-liz@example.com.
	// - The group
	// example@googlegroups.com would be group-example@googlegroups.com.
	// -
	// To refer to all members of the Google Apps for Business domain
	// example.com, the entity would be domain-example.com.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity, if any.
	EntityId string `json:"entityId,omitempty"`

	// Id: The ID of the access-control entry.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For bucket access control entries,
	// this is always storage#bucketAccessControl.
	Kind string `json:"kind,omitempty"`

	// Role: The access permission for the entity. Can be READER, WRITER, or
	// OWNER.
	Role string `json:"role,omitempty"`

	// SelfLink: The link to this access-control entry.
	SelfLink string `json:"selfLink,omitempty"`
}

type BucketAccessControls struct {
	// Items: The list of items.
	Items []*BucketAccessControl `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of bucket access control
	// entries, this is always storage#bucketAccessControls.
	Kind string `json:"kind,omitempty"`
}

type Buckets struct {
	// Items: The list of items.
	Items []*Bucket `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of buckets, this is always
	// storage#buckets.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Object struct {
	// Acl: Access controls on the object.
	Acl []*ObjectAccessControl `json:"acl,omitempty"`

	// Bucket: The bucket containing this object.
	Bucket string `json:"bucket,omitempty"`

	// CacheControl: Cache-Control directive for the object data.
	CacheControl string `json:"cacheControl,omitempty"`

	// ContentDisposition: Content-Disposition of the object data.
	ContentDisposition string `json:"contentDisposition,omitempty"`

	// ContentEncoding: Content-Encoding of the object data.
	ContentEncoding string `json:"contentEncoding,omitempty"`

	// ContentLanguage: Content-Language of the object data.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// Id: The ID of the object.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For objects, this is always
	// storage#object.
	Kind string `json:"kind,omitempty"`

	// Media: Object media data. Provided on your behalf when uploading raw
	// media or multipart/related with an auxiliary media part.
	Media *ObjectMedia `json:"media,omitempty"`

	// Metadata: User-provided metadata, in key/value pairs.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Name: The name of this object. Required if not specified by URL
	// parameter.
	Name string `json:"name,omitempty"`

	// Owner: The owner of the object. This will always be the uploader of
	// the object.
	Owner *ObjectOwner `json:"owner,omitempty"`

	// SelfLink: The link to this object.
	SelfLink string `json:"selfLink,omitempty"`
}

type ObjectMedia struct {
	// Algorithm: Hash algorithm used. Currently only MD5 is supported.
	// Required if a hash is provided.
	Algorithm string `json:"algorithm,omitempty"`

	// ContentType: Content-Type of the object data.
	ContentType string `json:"contentType,omitempty"`

	// Data: URL-safe Base64-encoded data. This property can be used to
	// insert objects under 64KB in size, and will only be returned in
	// response to the get method for objects so created. When this resource
	// is returned in response to the list method, this property is omitted.
	Data string `json:"data,omitempty"`

	// Hash: Hash of the data. Required if a hash algorithm is provided.
	Hash string `json:"hash,omitempty"`

	// Length: Content-Length of the data in bytes.
	Length uint64 `json:"length,omitempty,string"`

	// Link: Media download link.
	Link string `json:"link,omitempty"`

	// TimeCreated: Creation time of the data in RFC 3339 format.
	TimeCreated string `json:"timeCreated,omitempty"`
}

type ObjectOwner struct {
	// Entity: The entity, in the form user-userId.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity.
	EntityId string `json:"entityId,omitempty"`
}

type ObjectAccessControl struct {
	// Bucket: The name of the bucket.
	Bucket string `json:"bucket,omitempty"`

	// Domain: The domain associated with the entity, if any.
	Domain string `json:"domain,omitempty"`

	// Email: The email address associated with the entity, if any.
	Email string `json:"email,omitempty"`

	// Entity: The entity holding the permission, in one of the following
	// forms:
	// - user-userId
	// - user-email
	// - group-groupId
	// - group-email
	//
	// - domain-domain
	// - allUsers
	// - allAuthenticatedUsers Examples:
	// -
	// The user liz@example.com would be user-liz@example.com.
	// - The group
	// example@googlegroups.com would be group-example@googlegroups.com.
	// -
	// To refer to all members of the Google Apps for Business domain
	// example.com, the entity would be domain-example.com.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity, if any.
	EntityId string `json:"entityId,omitempty"`

	// Id: The ID of the access-control entry.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For object access control entries,
	// this is always storage#objectAccessControl.
	Kind string `json:"kind,omitempty"`

	// Object: The name of the object.
	Object string `json:"object,omitempty"`

	// Role: The access permission for the entity. Can be READER or OWNER.
	Role string `json:"role,omitempty"`

	// SelfLink: The link to this access-control entry.
	SelfLink string `json:"selfLink,omitempty"`
}

type ObjectAccessControls struct {
	// Items: The list of items.
	Items []*ObjectAccessControl `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of object access control
	// entries, this is always storage#objectAccessControls.
	Kind string `json:"kind,omitempty"`
}

type Objects struct {
	// Items: The list of items.
	Items []*Object `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of objects, this is always
	// storage#objects.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Prefixes: The list of prefixes of objects matching-but-not-listed up
	// to and including the requested delimiter.
	Prefixes []string `json:"prefixes,omitempty"`
}

// method id "storage.bucketAccessControls.delete":

type BucketAccessControlsDeleteCall struct {
	s      *Service
	bucket string
	entity string
	opt_   map[string]interface{}
}

// Delete: Deletes the ACL entry for the specified entity on the
// specified bucket.
func (r *BucketAccessControlsService) Delete(bucket string, entity string) *BucketAccessControlsDeleteCall {
	c := &BucketAccessControlsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	return c
}

func (c *BucketAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the ACL entry for the specified entity on the specified bucket.",
	//   "httpMethod": "DELETE",
	//   "id": "storage.bucketAccessControls.delete",
	//   "parameterOrder": [
	//     "bucket",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl/{entity}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.bucketAccessControls.get":

type BucketAccessControlsGetCall struct {
	s      *Service
	bucket string
	entity string
	opt_   map[string]interface{}
}

// Get: Returns the ACL entry for the specified entity on the specified
// bucket.
func (r *BucketAccessControlsService) Get(bucket string, entity string) *BucketAccessControlsGetCall {
	c := &BucketAccessControlsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	return c
}

func (c *BucketAccessControlsGetCall) Do() (*BucketAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(BucketAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the ACL entry for the specified entity on the specified bucket.",
	//   "httpMethod": "GET",
	//   "id": "storage.bucketAccessControls.get",
	//   "parameterOrder": [
	//     "bucket",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl/{entity}",
	//   "response": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.bucketAccessControls.insert":

type BucketAccessControlsInsertCall struct {
	s                   *Service
	bucket              string
	bucketaccesscontrol *BucketAccessControl
	opt_                map[string]interface{}
}

// Insert: Creates a new ACL entry on the specified bucket.
func (r *BucketAccessControlsService) Insert(bucket string, bucketaccesscontrol *BucketAccessControl) *BucketAccessControlsInsertCall {
	c := &BucketAccessControlsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.bucketaccesscontrol = bucketaccesscontrol
	return c
}

func (c *BucketAccessControlsInsertCall) Do() (*BucketAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucketaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(BucketAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new ACL entry on the specified bucket.",
	//   "httpMethod": "POST",
	//   "id": "storage.bucketAccessControls.insert",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl",
	//   "request": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "response": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.bucketAccessControls.list":

type BucketAccessControlsListCall struct {
	s      *Service
	bucket string
	opt_   map[string]interface{}
}

// List: Retrieves ACL entries on the specified bucket.
func (r *BucketAccessControlsService) List(bucket string) *BucketAccessControlsListCall {
	c := &BucketAccessControlsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

func (c *BucketAccessControlsListCall) Do() (*BucketAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(BucketAccessControls)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves ACL entries on the specified bucket.",
	//   "httpMethod": "GET",
	//   "id": "storage.bucketAccessControls.list",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl",
	//   "response": {
	//     "$ref": "BucketAccessControls"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.bucketAccessControls.patch":

type BucketAccessControlsPatchCall struct {
	s                   *Service
	bucket              string
	entity              string
	bucketaccesscontrol *BucketAccessControl
	opt_                map[string]interface{}
}

// Patch: Updates an ACL entry on the specified bucket. This method
// supports patch semantics.
func (r *BucketAccessControlsService) Patch(bucket string, entity string, bucketaccesscontrol *BucketAccessControl) *BucketAccessControlsPatchCall {
	c := &BucketAccessControlsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	c.bucketaccesscontrol = bucketaccesscontrol
	return c
}

func (c *BucketAccessControlsPatchCall) Do() (*BucketAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucketaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(BucketAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an ACL entry on the specified bucket. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "storage.bucketAccessControls.patch",
	//   "parameterOrder": [
	//     "bucket",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl/{entity}",
	//   "request": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "response": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.bucketAccessControls.update":

type BucketAccessControlsUpdateCall struct {
	s                   *Service
	bucket              string
	entity              string
	bucketaccesscontrol *BucketAccessControl
	opt_                map[string]interface{}
}

// Update: Updates an ACL entry on the specified bucket.
func (r *BucketAccessControlsService) Update(bucket string, entity string, bucketaccesscontrol *BucketAccessControl) *BucketAccessControlsUpdateCall {
	c := &BucketAccessControlsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	c.bucketaccesscontrol = bucketaccesscontrol
	return c
}

func (c *BucketAccessControlsUpdateCall) Do() (*BucketAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucketaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(BucketAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an ACL entry on the specified bucket.",
	//   "httpMethod": "PUT",
	//   "id": "storage.bucketAccessControls.update",
	//   "parameterOrder": [
	//     "bucket",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/acl/{entity}",
	//   "request": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "response": {
	//     "$ref": "BucketAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.buckets.delete":

type BucketsDeleteCall struct {
	s      *Service
	bucket string
	opt_   map[string]interface{}
}

// Delete: Deletes an empty bucket.
func (r *BucketsService) Delete(bucket string) *BucketsDeleteCall {
	c := &BucketsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

func (c *BucketsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes an empty bucket.",
	//   "httpMethod": "DELETE",
	//   "id": "storage.buckets.delete",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.buckets.get":

type BucketsGetCall struct {
	s      *Service
	bucket string
	opt_   map[string]interface{}
}

// Get: Returns metadata for the specified bucket.
func (r *BucketsService) Get(bucket string) *BucketsGetCall {
	c := &BucketsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl.
func (c *BucketsGetCall) Projection(projection string) *BucketsGetCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsGetCall) Do() (*Bucket, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(Bucket)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns metadata for the specified bucket.",
	//   "httpMethod": "GET",
	//   "id": "storage.buckets.get",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit acl and defaultObjectAcl properties."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}",
	//   "response": {
	//     "$ref": "Bucket"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.buckets.insert":

type BucketsInsertCall struct {
	s      *Service
	bucket *Bucket
	opt_   map[string]interface{}
}

// Insert: Creates a new bucket.
func (r *BucketsService) Insert(bucket *Bucket) *BucketsInsertCall {
	c := &BucketsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl, unless the bucket resource
// specifies acl or defaultObjectAcl properties, when it defaults to
// full.
func (c *BucketsInsertCall) Projection(projection string) *BucketsInsertCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsInsertCall) Do() (*Bucket, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucket)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(Bucket)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new bucket.",
	//   "httpMethod": "POST",
	//   "id": "storage.buckets.insert",
	//   "parameters": {
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl, unless the bucket resource specifies acl or defaultObjectAcl properties, when it defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit acl and defaultObjectAcl properties."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b",
	//   "request": {
	//     "$ref": "Bucket"
	//   },
	//   "response": {
	//     "$ref": "Bucket"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.buckets.list":

type BucketsListCall struct {
	s         *Service
	projectId uint64
	opt_      map[string]interface{}
}

// List: Retrieves a list of buckets for a given project.
func (r *BucketsService) List(projectId uint64) *BucketsListCall {
	c := &BucketsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of buckets to return.
func (c *BucketsListCall) MaxResults(maxResults int64) *BucketsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *BucketsListCall) PageToken(pageToken string) *BucketsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl.
func (c *BucketsListCall) Projection(projection string) *BucketsListCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsListCall) Do() (*Buckets, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("projectId", fmt.Sprintf("%v", c.projectId))
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(Buckets)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of buckets for a given project.",
	//   "httpMethod": "GET",
	//   "id": "storage.buckets.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "max-results": {
	//       "description": "Maximum number of buckets to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A previously-returned page token representing part of the larger set of results to view.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "A valid API project identifier.",
	//       "format": "uint64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit acl and defaultObjectAcl properties."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b",
	//   "response": {
	//     "$ref": "Buckets"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.buckets.patch":

type BucketsPatchCall struct {
	s       *Service
	bucket  string
	bucket2 *Bucket
	opt_    map[string]interface{}
}

// Patch: Updates a bucket. This method supports patch semantics.
func (r *BucketsService) Patch(bucket string, bucket2 *Bucket) *BucketsPatchCall {
	c := &BucketsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.bucket2 = bucket2
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
func (c *BucketsPatchCall) Projection(projection string) *BucketsPatchCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsPatchCall) Do() (*Bucket, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucket2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(Bucket)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a bucket. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "storage.buckets.patch",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit acl and defaultObjectAcl properties."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}",
	//   "request": {
	//     "$ref": "Bucket"
	//   },
	//   "response": {
	//     "$ref": "Bucket"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.buckets.update":

type BucketsUpdateCall struct {
	s       *Service
	bucket  string
	bucket2 *Bucket
	opt_    map[string]interface{}
}

// Update: Updates a bucket.
func (r *BucketsService) Update(bucket string, bucket2 *Bucket) *BucketsUpdateCall {
	c := &BucketsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.bucket2 = bucket2
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
func (c *BucketsUpdateCall) Projection(projection string) *BucketsUpdateCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsUpdateCall) Do() (*Bucket, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bucket2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(Bucket)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a bucket.",
	//   "httpMethod": "PUT",
	//   "id": "storage.buckets.update",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit acl and defaultObjectAcl properties."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}",
	//   "request": {
	//     "$ref": "Bucket"
	//   },
	//   "response": {
	//     "$ref": "Bucket"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.objectAccessControls.delete":

type ObjectAccessControlsDeleteCall struct {
	s      *Service
	bucket string
	object string
	entity string
	opt_   map[string]interface{}
}

// Delete: Deletes the ACL entry for the specified entity on the
// specified object.
func (r *ObjectAccessControlsService) Delete(bucket string, object string, entity string) *ObjectAccessControlsDeleteCall {
	c := &ObjectAccessControlsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.entity = entity
	return c
}

func (c *ObjectAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the ACL entry for the specified entity on the specified object.",
	//   "httpMethod": "DELETE",
	//   "id": "storage.objectAccessControls.delete",
	//   "parameterOrder": [
	//     "bucket",
	//     "object",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl/{entity}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objectAccessControls.get":

type ObjectAccessControlsGetCall struct {
	s      *Service
	bucket string
	object string
	entity string
	opt_   map[string]interface{}
}

// Get: Returns the ACL entry for the specified entity on the specified
// object.
func (r *ObjectAccessControlsService) Get(bucket string, object string, entity string) *ObjectAccessControlsGetCall {
	c := &ObjectAccessControlsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.entity = entity
	return c
}

func (c *ObjectAccessControlsGetCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the ACL entry for the specified entity on the specified object.",
	//   "httpMethod": "GET",
	//   "id": "storage.objectAccessControls.get",
	//   "parameterOrder": [
	//     "bucket",
	//     "object",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl/{entity}",
	//   "response": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objectAccessControls.insert":

type ObjectAccessControlsInsertCall struct {
	s                   *Service
	bucket              string
	object              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Insert: Creates a new ACL entry on the specified object.
func (r *ObjectAccessControlsService) Insert(bucket string, object string, objectaccesscontrol *ObjectAccessControl) *ObjectAccessControlsInsertCall {
	c := &ObjectAccessControlsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *ObjectAccessControlsInsertCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new ACL entry on the specified object.",
	//   "httpMethod": "POST",
	//   "id": "storage.objectAccessControls.insert",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl",
	//   "request": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "response": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objectAccessControls.list":

type ObjectAccessControlsListCall struct {
	s      *Service
	bucket string
	object string
	opt_   map[string]interface{}
}

// List: Retrieves ACL entries on the specified object.
func (r *ObjectAccessControlsService) List(bucket string, object string) *ObjectAccessControlsListCall {
	c := &ObjectAccessControlsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

func (c *ObjectAccessControlsListCall) Do() (*ObjectAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
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
	ret := new(ObjectAccessControls)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves ACL entries on the specified object.",
	//   "httpMethod": "GET",
	//   "id": "storage.objectAccessControls.list",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl",
	//   "response": {
	//     "$ref": "ObjectAccessControls"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objectAccessControls.patch":

type ObjectAccessControlsPatchCall struct {
	s                   *Service
	bucket              string
	object              string
	entity              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Patch: Updates an ACL entry on the specified object. This method
// supports patch semantics.
func (r *ObjectAccessControlsService) Patch(bucket string, object string, entity string, objectaccesscontrol *ObjectAccessControl) *ObjectAccessControlsPatchCall {
	c := &ObjectAccessControlsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.entity = entity
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *ObjectAccessControlsPatchCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an ACL entry on the specified object. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "storage.objectAccessControls.patch",
	//   "parameterOrder": [
	//     "bucket",
	//     "object",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl/{entity}",
	//   "request": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "response": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objectAccessControls.update":

type ObjectAccessControlsUpdateCall struct {
	s                   *Service
	bucket              string
	object              string
	entity              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Update: Updates an ACL entry on the specified object.
func (r *ObjectAccessControlsService) Update(bucket string, object string, entity string, objectaccesscontrol *ObjectAccessControl) *ObjectAccessControlsUpdateCall {
	c := &ObjectAccessControlsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.entity = entity
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *ObjectAccessControlsUpdateCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{entity}", url.QueryEscape(c.entity), 1)
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an ACL entry on the specified object.",
	//   "httpMethod": "PUT",
	//   "id": "storage.objectAccessControls.update",
	//   "parameterOrder": [
	//     "bucket",
	//     "object",
	//     "entity"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of a bucket.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entity": {
	//       "description": "The entity holding the permission. Can be user-userId, user-emailAddress, group-groupId, group-emailAddress, allUsers, or allAuthenticatedUsers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}/acl/{entity}",
	//   "request": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "response": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.objects.delete":

type ObjectsDeleteCall struct {
	s      *Service
	bucket string
	object string
	opt_   map[string]interface{}
}

// Delete: Deletes data blobs and associated metadata.
func (r *ObjectsService) Delete(bucket string, object string) *ObjectsDeleteCall {
	c := &ObjectsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

func (c *ObjectsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes data blobs and associated metadata.",
	//   "httpMethod": "DELETE",
	//   "id": "storage.objects.delete",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which the object resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.objects.get":

type ObjectsGetCall struct {
	s      *Service
	bucket string
	object string
	opt_   map[string]interface{}
}

// Get: Retrieves objects or their associated metadata.
func (r *ObjectsService) Get(bucket string, object string) *ObjectsGetCall {
	c := &ObjectsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl.
func (c *ObjectsGetCall) Projection(projection string) *ObjectsGetCall {
	c.opt_["projection"] = projection
	return c
}

func (c *ObjectsGetCall) Do() (*Object, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
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
	ret := new(Object)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves objects or their associated metadata.",
	//   "httpMethod": "GET",
	//   "id": "storage.objects.get",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which the object resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}",
	//   "response": {
	//     "$ref": "Object"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "storage.objects.insert":

type ObjectsInsertCall struct {
	s      *Service
	bucket string
	object *Object
	opt_   map[string]interface{}
	media_ io.Reader
}

// Insert: Stores new data blobs and associated metadata.
func (r *ObjectsService) Insert(bucket string, object *Object) *ObjectsInsertCall {
	c := &ObjectsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// Name sets the optional parameter "name": Name of the object. Required
// when the object metadata is not otherwise provided. Overrides the
// object metadata's name value, if any.
func (c *ObjectsInsertCall) Name(name string) *ObjectsInsertCall {
	c.opt_["name"] = name
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl, unless the object resource
// specifies the acl property, when it defaults to full.
func (c *ObjectsInsertCall) Projection(projection string) *ObjectsInsertCall {
	c.opt_["projection"] = projection
	return c
}
func (c *ObjectsInsertCall) Media(r io.Reader) *ObjectsInsertCall {
	c.media_ = r
	return c
}

func (c *ObjectsInsertCall) Do() (*Object, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.object)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["name"]; ok {
		params.Set("name", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(Object)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Stores new data blobs and associated metadata.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/storage/v1beta1/b/{bucket}/o"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/storage/v1beta1/b/{bucket}/o"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which to store the new object. Overrides the provided object metadata's bucket value, if any.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Name of the object. Required when the object metadata is not otherwise provided. Overrides the object metadata's name value, if any.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl, unless the object resource specifies the acl property, when it defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o",
	//   "request": {
	//     "$ref": "Object"
	//   },
	//   "response": {
	//     "$ref": "Object"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsMediaDownload": true,
	//   "supportsMediaUpload": true
	// }

}

// method id "storage.objects.list":

type ObjectsListCall struct {
	s      *Service
	bucket string
	opt_   map[string]interface{}
}

// List: Retrieves a list of objects matching the criteria.
func (r *ObjectsService) List(bucket string) *ObjectsListCall {
	c := &ObjectsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

// Delimiter sets the optional parameter "delimiter": Returns results in
// a directory-like mode. items will contain only objects whose names,
// aside from the prefix, do not contain delimiter. Objects whose names,
// aside from the prefix, contain delimiter will have their name,
// truncated after the delimiter, returned in prefixes. Duplicate
// prefixes are omitted.
func (c *ObjectsListCall) Delimiter(delimiter string) *ObjectsListCall {
	c.opt_["delimiter"] = delimiter
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of items plus prefixes to return. As duplicate prefixes are omitted,
// fewer total results may be returned than requested.
func (c *ObjectsListCall) MaxResults(maxResults int64) *ObjectsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *ObjectsListCall) PageToken(pageToken string) *ObjectsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Prefix sets the optional parameter "prefix": Filter results to
// objects whose names begin with this prefix.
func (c *ObjectsListCall) Prefix(prefix string) *ObjectsListCall {
	c.opt_["prefix"] = prefix
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to no_acl.
func (c *ObjectsListCall) Projection(projection string) *ObjectsListCall {
	c.opt_["projection"] = projection
	return c
}

func (c *ObjectsListCall) Do() (*Objects, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["delimiter"]; ok {
		params.Set("delimiter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["prefix"]; ok {
		params.Set("prefix", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
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
	ret := new(Objects)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of objects matching the criteria.",
	//   "httpMethod": "GET",
	//   "id": "storage.objects.list",
	//   "parameterOrder": [
	//     "bucket"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which to look for objects.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "delimiter": {
	//       "description": "Returns results in a directory-like mode. items will contain only objects whose names, aside from the prefix, do not contain delimiter. Objects whose names, aside from the prefix, contain delimiter will have their name, truncated after the delimiter, returned in prefixes. Duplicate prefixes are omitted.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "Maximum number of items plus prefixes to return. As duplicate prefixes are omitted, fewer total results may be returned than requested.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A previously-returned page token representing part of the larger set of results to view.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "prefix": {
	//       "description": "Filter results to objects whose names begin with this prefix.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to no_acl.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o",
	//   "response": {
	//     "$ref": "Objects"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsSubscription": true
	// }

}

// method id "storage.objects.patch":

type ObjectsPatchCall struct {
	s       *Service
	bucket  string
	object  string
	object2 *Object
	opt_    map[string]interface{}
}

// Patch: Updates a data blob's associated metadata. This method
// supports patch semantics.
func (r *ObjectsService) Patch(bucket string, object string, object2 *Object) *ObjectsPatchCall {
	c := &ObjectsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.object2 = object2
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
func (c *ObjectsPatchCall) Projection(projection string) *ObjectsPatchCall {
	c.opt_["projection"] = projection
	return c
}

func (c *ObjectsPatchCall) Do() (*Object, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.object2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
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
	ret := new(Object)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a data blob's associated metadata. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "storage.objects.patch",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which the object resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}",
	//   "request": {
	//     "$ref": "Object"
	//   },
	//   "response": {
	//     "$ref": "Object"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.objects.update":

type ObjectsUpdateCall struct {
	s       *Service
	bucket  string
	object  string
	object2 *Object
	opt_    map[string]interface{}
}

// Update: Updates a data blob's associated metadata.
func (r *ObjectsService) Update(bucket string, object string, object2 *Object) *ObjectsUpdateCall {
	c := &ObjectsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.object2 = object2
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
func (c *ObjectsUpdateCall) Projection(projection string) *ObjectsUpdateCall {
	c.opt_["projection"] = projection
	return c
}

func (c *ObjectsUpdateCall) Do() (*Object, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.object2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{bucket}", url.QueryEscape(c.bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{object}", url.QueryEscape(c.object), 1)
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
	ret := new(Object)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a data blob's associated metadata.",
	//   "httpMethod": "PUT",
	//   "id": "storage.objects.update",
	//   "parameterOrder": [
	//     "bucket",
	//     "object"
	//   ],
	//   "parameters": {
	//     "bucket": {
	//       "description": "Name of the bucket in which the object resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "no_acl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/o/{object}",
	//   "request": {
	//     "$ref": "Object"
	//   },
	//   "response": {
	//     "$ref": "Object"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsMediaDownload": true
	// }

}
