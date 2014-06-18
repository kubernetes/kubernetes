// Package storage provides access to the Cloud Storage API.
//
// See https://developers.google.com/storage/docs/json_api/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/storage/v1beta2"
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

const apiId = "storage:v1beta2"
const apiName = "storage"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/storage/v1beta2/"

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
	s.Channels = NewChannelsService(s)
	s.DefaultObjectAccessControls = NewDefaultObjectAccessControlsService(s)
	s.ObjectAccessControls = NewObjectAccessControlsService(s)
	s.Objects = NewObjectsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	BucketAccessControls *BucketAccessControlsService

	Buckets *BucketsService

	Channels *ChannelsService

	DefaultObjectAccessControls *DefaultObjectAccessControlsService

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

func NewChannelsService(s *Service) *ChannelsService {
	rs := &ChannelsService{s: s}
	return rs
}

type ChannelsService struct {
	s *Service
}

func NewDefaultObjectAccessControlsService(s *Service) *DefaultObjectAccessControlsService {
	rs := &DefaultObjectAccessControlsService{s: s}
	return rs
}

type DefaultObjectAccessControlsService struct {
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

	// Cors: The bucket's Cross-Origin Resource Sharing (CORS)
	// configuration.
	Cors []*BucketCors `json:"cors,omitempty"`

	// DefaultObjectAcl: Default access controls to apply to new objects
	// when no ACL is provided.
	DefaultObjectAcl []*ObjectAccessControl `json:"defaultObjectAcl,omitempty"`

	// Etag: HTTP 1.1 Entity tag for the bucket.
	Etag string `json:"etag,omitempty"`

	// Id: The ID of the bucket.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For buckets, this is always
	// storage#bucket.
	Kind string `json:"kind,omitempty"`

	// Lifecycle: The bucket's lifecycle configuration. See object lifecycle
	// management for more information.
	Lifecycle *BucketLifecycle `json:"lifecycle,omitempty"`

	// Location: The location of the bucket. Object data for objects in the
	// bucket resides in physical storage within this region. Typical values
	// are US and EU. Defaults to US. See the developer's guide for the
	// authoritative list.
	Location string `json:"location,omitempty"`

	// Logging: The bucket's logging configuration, which defines the
	// destination bucket and optional name prefix for the current bucket's
	// logs.
	Logging *BucketLogging `json:"logging,omitempty"`

	// Metageneration: The metadata generation of this bucket.
	Metageneration int64 `json:"metageneration,omitempty,string"`

	// Name: The name of the bucket.
	Name string `json:"name,omitempty"`

	// Owner: The owner of the bucket. This will always be the project
	// team's owner group.
	Owner *BucketOwner `json:"owner,omitempty"`

	// SelfLink: The URI of this bucket.
	SelfLink string `json:"selfLink,omitempty"`

	// StorageClass: The bucket's storage class. This defines how objects in
	// the bucket will be stored and determines the SLA and the cost of
	// storage. Typical values are STANDARD and
	// DURABLE_REDUCED_AVAILABILITY. Defaults to STANDARD. See the
	// developer's guide for the authoritative list.
	StorageClass string `json:"storageClass,omitempty"`

	// TimeCreated: Creation time of the bucket in RFC 3339 format.
	TimeCreated string `json:"timeCreated,omitempty"`

	// Versioning: The bucket's versioning configuration.
	Versioning *BucketVersioning `json:"versioning,omitempty"`

	// Website: The bucket's website configuration.
	Website *BucketWebsite `json:"website,omitempty"`
}

type BucketCors struct {
	// MaxAgeSeconds: The value, in seconds, to return in the
	// Access-Control-Max-Age header used in preflight responses.
	MaxAgeSeconds int64 `json:"maxAgeSeconds,omitempty"`

	// Method: The list of HTTP methods on which to include CORS response
	// headers, e.g. GET, OPTIONS, POST. Note, "*" is permitted in the list
	// of methods, and means "any method".
	Method []string `json:"method,omitempty"`

	// Origin: The list of Origins eligible to receive CORS response
	// headers. Note: "*" is permitted in the list of origins, and means
	// "any Origin".
	Origin []string `json:"origin,omitempty"`

	// ResponseHeader: The list of HTTP headers other than the simple
	// response headers to give permission for the user-agent to share
	// across domains.
	ResponseHeader []string `json:"responseHeader,omitempty"`
}

type BucketLifecycle struct {
	// Rule: A lifecycle management rule, which is made of an action to take
	// and the condition(s) under which the action will be taken.
	Rule []*BucketLifecycleRule `json:"rule,omitempty"`
}

type BucketLifecycleRule struct {
	// Action: The action to take.
	Action *BucketLifecycleRuleAction `json:"action,omitempty"`

	// Condition: The condition(s) under which the action will be taken.
	Condition *BucketLifecycleRuleCondition `json:"condition,omitempty"`
}

type BucketLifecycleRuleAction struct {
	// Type: Type of the action, e.g. Delete.
	Type string `json:"type,omitempty"`
}

type BucketLifecycleRuleCondition struct {
	// Age: Age of an object (in days). This condition is satisfied when an
	// object reaches the specified age.
	Age int64 `json:"age,omitempty"`

	// CreatedBefore: A date in RFC 3339 format with only the date part,
	// e.g. "2013-01-15". This condition is satisfied when an object is
	// created before midnight of the specified date in UTC.
	CreatedBefore string `json:"createdBefore,omitempty"`

	// IsLive: Relevant only for versioned objects. If the value is true,
	// this condition matches live objects; if the value is false, it
	// matches archived objects.
	IsLive bool `json:"isLive,omitempty"`

	// NumNewerVersions: Relevant only for versioned objects. If the value
	// is N, this condition is satisfied when there are at least N versions
	// (including the live version) newer than this version of the object.
	NumNewerVersions int64 `json:"numNewerVersions,omitempty"`
}

type BucketLogging struct {
	// LogBucket: The destination bucket where the current bucket's logs
	// should be placed.
	LogBucket string `json:"logBucket,omitempty"`

	// LogObjectPrefix: A prefix for log object names.
	LogObjectPrefix string `json:"logObjectPrefix,omitempty"`
}

type BucketOwner struct {
	// Entity: The entity, in the form group-groupId.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity.
	EntityId string `json:"entityId,omitempty"`
}

type BucketVersioning struct {
	// Enabled: While set to true, versioning is fully enabled for this
	// bucket.
	Enabled bool `json:"enabled,omitempty"`
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

	// Etag: HTTP 1.1 Entity tag for the access-control entry.
	Etag string `json:"etag,omitempty"`

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
}

type ComposeRequest struct {
	// Destination: Properties of the resulting object
	Destination *Object `json:"destination,omitempty"`

	// Kind: The kind of item this is.
	Kind string `json:"kind,omitempty"`

	// SourceObjects: The list of source objects that will be concatenated
	// into a single object.
	SourceObjects []*ComposeRequestSourceObjects `json:"sourceObjects,omitempty"`
}

type ComposeRequestSourceObjects struct {
	// Generation: The generation of this object to use as the source.
	Generation int64 `json:"generation,omitempty,string"`

	// Name: The source object's name. The source object's bucket is
	// implicitly the destination bucket.
	Name string `json:"name,omitempty"`

	// ObjectPreconditions: Conditions that must be met for this operation
	// to execute.
	ObjectPreconditions *ComposeRequestSourceObjectsObjectPreconditions `json:"objectPreconditions,omitempty"`
}

type ComposeRequestSourceObjectsObjectPreconditions struct {
	// IfGenerationMatch: Only perform the composition if the generation of
	// the source object that would be used matches this value. If this
	// value and a generation are both specified, they must be the same
	// value or the call will fail.
	IfGenerationMatch int64 `json:"ifGenerationMatch,omitempty,string"`
}

type Object struct {
	// Acl: Access controls on the object.
	Acl []*ObjectAccessControl `json:"acl,omitempty"`

	// Bucket: The bucket containing this object.
	Bucket string `json:"bucket,omitempty"`

	// CacheControl: Cache-Control directive for the object data.
	CacheControl string `json:"cacheControl,omitempty"`

	// ComponentCount: Number of underlying components that make up this
	// object. Components are accumulated by compose operations and are
	// limited to a count of 32.
	ComponentCount int64 `json:"componentCount,omitempty"`

	// ContentDisposition: Content-Disposition of the object data.
	ContentDisposition string `json:"contentDisposition,omitempty"`

	// ContentEncoding: Content-Encoding of the object data.
	ContentEncoding string `json:"contentEncoding,omitempty"`

	// ContentLanguage: Content-Language of the object data.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// ContentType: Content-Type of the object data.
	ContentType string `json:"contentType,omitempty"`

	// Crc32c: CRC32c checksum, as described in RFC 4960, Appendix B;
	// encoded using base64.
	Crc32c string `json:"crc32c,omitempty"`

	// Etag: HTTP 1.1 Entity tag for the object.
	Etag string `json:"etag,omitempty"`

	// Generation: The content generation of this object. Used for object
	// versioning.
	Generation int64 `json:"generation,omitempty,string"`

	// Id: The ID of the object.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item this is. For objects, this is always
	// storage#object.
	Kind string `json:"kind,omitempty"`

	// Md5Hash: MD5 hash of the data; encoded using base64.
	Md5Hash string `json:"md5Hash,omitempty"`

	// MediaLink: Media download link.
	MediaLink string `json:"mediaLink,omitempty"`

	// Metadata: User-provided metadata, in key/value pairs.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Metageneration: The generation of the metadata for this object at
	// this generation. Used for metadata versioning. Has no meaning outside
	// of the context of this generation.
	Metageneration int64 `json:"metageneration,omitempty,string"`

	// Name: The name of this object. Required if not specified by URL
	// parameter.
	Name string `json:"name,omitempty"`

	// Owner: The owner of the object. This will always be the uploader of
	// the object.
	Owner *ObjectOwner `json:"owner,omitempty"`

	// SelfLink: The link to this object.
	SelfLink string `json:"selfLink,omitempty"`

	// Size: Content-Length of the data in bytes.
	Size uint64 `json:"size,omitempty,string"`

	// StorageClass: Storage class of the object.
	StorageClass string `json:"storageClass,omitempty"`

	// TimeDeleted: Deletion time of the object in RFC 3339 format. Will be
	// returned if and only if this version of the object has been deleted.
	TimeDeleted string `json:"timeDeleted,omitempty"`

	// Updated: Modification time of the object metadata in RFC 3339 format.
	Updated string `json:"updated,omitempty"`
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

	// Etag: HTTP 1.1 Entity tag for the access-control entry.
	Etag string `json:"etag,omitempty"`

	// Generation: The content generation of the object.
	Generation int64 `json:"generation,omitempty,string"`

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
	Items []interface{} `json:"items,omitempty"`

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

// Delete: Permanently deletes the ACL entry for the specified entity on
// the specified bucket.
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
	//   "description": "Permanently deletes the ACL entry for the specified entity on the specified bucket.",
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

// Delete: Permanently deletes an empty bucket.
func (r *BucketsService) Delete(bucket string) *BucketsDeleteCall {
	c := &BucketsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration matches
// the given value.
func (c *BucketsDeleteCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *BucketsDeleteCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsDeleteCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *BucketsDeleteCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

func (c *BucketsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//   "description": "Permanently deletes an empty bucket.",
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
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
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

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration matches
// the given value.
func (c *BucketsGetCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *BucketsGetCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsGetCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *BucketsGetCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
func (c *BucketsGetCall) Projection(projection string) *BucketsGetCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsGetCall) Do() (*Bucket, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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
	s         *Service
	projectid string
	bucket    *Bucket
	opt_      map[string]interface{}
}

// Insert: Creates a new bucket.
func (r *BucketsService) Insert(projectid string, bucket *Bucket) *BucketsInsertCall {
	c := &BucketsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectid = projectid
	c.bucket = bucket
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the bucket resource
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
	params.Set("project", fmt.Sprintf("%v", c.projectid))
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
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "A valid API project identifier.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl, unless the bucket resource specifies acl or defaultObjectAcl properties, when it defaults to full.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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
	projectid string
	opt_      map[string]interface{}
}

// List: Retrieves a list of buckets for a given project.
func (r *BucketsService) List(projectid string) *BucketsListCall {
	c := &BucketsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectid = projectid
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of buckets to return.
func (c *BucketsListCall) MaxResults(maxResults int64) *BucketsListCall {
	c.opt_["maxResults"] = maxResults
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
// properties to return. Defaults to noAcl.
func (c *BucketsListCall) Projection(projection string) *BucketsListCall {
	c.opt_["projection"] = projection
	return c
}

func (c *BucketsListCall) Do() (*Buckets, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("project", fmt.Sprintf("%v", c.projectid))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
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
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
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
	//     "project": {
	//       "description": "A valid API project identifier.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration matches
// the given value.
func (c *BucketsPatchCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *BucketsPatchCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsPatchCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *BucketsPatchCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
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
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration matches
// the given value.
func (c *BucketsUpdateCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *BucketsUpdateCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsUpdateCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *BucketsUpdateCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
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
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to full.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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

// method id "storage.channels.stop":

type ChannelsStopCall struct {
	s       *Service
	channel *Channel
	opt_    map[string]interface{}
}

// Stop: Stop watching resources through this channel
func (r *ChannelsService) Stop(channel *Channel) *ChannelsStopCall {
	c := &ChannelsStopCall{s: r.s, opt_: make(map[string]interface{})}
	c.channel = channel
	return c
}

func (c *ChannelsStopCall) Do() error {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "channels/stop")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
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
	//   "description": "Stop watching resources through this channel",
	//   "httpMethod": "POST",
	//   "id": "storage.channels.stop",
	//   "path": "channels/stop",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.defaultObjectAccessControls.delete":

type DefaultObjectAccessControlsDeleteCall struct {
	s      *Service
	bucket string
	entity string
	opt_   map[string]interface{}
}

// Delete: Permanently deletes the default object ACL entry for the
// specified entity on the specified bucket.
func (r *DefaultObjectAccessControlsService) Delete(bucket string, entity string) *DefaultObjectAccessControlsDeleteCall {
	c := &DefaultObjectAccessControlsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	return c
}

func (c *DefaultObjectAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
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
	//   "description": "Permanently deletes the default object ACL entry for the specified entity on the specified bucket.",
	//   "httpMethod": "DELETE",
	//   "id": "storage.defaultObjectAccessControls.delete",
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
	//   "path": "b/{bucket}/defaultObjectAcl/{entity}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.defaultObjectAccessControls.get":

type DefaultObjectAccessControlsGetCall struct {
	s      *Service
	bucket string
	entity string
	opt_   map[string]interface{}
}

// Get: Returns the default object ACL entry for the specified entity on
// the specified bucket.
func (r *DefaultObjectAccessControlsService) Get(bucket string, entity string) *DefaultObjectAccessControlsGetCall {
	c := &DefaultObjectAccessControlsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	return c
}

func (c *DefaultObjectAccessControlsGetCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the default object ACL entry for the specified entity on the specified bucket.",
	//   "httpMethod": "GET",
	//   "id": "storage.defaultObjectAccessControls.get",
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
	//   "path": "b/{bucket}/defaultObjectAcl/{entity}",
	//   "response": {
	//     "$ref": "ObjectAccessControl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.defaultObjectAccessControls.insert":

type DefaultObjectAccessControlsInsertCall struct {
	s                   *Service
	bucket              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Insert: Creates a new default object ACL entry on the specified
// bucket.
func (r *DefaultObjectAccessControlsService) Insert(bucket string, objectaccesscontrol *ObjectAccessControl) *DefaultObjectAccessControlsInsertCall {
	c := &DefaultObjectAccessControlsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *DefaultObjectAccessControlsInsertCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl")
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new default object ACL entry on the specified bucket.",
	//   "httpMethod": "POST",
	//   "id": "storage.defaultObjectAccessControls.insert",
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
	//   "path": "b/{bucket}/defaultObjectAcl",
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

// method id "storage.defaultObjectAccessControls.list":

type DefaultObjectAccessControlsListCall struct {
	s      *Service
	bucket string
	opt_   map[string]interface{}
}

// List: Retrieves default object ACL entries on the specified bucket.
func (r *DefaultObjectAccessControlsService) List(bucket string) *DefaultObjectAccessControlsListCall {
	c := &DefaultObjectAccessControlsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the destination object's current metageneration matches the given
// value.
func (c *DefaultObjectAccessControlsListCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *DefaultObjectAccessControlsListCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the destination object's current metageneration does not
// match the given value.
func (c *DefaultObjectAccessControlsListCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *DefaultObjectAccessControlsListCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

func (c *DefaultObjectAccessControlsListCall) Do() (*ObjectAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl")
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
	ret := new(ObjectAccessControls)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves default object ACL entries on the specified bucket.",
	//   "httpMethod": "GET",
	//   "id": "storage.defaultObjectAccessControls.list",
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
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}/defaultObjectAcl",
	//   "response": {
	//     "$ref": "ObjectAccessControls"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control"
	//   ]
	// }

}

// method id "storage.defaultObjectAccessControls.patch":

type DefaultObjectAccessControlsPatchCall struct {
	s                   *Service
	bucket              string
	entity              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Patch: Updates a default object ACL entry on the specified bucket.
// This method supports patch semantics.
func (r *DefaultObjectAccessControlsService) Patch(bucket string, entity string, objectaccesscontrol *ObjectAccessControl) *DefaultObjectAccessControlsPatchCall {
	c := &DefaultObjectAccessControlsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *DefaultObjectAccessControlsPatchCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a default object ACL entry on the specified bucket. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "storage.defaultObjectAccessControls.patch",
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
	//   "path": "b/{bucket}/defaultObjectAcl/{entity}",
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

// method id "storage.defaultObjectAccessControls.update":

type DefaultObjectAccessControlsUpdateCall struct {
	s                   *Service
	bucket              string
	entity              string
	objectaccesscontrol *ObjectAccessControl
	opt_                map[string]interface{}
}

// Update: Updates a default object ACL entry on the specified bucket.
func (r *DefaultObjectAccessControlsService) Update(bucket string, entity string, objectaccesscontrol *ObjectAccessControl) *DefaultObjectAccessControlsUpdateCall {
	c := &DefaultObjectAccessControlsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.entity = entity
	c.objectaccesscontrol = objectaccesscontrol
	return c
}

func (c *DefaultObjectAccessControlsUpdateCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.objectaccesscontrol)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
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
	ret := new(ObjectAccessControl)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a default object ACL entry on the specified bucket.",
	//   "httpMethod": "PUT",
	//   "id": "storage.defaultObjectAccessControls.update",
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
	//   "path": "b/{bucket}/defaultObjectAcl/{entity}",
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

// method id "storage.objectAccessControls.delete":

type ObjectAccessControlsDeleteCall struct {
	s      *Service
	bucket string
	object string
	entity string
	opt_   map[string]interface{}
}

// Delete: Permanently deletes the ACL entry for the specified entity on
// the specified object.
func (r *ObjectAccessControlsService) Delete(bucket string, object string, entity string) *ObjectAccessControlsDeleteCall {
	c := &ObjectAccessControlsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	c.entity = entity
	return c
}

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsDeleteCall) Generation(generation uint64) *ObjectAccessControlsDeleteCall {
	c.opt_["generation"] = generation
	return c
}

func (c *ObjectAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//   "description": "Permanently deletes the ACL entry for the specified entity on the specified object.",
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsGetCall) Generation(generation uint64) *ObjectAccessControlsGetCall {
	c.opt_["generation"] = generation
	return c
}

func (c *ObjectAccessControlsGetCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsInsertCall) Generation(generation uint64) *ObjectAccessControlsInsertCall {
	c.opt_["generation"] = generation
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
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsListCall) Generation(generation uint64) *ObjectAccessControlsListCall {
	c.opt_["generation"] = generation
	return c
}

func (c *ObjectAccessControlsListCall) Do() (*ObjectAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsPatchCall) Generation(generation uint64) *ObjectAccessControlsPatchCall {
	c.opt_["generation"] = generation
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
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectAccessControlsUpdateCall) Generation(generation uint64) *ObjectAccessControlsUpdateCall {
	c.opt_["generation"] = generation
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
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
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

// method id "storage.objects.compose":

type ObjectsComposeCall struct {
	s                 *Service
	destinationBucket string
	destinationObject string
	composerequest    *ComposeRequest
	opt_              map[string]interface{}
}

// Compose: Concatenates a list of existing objects into a new object in
// the same bucket.
func (r *ObjectsService) Compose(destinationBucket string, destinationObject string, composerequest *ComposeRequest) *ObjectsComposeCall {
	c := &ObjectsComposeCall{s: r.s, opt_: make(map[string]interface{})}
	c.destinationBucket = destinationBucket
	c.destinationObject = destinationObject
	c.composerequest = composerequest
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsComposeCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsComposeCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsComposeCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsComposeCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

func (c *ObjectsComposeCall) Do() (*Object, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.composerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{destinationBucket}/o/{destinationObject}/compose")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{destinationBucket}", url.QueryEscape(c.destinationBucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{destinationObject}", url.QueryEscape(c.destinationObject), 1)
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
	//   "description": "Concatenates a list of existing objects into a new object in the same bucket.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.compose",
	//   "parameterOrder": [
	//     "destinationBucket",
	//     "destinationObject"
	//   ],
	//   "parameters": {
	//     "destinationBucket": {
	//       "description": "Name of the bucket in which to store the new object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "destinationObject": {
	//       "description": "Name of the new object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{destinationBucket}/o/{destinationObject}/compose",
	//   "request": {
	//     "$ref": "ComposeRequest"
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

// method id "storage.objects.copy":

type ObjectsCopyCall struct {
	s                 *Service
	sourceBucket      string
	sourceObject      string
	destinationBucket string
	destinationObject string
	object            *Object
	opt_              map[string]interface{}
}

// Copy: Copies an object to a destination in the same location.
// Optionally overrides metadata.
func (r *ObjectsService) Copy(sourceBucket string, sourceObject string, destinationBucket string, destinationObject string, object *Object) *ObjectsCopyCall {
	c := &ObjectsCopyCall{s: r.s, opt_: make(map[string]interface{})}
	c.sourceBucket = sourceBucket
	c.sourceObject = sourceObject
	c.destinationBucket = destinationBucket
	c.destinationObject = destinationObject
	c.object = object
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the destination object's
// current generation matches the given value.
func (c *ObjectsCopyCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsCopyCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the destination object's current generation does not match the given
// value.
func (c *ObjectsCopyCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsCopyCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the destination object's current metageneration matches the given
// value.
func (c *ObjectsCopyCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsCopyCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the destination object's current metageneration does not
// match the given value.
func (c *ObjectsCopyCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsCopyCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// IfSourceGenerationMatch sets the optional parameter
// "ifSourceGenerationMatch": Makes the operation conditional on whether
// the source object's generation matches the given value.
func (c *ObjectsCopyCall) IfSourceGenerationMatch(ifSourceGenerationMatch uint64) *ObjectsCopyCall {
	c.opt_["ifSourceGenerationMatch"] = ifSourceGenerationMatch
	return c
}

// IfSourceGenerationNotMatch sets the optional parameter
// "ifSourceGenerationNotMatch": Makes the operation conditional on
// whether the source object's generation does not match the given
// value.
func (c *ObjectsCopyCall) IfSourceGenerationNotMatch(ifSourceGenerationNotMatch uint64) *ObjectsCopyCall {
	c.opt_["ifSourceGenerationNotMatch"] = ifSourceGenerationNotMatch
	return c
}

// IfSourceMetagenerationMatch sets the optional parameter
// "ifSourceMetagenerationMatch": Makes the operation conditional on
// whether the source object's current metageneration matches the given
// value.
func (c *ObjectsCopyCall) IfSourceMetagenerationMatch(ifSourceMetagenerationMatch uint64) *ObjectsCopyCall {
	c.opt_["ifSourceMetagenerationMatch"] = ifSourceMetagenerationMatch
	return c
}

// IfSourceMetagenerationNotMatch sets the optional parameter
// "ifSourceMetagenerationNotMatch": Makes the operation conditional on
// whether the source object's current metageneration does not match the
// given value.
func (c *ObjectsCopyCall) IfSourceMetagenerationNotMatch(ifSourceMetagenerationNotMatch uint64) *ObjectsCopyCall {
	c.opt_["ifSourceMetagenerationNotMatch"] = ifSourceMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the object resource
// specifies the acl property, when it defaults to full.
func (c *ObjectsCopyCall) Projection(projection string) *ObjectsCopyCall {
	c.opt_["projection"] = projection
	return c
}

// SourceGeneration sets the optional parameter "sourceGeneration": If
// present, selects a specific revision of the source object (as opposed
// to the latest version, the default).
func (c *ObjectsCopyCall) SourceGeneration(sourceGeneration uint64) *ObjectsCopyCall {
	c.opt_["sourceGeneration"] = sourceGeneration
	return c
}

func (c *ObjectsCopyCall) Do() (*Object, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.object)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifSourceGenerationMatch"]; ok {
		params.Set("ifSourceGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifSourceGenerationNotMatch"]; ok {
		params.Set("ifSourceGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifSourceMetagenerationMatch"]; ok {
		params.Set("ifSourceMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifSourceMetagenerationNotMatch"]; ok {
		params.Set("ifSourceMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sourceGeneration"]; ok {
		params.Set("sourceGeneration", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{sourceBucket}/o/{sourceObject}/copyTo/b/{destinationBucket}/o/{destinationObject}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{sourceBucket}", url.QueryEscape(c.sourceBucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{sourceObject}", url.QueryEscape(c.sourceObject), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{destinationBucket}", url.QueryEscape(c.destinationBucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{destinationObject}", url.QueryEscape(c.destinationObject), 1)
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
	//   "description": "Copies an object to a destination in the same location. Optionally overrides metadata.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.copy",
	//   "parameterOrder": [
	//     "sourceBucket",
	//     "sourceObject",
	//     "destinationBucket",
	//     "destinationObject"
	//   ],
	//   "parameters": {
	//     "destinationBucket": {
	//       "description": "Name of the bucket in which to store the new object. Overrides the provided object metadata's bucket value, if any.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "destinationObject": {
	//       "description": "Name of the new object. Required when the object metadata is not otherwise provided. Overrides the object metadata's name value, if any.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl, unless the object resource specifies the acl property, when it defaults to full.",
	//       "enum": [
	//         "full",
	//         "noAcl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sourceBucket": {
	//       "description": "Name of the bucket in which to find the source object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sourceGeneration": {
	//       "description": "If present, selects a specific revision of the source object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sourceObject": {
	//       "description": "Name of the source object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{sourceBucket}/o/{sourceObject}/copyTo/b/{destinationBucket}/o/{destinationObject}",
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

// method id "storage.objects.delete":

type ObjectsDeleteCall struct {
	s      *Service
	bucket string
	object string
	opt_   map[string]interface{}
}

// Delete: Deletes data blobs and associated metadata. Deletions are
// permanent if versioning is not enabled for the bucket, or if the
// generation parameter is used.
func (r *ObjectsService) Delete(bucket string, object string) *ObjectsDeleteCall {
	c := &ObjectsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// Generation sets the optional parameter "generation": If present,
// permanently deletes a specific revision of this object (as opposed to
// the latest version, the default).
func (c *ObjectsDeleteCall) Generation(generation uint64) *ObjectsDeleteCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsDeleteCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsDeleteCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsDeleteCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsDeleteCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsDeleteCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsDeleteCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsDeleteCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsDeleteCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

func (c *ObjectsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//   "description": "Deletes data blobs and associated metadata. Deletions are permanent if versioning is not enabled for the bucket, or if the generation parameter is used.",
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
	//     "generation": {
	//       "description": "If present, permanently deletes a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectsGetCall) Generation(generation uint64) *ObjectsGetCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's generation
// matches the given value.
func (c *ObjectsGetCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsGetCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's generation does not match the given value.
func (c *ObjectsGetCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsGetCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsGetCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsGetCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsGetCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsGetCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
func (c *ObjectsGetCall) Projection(projection string) *ObjectsGetCall {
	c.opt_["projection"] = projection
	return c
}

func (c *ObjectsGetCall) Do() (*Object, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsInsertCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsInsertCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsInsertCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsInsertCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsInsertCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsInsertCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsInsertCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsInsertCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
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
// properties to return. Defaults to noAcl, unless the object resource
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
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//         "path": "/resumable/upload/storage/v1beta2/b/{bucket}/o"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/storage/v1beta2/b/{bucket}/o"
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
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Name of the object. Required when the object metadata is not otherwise provided. Overrides the object metadata's name value, if any.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Set of properties to return. Defaults to noAcl, unless the object resource specifies the acl property, when it defaults to full.",
	//       "enum": [
	//         "full",
	//         "noAcl"
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

// MaxResults sets the optional parameter "maxResults": Maximum number
// of items plus prefixes to return. As duplicate prefixes are omitted,
// fewer total results may be returned than requested.
func (c *ObjectsListCall) MaxResults(maxResults int64) *ObjectsListCall {
	c.opt_["maxResults"] = maxResults
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
// properties to return. Defaults to noAcl.
func (c *ObjectsListCall) Projection(projection string) *ObjectsListCall {
	c.opt_["projection"] = projection
	return c
}

// Versions sets the optional parameter "versions": If true, lists all
// versions of a file as distinct results.
func (c *ObjectsListCall) Versions(versions bool) *ObjectsListCall {
	c.opt_["versions"] = versions
	return c
}

func (c *ObjectsListCall) Do() (*Objects, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["delimiter"]; ok {
		params.Set("delimiter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["versions"]; ok {
		params.Set("versions", fmt.Sprintf("%v", v))
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
	//     "maxResults": {
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
	//       "description": "Set of properties to return. Defaults to noAcl.",
	//       "enum": [
	//         "full",
	//         "noAcl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "versions": {
	//       "description": "If true, lists all versions of a file as distinct results.",
	//       "location": "query",
	//       "type": "boolean"
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectsPatchCall) Generation(generation uint64) *ObjectsPatchCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsPatchCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsPatchCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsPatchCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsPatchCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsPatchCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsPatchCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsPatchCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsPatchCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
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
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
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
	//         "noAcl"
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

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectsUpdateCall) Generation(generation uint64) *ObjectsUpdateCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsUpdateCall) IfGenerationMatch(ifGenerationMatch uint64) *ObjectsUpdateCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsUpdateCall) IfGenerationNotMatch(ifGenerationNotMatch uint64) *ObjectsUpdateCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsUpdateCall) IfMetagenerationMatch(ifMetagenerationMatch uint64) *ObjectsUpdateCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsUpdateCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch uint64) *ObjectsUpdateCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
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
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationNotMatch"]; ok {
		params.Set("ifGenerationNotMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationNotMatch"]; ok {
		params.Set("ifMetagenerationNotMatch", fmt.Sprintf("%v", v))
	}
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
	//     "generation": {
	//       "description": "If present, selects a specific revision of this object (as opposed to the latest version, the default).",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "uint64",
	//       "location": "query",
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
	//         "noAcl"
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

// method id "storage.objects.watchAll":

type ObjectsWatchAllCall struct {
	s       *Service
	bucket  string
	channel *Channel
	opt_    map[string]interface{}
}

// WatchAll: Watch for changes on all objects in a bucket.
func (r *ObjectsService) WatchAll(bucket string, channel *Channel) *ObjectsWatchAllCall {
	c := &ObjectsWatchAllCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.channel = channel
	return c
}

// Delimiter sets the optional parameter "delimiter": Returns results in
// a directory-like mode. items will contain only objects whose names,
// aside from the prefix, do not contain delimiter. Objects whose names,
// aside from the prefix, contain delimiter will have their name,
// truncated after the delimiter, returned in prefixes. Duplicate
// prefixes are omitted.
func (c *ObjectsWatchAllCall) Delimiter(delimiter string) *ObjectsWatchAllCall {
	c.opt_["delimiter"] = delimiter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of items plus prefixes to return. As duplicate prefixes are omitted,
// fewer total results may be returned than requested.
func (c *ObjectsWatchAllCall) MaxResults(maxResults int64) *ObjectsWatchAllCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *ObjectsWatchAllCall) PageToken(pageToken string) *ObjectsWatchAllCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Prefix sets the optional parameter "prefix": Filter results to
// objects whose names begin with this prefix.
func (c *ObjectsWatchAllCall) Prefix(prefix string) *ObjectsWatchAllCall {
	c.opt_["prefix"] = prefix
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
func (c *ObjectsWatchAllCall) Projection(projection string) *ObjectsWatchAllCall {
	c.opt_["projection"] = projection
	return c
}

// Versions sets the optional parameter "versions": If true, lists all
// versions of a file as distinct results.
func (c *ObjectsWatchAllCall) Versions(versions bool) *ObjectsWatchAllCall {
	c.opt_["versions"] = versions
	return c
}

func (c *ObjectsWatchAllCall) Do() (*Channel, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["delimiter"]; ok {
		params.Set("delimiter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["versions"]; ok {
		params.Set("versions", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/watch")
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
	ret := new(Channel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Watch for changes on all objects in a bucket.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.watchAll",
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
	//     "maxResults": {
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
	//       "description": "Set of properties to return. Defaults to noAcl.",
	//       "enum": [
	//         "full",
	//         "noAcl"
	//       ],
	//       "enumDescriptions": [
	//         "Include all properties.",
	//         "Omit the acl property."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "versions": {
	//       "description": "If true, lists all versions of a file as distinct results.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "b/{bucket}/o/watch",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "response": {
	//     "$ref": "Channel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsSubscription": true
	// }

}
