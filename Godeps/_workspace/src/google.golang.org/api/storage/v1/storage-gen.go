// Package storage provides access to the Cloud Storage API.
//
// See https://developers.google.com/storage/docs/json_api/
//
// Usage example:
//
//   import "google.golang.org/api/storage/v1"
//   ...
//   storageService, err := storage.New(oauthHttpClient)
package storage

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
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
var _ = context.Background

const apiId = "storage:v1"
const apiName = "storage"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/storage/v1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Manage your data and permissions in Google Cloud Storage
	DevstorageFullControlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageReadOnlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"
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
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	BucketAccessControls *BucketAccessControlsService

	Buckets *BucketsService

	Channels *ChannelsService

	DefaultObjectAccessControls *DefaultObjectAccessControlsService

	ObjectAccessControls *ObjectAccessControlsService

	Objects *ObjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
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

// Bucket: A bucket.
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

	// Lifecycle: The bucket's lifecycle configuration. See lifecycle
	// management for more information.
	Lifecycle *BucketLifecycle `json:"lifecycle,omitempty"`

	// Location: The location of the bucket. Object data for objects in the
	// bucket resides in physical storage within this region. Defaults to
	// US. See the developer's guide for the authoritative list.
	Location string `json:"location,omitempty"`

	// Logging: The bucket's logging configuration, which defines the
	// destination bucket and optional name prefix for the current bucket's
	// logs.
	Logging *BucketLogging `json:"logging,omitempty"`

	// Metageneration: The metadata generation of this bucket.
	Metageneration int64 `json:"metageneration,omitempty,string"`

	// Name: The name of the bucket.
	Name string `json:"name,omitempty"`

	// Owner: The owner of the bucket. This is always the project team's
	// owner group.
	Owner *BucketOwner `json:"owner,omitempty"`

	// ProjectNumber: The project number of the project the bucket belongs
	// to.
	ProjectNumber uint64 `json:"projectNumber,omitempty,string"`

	// SelfLink: The URI of this bucket.
	SelfLink string `json:"selfLink,omitempty"`

	// StorageClass: The bucket's storage class. This defines how objects in
	// the bucket are stored and determines the SLA and the cost of storage.
	// Values include STANDARD, NEARLINE and DURABLE_REDUCED_AVAILABILITY.
	// Defaults to STANDARD. For more information, see storage classes.
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
	// headers, (GET, OPTIONS, POST, etc) Note: "*" is permitted in the list
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

// BucketLifecycle: The bucket's lifecycle configuration. See lifecycle
// management for more information.
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

// BucketLifecycleRuleAction: The action to take.
type BucketLifecycleRuleAction struct {
	// Type: Type of the action. Currently, only Delete is supported.
	Type string `json:"type,omitempty"`
}

// BucketLifecycleRuleCondition: The condition(s) under which the action
// will be taken.
type BucketLifecycleRuleCondition struct {
	// Age: Age of an object (in days). This condition is satisfied when an
	// object reaches the specified age.
	Age int64 `json:"age,omitempty"`

	// CreatedBefore: A date in RFC 3339 format with only the date part (for
	// instance, "2013-01-15"). This condition is satisfied when an object
	// is created before midnight of the specified date in UTC.
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

// BucketLogging: The bucket's logging configuration, which defines the
// destination bucket and optional name prefix for the current bucket's
// logs.
type BucketLogging struct {
	// LogBucket: The destination bucket where the current bucket's logs
	// should be placed.
	LogBucket string `json:"logBucket,omitempty"`

	// LogObjectPrefix: A prefix for log object names.
	LogObjectPrefix string `json:"logObjectPrefix,omitempty"`
}

// BucketOwner: The owner of the bucket. This is always the project
// team's owner group.
type BucketOwner struct {
	// Entity: The entity, in the form project-owner-projectId.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity.
	EntityId string `json:"entityId,omitempty"`
}

// BucketVersioning: The bucket's versioning configuration.
type BucketVersioning struct {
	// Enabled: While set to true, versioning is fully enabled for this
	// bucket.
	Enabled bool `json:"enabled,omitempty"`
}

// BucketWebsite: The bucket's website configuration.
type BucketWebsite struct {
	// MainPageSuffix: Behaves as the bucket's directory index where missing
	// objects are treated as potential directories.
	MainPageSuffix string `json:"mainPageSuffix,omitempty"`

	// NotFoundPage: The custom object to return when a requested resource
	// is not found.
	NotFoundPage string `json:"notFoundPage,omitempty"`
}

// BucketAccessControl: An access-control entry.
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
	// - domain-domain
	// - project-team-projectId
	// - allUsers
	// - allAuthenticatedUsers Examples:
	// - The user liz@example.com would be user-liz@example.com.
	// - The group example@googlegroups.com would be
	// group-example@googlegroups.com.
	// - To refer to all members of the Google Apps for Business domain
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

	// ProjectTeam: The project team associated with the entity, if any.
	ProjectTeam *BucketAccessControlProjectTeam `json:"projectTeam,omitempty"`

	// Role: The access permission for the entity. Can be READER, WRITER, or
	// OWNER.
	Role string `json:"role,omitempty"`

	// SelfLink: The link to this access-control entry.
	SelfLink string `json:"selfLink,omitempty"`
}

// BucketAccessControlProjectTeam: The project team associated with the
// entity, if any.
type BucketAccessControlProjectTeam struct {
	// ProjectNumber: The project number.
	ProjectNumber string `json:"projectNumber,omitempty"`

	// Team: The team. Can be owners, editors, or viewers.
	Team string `json:"team,omitempty"`
}

// BucketAccessControls: An access-control list.
type BucketAccessControls struct {
	// Items: The list of items.
	Items []*BucketAccessControl `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of bucket access control
	// entries, this is always storage#bucketAccessControls.
	Kind string `json:"kind,omitempty"`
}

// Buckets: A list of buckets.
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
}

// ComposeRequest: A Compose request.
type ComposeRequest struct {
	// Destination: Properties of the resulting object.
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

// ComposeRequestSourceObjectsObjectPreconditions: Conditions that must
// be met for this operation to execute.
type ComposeRequestSourceObjectsObjectPreconditions struct {
	// IfGenerationMatch: Only perform the composition if the generation of
	// the source object that would be used matches this value. If this
	// value and a generation are both specified, they must be the same
	// value or the call will fail.
	IfGenerationMatch int64 `json:"ifGenerationMatch,omitempty,string"`
}

// Object: An object.
type Object struct {
	// Acl: Access controls on the object.
	Acl []*ObjectAccessControl `json:"acl,omitempty"`

	// Bucket: The name of the bucket containing this object.
	Bucket string `json:"bucket,omitempty"`

	// CacheControl: Cache-Control directive for the object data.
	CacheControl string `json:"cacheControl,omitempty"`

	// ComponentCount: Number of underlying components that make up this
	// object. Components are accumulated by compose operations.
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
	// encoded using base64 in big-endian byte order. For more information
	// about using the CRC32c checksum, see Hashes and ETags: Best
	// Practices.
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

	// Md5Hash: MD5 hash of the data; encoded using base64. For more
	// information about using the MD5 hash, see Hashes and ETags: Best
	// Practices.
	Md5Hash string `json:"md5Hash,omitempty"`

	// MediaLink: Media download link.
	MediaLink string `json:"mediaLink,omitempty"`

	// Metadata: User-provided metadata, in key/value pairs.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Metageneration: The version of the metadata for this object at this
	// generation. Used for preconditions and for detecting changes in
	// metadata. A metageneration number is only meaningful in the context
	// of a particular generation of a particular object.
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

	// TimeDeleted: The deletion time of the object in RFC 3339 format. Will
	// be returned if and only if this version of the object has been
	// deleted.
	TimeDeleted string `json:"timeDeleted,omitempty"`

	// Updated: The creation or modification time of the object in RFC 3339
	// format. For buckets with versioning enabled, changing an object's
	// metadata does not change this property.
	Updated string `json:"updated,omitempty"`
}

// ObjectOwner: The owner of the object. This will always be the
// uploader of the object.
type ObjectOwner struct {
	// Entity: The entity, in the form user-userId.
	Entity string `json:"entity,omitempty"`

	// EntityId: The ID for the entity.
	EntityId string `json:"entityId,omitempty"`
}

// ObjectAccessControl: An access-control entry.
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
	// - domain-domain
	// - project-team-projectId
	// - allUsers
	// - allAuthenticatedUsers Examples:
	// - The user liz@example.com would be user-liz@example.com.
	// - The group example@googlegroups.com would be
	// group-example@googlegroups.com.
	// - To refer to all members of the Google Apps for Business domain
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

	// ProjectTeam: The project team associated with the entity, if any.
	ProjectTeam *ObjectAccessControlProjectTeam `json:"projectTeam,omitempty"`

	// Role: The access permission for the entity. Can be READER or OWNER.
	Role string `json:"role,omitempty"`

	// SelfLink: The link to this access-control entry.
	SelfLink string `json:"selfLink,omitempty"`
}

// ObjectAccessControlProjectTeam: The project team associated with the
// entity, if any.
type ObjectAccessControlProjectTeam struct {
	// ProjectNumber: The project number.
	ProjectNumber string `json:"projectNumber,omitempty"`

	// Team: The team. Can be owners, editors, or viewers.
	Team string `json:"team,omitempty"`
}

// ObjectAccessControls: An access-control list.
type ObjectAccessControls struct {
	// Items: The list of items.
	Items []interface{} `json:"items,omitempty"`

	// Kind: The kind of item this is. For lists of object access control
	// entries, this is always storage#objectAccessControls.
	Kind string `json:"kind,omitempty"`
}

// Objects: A list of objects.
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

// RewriteResponse: A rewrite response.
type RewriteResponse struct {
	// Done: true if the copy is finished; otherwise, false if the copy is
	// in progress. This property is always present in the response.
	Done bool `json:"done,omitempty"`

	// Kind: The kind of item this is.
	Kind string `json:"kind,omitempty"`

	// ObjectSize: The total size of the object being copied in bytes. This
	// property is always present in the response.
	ObjectSize uint64 `json:"objectSize,omitempty,string"`

	// Resource: A resource containing the metadata for the copied-to
	// object. This property is present in the response only when copying
	// completes.
	Resource *Object `json:"resource,omitempty"`

	// RewriteToken: A token to use in subsequent requests to continue
	// copying data. This token is present in the response only when there
	// is more data to copy.
	RewriteToken string `json:"rewriteToken,omitempty"`

	// TotalBytesRewritten: The total bytes written so far, which can be
	// used to provide a waiting user with a progress indicator. This
	// property is always present in the response.
	TotalBytesRewritten uint64 `json:"totalBytesRewritten,omitempty,string"`
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsDeleteCall) Fields(s ...googleapi.Field) *BucketAccessControlsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BucketAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsGetCall) Fields(s ...googleapi.Field) *BucketAccessControlsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BucketAccessControlsGetCall) Do() (*BucketAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BucketAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsInsertCall) Fields(s ...googleapi.Field) *BucketAccessControlsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BucketAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsListCall) Fields(s ...googleapi.Field) *BucketAccessControlsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BucketAccessControlsListCall) Do() (*BucketAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BucketAccessControls
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsPatchCall) Fields(s ...googleapi.Field) *BucketAccessControlsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BucketAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketAccessControlsUpdateCall) Fields(s ...googleapi.Field) *BucketAccessControlsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BucketAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
// "ifMetagenerationMatch": If set, only deletes the bucket if its
// metageneration matches this value.
func (c *BucketsDeleteCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *BucketsDeleteCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": If set, only deletes the bucket if its
// metageneration does not match this value.
func (c *BucketsDeleteCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *BucketsDeleteCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsDeleteCall) Fields(s ...googleapi.Field) *BucketsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
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
	//       "description": "If set, only deletes the bucket if its metageneration matches this value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "If set, only deletes the bucket if its metageneration does not match this value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "b/{bucket}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
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
func (c *BucketsGetCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *BucketsGetCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsGetCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *BucketsGetCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit acl and defaultObjectAcl properties.
func (c *BucketsGetCall) Projection(projection string) *BucketsGetCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsGetCall) Fields(s ...googleapi.Field) *BucketsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Bucket
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "int64",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Project team owners get OWNER access, and
// allAuthenticatedUsers get READER access.
//   "private" - Project team owners get OWNER access.
//   "projectPrivate" - Project team members get access according to
// their roles.
//   "publicRead" - Project team owners get OWNER access, and allUsers
// get READER access.
//   "publicReadWrite" - Project team owners get OWNER access, and
// allUsers get WRITER access.
func (c *BucketsInsertCall) PredefinedAcl(predefinedAcl string) *BucketsInsertCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// PredefinedDefaultObjectAcl sets the optional parameter
// "predefinedDefaultObjectAcl": Apply a predefined set of default
// object access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *BucketsInsertCall) PredefinedDefaultObjectAcl(predefinedDefaultObjectAcl string) *BucketsInsertCall {
	c.opt_["predefinedDefaultObjectAcl"] = predefinedDefaultObjectAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the bucket resource
// specifies acl or defaultObjectAcl properties, when it defaults to
// full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit acl and defaultObjectAcl properties.
func (c *BucketsInsertCall) Projection(projection string) *BucketsInsertCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsInsertCall) Fields(s ...googleapi.Field) *BucketsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["predefinedDefaultObjectAcl"]; ok {
		params.Set("predefinedDefaultObjectAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Bucket
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead",
	//         "publicReadWrite"
	//       ],
	//       "enumDescriptions": [
	//         "Project team owners get OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Project team owners get OWNER access.",
	//         "Project team members get access according to their roles.",
	//         "Project team owners get OWNER access, and allUsers get READER access.",
	//         "Project team owners get OWNER access, and allUsers get WRITER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedDefaultObjectAcl": {
	//       "description": "Apply a predefined set of default object access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Prefix sets the optional parameter "prefix": Filter results to
// buckets whose names begin with this prefix.
func (c *BucketsListCall) Prefix(prefix string) *BucketsListCall {
	c.opt_["prefix"] = prefix
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit acl and defaultObjectAcl properties.
func (c *BucketsListCall) Projection(projection string) *BucketsListCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsListCall) Fields(s ...googleapi.Field) *BucketsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["prefix"]; ok {
		params.Set("prefix", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Buckets
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//     "prefix": {
	//       "description": "Filter results to buckets whose names begin with this prefix.",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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
func (c *BucketsPatchCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *BucketsPatchCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsPatchCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *BucketsPatchCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Project team owners get OWNER access, and
// allAuthenticatedUsers get READER access.
//   "private" - Project team owners get OWNER access.
//   "projectPrivate" - Project team members get access according to
// their roles.
//   "publicRead" - Project team owners get OWNER access, and allUsers
// get READER access.
//   "publicReadWrite" - Project team owners get OWNER access, and
// allUsers get WRITER access.
func (c *BucketsPatchCall) PredefinedAcl(predefinedAcl string) *BucketsPatchCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// PredefinedDefaultObjectAcl sets the optional parameter
// "predefinedDefaultObjectAcl": Apply a predefined set of default
// object access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *BucketsPatchCall) PredefinedDefaultObjectAcl(predefinedDefaultObjectAcl string) *BucketsPatchCall {
	c.opt_["predefinedDefaultObjectAcl"] = predefinedDefaultObjectAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit acl and defaultObjectAcl properties.
func (c *BucketsPatchCall) Projection(projection string) *BucketsPatchCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsPatchCall) Fields(s ...googleapi.Field) *BucketsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["predefinedDefaultObjectAcl"]; ok {
		params.Set("predefinedDefaultObjectAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Bucket
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead",
	//         "publicReadWrite"
	//       ],
	//       "enumDescriptions": [
	//         "Project team owners get OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Project team owners get OWNER access.",
	//         "Project team members get access according to their roles.",
	//         "Project team owners get OWNER access, and allUsers get READER access.",
	//         "Project team owners get OWNER access, and allUsers get WRITER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedDefaultObjectAcl": {
	//       "description": "Apply a predefined set of default object access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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
func (c *BucketsUpdateCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *BucketsUpdateCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the return of the bucket metadata
// conditional on whether the bucket's current metageneration does not
// match the given value.
func (c *BucketsUpdateCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *BucketsUpdateCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Project team owners get OWNER access, and
// allAuthenticatedUsers get READER access.
//   "private" - Project team owners get OWNER access.
//   "projectPrivate" - Project team members get access according to
// their roles.
//   "publicRead" - Project team owners get OWNER access, and allUsers
// get READER access.
//   "publicReadWrite" - Project team owners get OWNER access, and
// allUsers get WRITER access.
func (c *BucketsUpdateCall) PredefinedAcl(predefinedAcl string) *BucketsUpdateCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// PredefinedDefaultObjectAcl sets the optional parameter
// "predefinedDefaultObjectAcl": Apply a predefined set of default
// object access controls to this bucket.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *BucketsUpdateCall) PredefinedDefaultObjectAcl(predefinedDefaultObjectAcl string) *BucketsUpdateCall {
	c.opt_["predefinedDefaultObjectAcl"] = predefinedDefaultObjectAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit acl and defaultObjectAcl properties.
func (c *BucketsUpdateCall) Projection(projection string) *BucketsUpdateCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BucketsUpdateCall) Fields(s ...googleapi.Field) *BucketsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["predefinedDefaultObjectAcl"]; ok {
		params.Set("predefinedDefaultObjectAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Bucket
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the return of the bucket metadata conditional on whether the bucket's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead",
	//         "publicReadWrite"
	//       ],
	//       "enumDescriptions": [
	//         "Project team owners get OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Project team owners get OWNER access.",
	//         "Project team members get access according to their roles.",
	//         "Project team owners get OWNER access, and allUsers get READER access.",
	//         "Project team owners get OWNER access, and allUsers get WRITER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedDefaultObjectAcl": {
	//       "description": "Apply a predefined set of default object access controls to this bucket.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChannelsStopCall) Fields(s ...googleapi.Field) *ChannelsStopCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channels/stop")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsDeleteCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DefaultObjectAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsGetCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DefaultObjectAccessControlsGetCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsInsertCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
// "ifMetagenerationMatch": If present, only return default ACL listing
// if the bucket's current metageneration matches this value.
func (c *DefaultObjectAccessControlsListCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *DefaultObjectAccessControlsListCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": If present, only return default ACL
// listing if the bucket's current metageneration does not match the
// given value.
func (c *DefaultObjectAccessControlsListCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *DefaultObjectAccessControlsListCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsListCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControls
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "If present, only return default ACL listing if the bucket's current metageneration matches this value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "If present, only return default ACL listing if the bucket's current metageneration does not match the given value.",
	//       "format": "int64",
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsPatchCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DefaultObjectAccessControlsUpdateCall) Fields(s ...googleapi.Field) *DefaultObjectAccessControlsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/defaultObjectAcl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
func (c *ObjectAccessControlsDeleteCall) Generation(generation int64) *ObjectAccessControlsDeleteCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsDeleteCall) Fields(s ...googleapi.Field) *ObjectAccessControlsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ObjectAccessControlsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
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
	//       "format": "int64",
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
func (c *ObjectAccessControlsGetCall) Generation(generation int64) *ObjectAccessControlsGetCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsGetCall) Fields(s ...googleapi.Field) *ObjectAccessControlsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ObjectAccessControlsGetCall) Do() (*ObjectAccessControl, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
		"entity": c.entity,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
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
func (c *ObjectAccessControlsInsertCall) Generation(generation int64) *ObjectAccessControlsInsertCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsInsertCall) Fields(s ...googleapi.Field) *ObjectAccessControlsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
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
func (c *ObjectAccessControlsListCall) Generation(generation int64) *ObjectAccessControlsListCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsListCall) Fields(s ...googleapi.Field) *ObjectAccessControlsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ObjectAccessControlsListCall) Do() (*ObjectAccessControls, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["generation"]; ok {
		params.Set("generation", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControls
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
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
func (c *ObjectAccessControlsPatchCall) Generation(generation int64) *ObjectAccessControlsPatchCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsPatchCall) Fields(s ...googleapi.Field) *ObjectAccessControlsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
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
func (c *ObjectAccessControlsUpdateCall) Generation(generation int64) *ObjectAccessControlsUpdateCall {
	c.opt_["generation"] = generation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectAccessControlsUpdateCall) Fields(s ...googleapi.Field) *ObjectAccessControlsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}/acl/{entity}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
		"entity": c.entity,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ObjectAccessControl
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "format": "int64",
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

// DestinationPredefinedAcl sets the optional parameter
// "destinationPredefinedAcl": Apply a predefined set of access controls
// to the destination object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsComposeCall) DestinationPredefinedAcl(destinationPredefinedAcl string) *ObjectsComposeCall {
	c.opt_["destinationPredefinedAcl"] = destinationPredefinedAcl
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsComposeCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsComposeCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsComposeCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsComposeCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsComposeCall) Fields(s ...googleapi.Field) *ObjectsComposeCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["destinationPredefinedAcl"]; ok {
		params.Set("destinationPredefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifGenerationMatch"]; ok {
		params.Set("ifGenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ifMetagenerationMatch"]; ok {
		params.Set("ifMetagenerationMatch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{destinationBucket}/o/{destinationObject}/compose")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"destinationBucket": c.destinationBucket,
		"destinationObject": c.destinationObject,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//     "destinationPredefinedAcl": {
	//       "description": "Apply a predefined set of access controls to the destination object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Copy: Copies a source object to a destination object. Optionally
// overrides metadata.
func (r *ObjectsService) Copy(sourceBucket string, sourceObject string, destinationBucket string, destinationObject string, object *Object) *ObjectsCopyCall {
	c := &ObjectsCopyCall{s: r.s, opt_: make(map[string]interface{})}
	c.sourceBucket = sourceBucket
	c.sourceObject = sourceObject
	c.destinationBucket = destinationBucket
	c.destinationObject = destinationObject
	c.object = object
	return c
}

// DestinationPredefinedAcl sets the optional parameter
// "destinationPredefinedAcl": Apply a predefined set of access controls
// to the destination object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsCopyCall) DestinationPredefinedAcl(destinationPredefinedAcl string) *ObjectsCopyCall {
	c.opt_["destinationPredefinedAcl"] = destinationPredefinedAcl
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the destination object's
// current generation matches the given value.
func (c *ObjectsCopyCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsCopyCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the destination object's current generation does not match the given
// value.
func (c *ObjectsCopyCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsCopyCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the destination object's current metageneration matches the given
// value.
func (c *ObjectsCopyCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsCopyCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the destination object's current metageneration does not
// match the given value.
func (c *ObjectsCopyCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsCopyCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// IfSourceGenerationMatch sets the optional parameter
// "ifSourceGenerationMatch": Makes the operation conditional on whether
// the source object's generation matches the given value.
func (c *ObjectsCopyCall) IfSourceGenerationMatch(ifSourceGenerationMatch int64) *ObjectsCopyCall {
	c.opt_["ifSourceGenerationMatch"] = ifSourceGenerationMatch
	return c
}

// IfSourceGenerationNotMatch sets the optional parameter
// "ifSourceGenerationNotMatch": Makes the operation conditional on
// whether the source object's generation does not match the given
// value.
func (c *ObjectsCopyCall) IfSourceGenerationNotMatch(ifSourceGenerationNotMatch int64) *ObjectsCopyCall {
	c.opt_["ifSourceGenerationNotMatch"] = ifSourceGenerationNotMatch
	return c
}

// IfSourceMetagenerationMatch sets the optional parameter
// "ifSourceMetagenerationMatch": Makes the operation conditional on
// whether the source object's current metageneration matches the given
// value.
func (c *ObjectsCopyCall) IfSourceMetagenerationMatch(ifSourceMetagenerationMatch int64) *ObjectsCopyCall {
	c.opt_["ifSourceMetagenerationMatch"] = ifSourceMetagenerationMatch
	return c
}

// IfSourceMetagenerationNotMatch sets the optional parameter
// "ifSourceMetagenerationNotMatch": Makes the operation conditional on
// whether the source object's current metageneration does not match the
// given value.
func (c *ObjectsCopyCall) IfSourceMetagenerationNotMatch(ifSourceMetagenerationNotMatch int64) *ObjectsCopyCall {
	c.opt_["ifSourceMetagenerationNotMatch"] = ifSourceMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the object resource
// specifies the acl property, when it defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsCopyCall) Projection(projection string) *ObjectsCopyCall {
	c.opt_["projection"] = projection
	return c
}

// SourceGeneration sets the optional parameter "sourceGeneration": If
// present, selects a specific revision of the source object (as opposed
// to the latest version, the default).
func (c *ObjectsCopyCall) SourceGeneration(sourceGeneration int64) *ObjectsCopyCall {
	c.opt_["sourceGeneration"] = sourceGeneration
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsCopyCall) Fields(s ...googleapi.Field) *ObjectsCopyCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["destinationPredefinedAcl"]; ok {
		params.Set("destinationPredefinedAcl", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{sourceBucket}/o/{sourceObject}/copyTo/b/{destinationBucket}/o/{destinationObject}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"sourceBucket":      c.sourceBucket,
		"sourceObject":      c.sourceObject,
		"destinationBucket": c.destinationBucket,
		"destinationObject": c.destinationObject,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Copies a source object to a destination object. Optionally overrides metadata.",
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
	//     "destinationPredefinedAcl": {
	//       "description": "Apply a predefined set of access controls to the destination object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration does not match the given value.",
	//       "format": "int64",
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
	//       "format": "int64",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Delete: Deletes an object and its metadata. Deletions are permanent
// if versioning is not enabled for the bucket, or if the generation
// parameter is used.
func (r *ObjectsService) Delete(bucket string, object string) *ObjectsDeleteCall {
	c := &ObjectsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// Generation sets the optional parameter "generation": If present,
// permanently deletes a specific revision of this object (as opposed to
// the latest version, the default).
func (c *ObjectsDeleteCall) Generation(generation int64) *ObjectsDeleteCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsDeleteCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsDeleteCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsDeleteCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsDeleteCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsDeleteCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsDeleteCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsDeleteCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsDeleteCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsDeleteCall) Fields(s ...googleapi.Field) *ObjectsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
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
	//   "description": "Deletes an object and its metadata. Deletions are permanent if versioning is not enabled for the bucket, or if the generation parameter is used.",
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "int64",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Get: Retrieves an object or its metadata.
func (r *ObjectsService) Get(bucket string, object string) *ObjectsGetCall {
	c := &ObjectsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// Generation sets the optional parameter "generation": If present,
// selects a specific revision of this object (as opposed to the latest
// version, the default).
func (c *ObjectsGetCall) Generation(generation int64) *ObjectsGetCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's generation
// matches the given value.
func (c *ObjectsGetCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsGetCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's generation does not match the given value.
func (c *ObjectsGetCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsGetCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsGetCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsGetCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsGetCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsGetCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsGetCall) Projection(projection string) *ObjectsGetCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsGetCall) Fields(s ...googleapi.Field) *ObjectsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an object or its metadata.",
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "int64",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "storage.objects.insert":

type ObjectsInsertCall struct {
	s          *Service
	bucket     string
	object     *Object
	opt_       map[string]interface{}
	media_     io.Reader
	resumable_ googleapi.SizeReaderAt
	mediaType_ string
	ctx_       context.Context
	protocol_  string
}

// Insert: Stores a new object and metadata.
func (r *ObjectsService) Insert(bucket string, object *Object) *ObjectsInsertCall {
	c := &ObjectsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.bucket = bucket
	c.object = object
	return c
}

// ContentEncoding sets the optional parameter "contentEncoding": If
// set, sets the contentEncoding property of the final object to this
// value. Setting this parameter is equivalent to setting the
// contentEncoding metadata property. This can be useful when uploading
// an object with uploadType=media to indicate the encoding of the
// content being uploaded.
func (c *ObjectsInsertCall) ContentEncoding(contentEncoding string) *ObjectsInsertCall {
	c.opt_["contentEncoding"] = contentEncoding
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsInsertCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsInsertCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsInsertCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsInsertCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsInsertCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsInsertCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsInsertCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsInsertCall {
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

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsInsertCall) PredefinedAcl(predefinedAcl string) *ObjectsInsertCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the object resource
// specifies the acl property, when it defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsInsertCall) Projection(projection string) *ObjectsInsertCall {
	c.opt_["projection"] = projection
	return c
}

// Media specifies the media to upload in a single chunk.
// At most one of Media and ResumableMedia may be set.
func (c *ObjectsInsertCall) Media(r io.Reader) *ObjectsInsertCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be cancelled with ctx.
// At most one of Media and ResumableMedia may be set.
// mediaType identifies the MIME media type of the upload, such as "image/png".
// If mediaType is "", it will be auto-detected.
func (c *ObjectsInsertCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *ObjectsInsertCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called after every chunk.
// It should be a low-latency function in order to not slow down the upload operation.
// This should only be called when using ResumableMedia (as opposed to Media).
func (c *ObjectsInsertCall) ProgressUpdater(pu googleapi.ProgressUpdater) *ObjectsInsertCall {
	c.opt_["progressUpdater"] = pu
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsInsertCall) Fields(s ...googleapi.Field) *ObjectsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["contentEncoding"]; ok {
		params.Set("contentEncoding", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["name"]; ok {
		params.Set("name", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o")
	var progressUpdater_ googleapi.ProgressUpdater
	if v, ok := c.opt_["progressUpdater"]; ok {
		if pu, ok := v.(googleapi.ProgressUpdater); ok {
			progressUpdater_ = pu
		}
	}
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", c.protocol_)
	}
	urls += "?" + params.Encode()
	if c.protocol_ != "resumable" {
		var cancel func()
		cancel, _ = googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
		if cancel != nil {
			defer cancel()
		}
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = googleapi.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
		req.Header.Set("Content-Type", "application/json; charset=utf-8")
	} else {
		req.Header.Set("Content-Type", ctype)
	}
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
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
			Callback:      progressUpdater_,
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Stores a new object and metadata.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/storage/v1/b/{bucket}/o"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/storage/v1/b/{bucket}/o"
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
	//     "contentEncoding": {
	//       "description": "If set, sets the contentEncoding property of the final object to this value. Setting this parameter is equivalent to setting the contentEncoding metadata property. This can be useful when uploading an object with uploadType=media to indicate the encoding of the content being uploaded.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Name of the object. Required when the object metadata is not otherwise provided. Overrides the object metadata's name value, if any.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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
// fewer total results may be returned than requested. The default value
// of this parameter is 1,000 items.
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
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsListCall) Projection(projection string) *ObjectsListCall {
	c.opt_["projection"] = projection
	return c
}

// Versions sets the optional parameter "versions": If true, lists all
// versions of an object as distinct results. The default is false. For
// more information, see Object Versioning.
func (c *ObjectsListCall) Versions(versions bool) *ObjectsListCall {
	c.opt_["versions"] = versions
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsListCall) Fields(s ...googleapi.Field) *ObjectsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Objects
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "Maximum number of items plus prefixes to return. As duplicate prefixes are omitted, fewer total results may be returned than requested. The default value of this parameter is 1,000 items.",
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
	//       "description": "If true, lists all versions of an object as distinct results. The default is false. For more information, see Object Versioning.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "b/{bucket}/o",
	//   "response": {
	//     "$ref": "Objects"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Patch: Updates an object's metadata. This method supports patch
// semantics.
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
func (c *ObjectsPatchCall) Generation(generation int64) *ObjectsPatchCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsPatchCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsPatchCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsPatchCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsPatchCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsPatchCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsPatchCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsPatchCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsPatchCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsPatchCall) PredefinedAcl(predefinedAcl string) *ObjectsPatchCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsPatchCall) Projection(projection string) *ObjectsPatchCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsPatchCall) Fields(s ...googleapi.Field) *ObjectsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an object's metadata. This method supports patch semantics.",
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "storage.objects.rewrite":

type ObjectsRewriteCall struct {
	s                 *Service
	sourceBucket      string
	sourceObject      string
	destinationBucket string
	destinationObject string
	object            *Object
	opt_              map[string]interface{}
}

// Rewrite: Rewrites a source object to a destination object. Optionally
// overrides metadata.
func (r *ObjectsService) Rewrite(sourceBucket string, sourceObject string, destinationBucket string, destinationObject string, object *Object) *ObjectsRewriteCall {
	c := &ObjectsRewriteCall{s: r.s, opt_: make(map[string]interface{})}
	c.sourceBucket = sourceBucket
	c.sourceObject = sourceObject
	c.destinationBucket = destinationBucket
	c.destinationObject = destinationObject
	c.object = object
	return c
}

// DestinationPredefinedAcl sets the optional parameter
// "destinationPredefinedAcl": Apply a predefined set of access controls
// to the destination object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsRewriteCall) DestinationPredefinedAcl(destinationPredefinedAcl string) *ObjectsRewriteCall {
	c.opt_["destinationPredefinedAcl"] = destinationPredefinedAcl
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the destination object's
// current generation matches the given value.
func (c *ObjectsRewriteCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsRewriteCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the destination object's current generation does not match the given
// value.
func (c *ObjectsRewriteCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsRewriteCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the destination object's current metageneration matches the given
// value.
func (c *ObjectsRewriteCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsRewriteCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the destination object's current metageneration does not
// match the given value.
func (c *ObjectsRewriteCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsRewriteCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// IfSourceGenerationMatch sets the optional parameter
// "ifSourceGenerationMatch": Makes the operation conditional on whether
// the source object's generation matches the given value.
func (c *ObjectsRewriteCall) IfSourceGenerationMatch(ifSourceGenerationMatch int64) *ObjectsRewriteCall {
	c.opt_["ifSourceGenerationMatch"] = ifSourceGenerationMatch
	return c
}

// IfSourceGenerationNotMatch sets the optional parameter
// "ifSourceGenerationNotMatch": Makes the operation conditional on
// whether the source object's generation does not match the given
// value.
func (c *ObjectsRewriteCall) IfSourceGenerationNotMatch(ifSourceGenerationNotMatch int64) *ObjectsRewriteCall {
	c.opt_["ifSourceGenerationNotMatch"] = ifSourceGenerationNotMatch
	return c
}

// IfSourceMetagenerationMatch sets the optional parameter
// "ifSourceMetagenerationMatch": Makes the operation conditional on
// whether the source object's current metageneration matches the given
// value.
func (c *ObjectsRewriteCall) IfSourceMetagenerationMatch(ifSourceMetagenerationMatch int64) *ObjectsRewriteCall {
	c.opt_["ifSourceMetagenerationMatch"] = ifSourceMetagenerationMatch
	return c
}

// IfSourceMetagenerationNotMatch sets the optional parameter
// "ifSourceMetagenerationNotMatch": Makes the operation conditional on
// whether the source object's current metageneration does not match the
// given value.
func (c *ObjectsRewriteCall) IfSourceMetagenerationNotMatch(ifSourceMetagenerationNotMatch int64) *ObjectsRewriteCall {
	c.opt_["ifSourceMetagenerationNotMatch"] = ifSourceMetagenerationNotMatch
	return c
}

// MaxBytesRewrittenPerCall sets the optional parameter
// "maxBytesRewrittenPerCall": The maximum number of bytes that will be
// rewritten per rewrite request. Most callers shouldn't need to specify
// this parameter - it is primarily in place to support testing. If
// specified the value must be an integral multiple of 1 MiB (1048576).
// Also, this only applies to requests where the source and destination
// span locations and/or storage classes. Finally, this value must not
// change across rewrite calls else you'll get an error that the
// rewriteToken is invalid.
func (c *ObjectsRewriteCall) MaxBytesRewrittenPerCall(maxBytesRewrittenPerCall int64) *ObjectsRewriteCall {
	c.opt_["maxBytesRewrittenPerCall"] = maxBytesRewrittenPerCall
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to noAcl, unless the object resource
// specifies the acl property, when it defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsRewriteCall) Projection(projection string) *ObjectsRewriteCall {
	c.opt_["projection"] = projection
	return c
}

// RewriteToken sets the optional parameter "rewriteToken": Include this
// field (from the previous rewrite response) on each rewrite request
// after the first one, until the rewrite response 'done' flag is true.
// Calls that provide a rewriteToken can omit all other request fields,
// but if included those fields must match the values provided in the
// first rewrite request.
func (c *ObjectsRewriteCall) RewriteToken(rewriteToken string) *ObjectsRewriteCall {
	c.opt_["rewriteToken"] = rewriteToken
	return c
}

// SourceGeneration sets the optional parameter "sourceGeneration": If
// present, selects a specific revision of the source object (as opposed
// to the latest version, the default).
func (c *ObjectsRewriteCall) SourceGeneration(sourceGeneration int64) *ObjectsRewriteCall {
	c.opt_["sourceGeneration"] = sourceGeneration
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsRewriteCall) Fields(s ...googleapi.Field) *ObjectsRewriteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ObjectsRewriteCall) Do() (*RewriteResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.object)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["destinationPredefinedAcl"]; ok {
		params.Set("destinationPredefinedAcl", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["maxBytesRewrittenPerCall"]; ok {
		params.Set("maxBytesRewrittenPerCall", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["rewriteToken"]; ok {
		params.Set("rewriteToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sourceGeneration"]; ok {
		params.Set("sourceGeneration", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{sourceBucket}/o/{sourceObject}/rewriteTo/b/{destinationBucket}/o/{destinationObject}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"sourceBucket":      c.sourceBucket,
		"sourceObject":      c.sourceObject,
		"destinationBucket": c.destinationBucket,
		"destinationObject": c.destinationObject,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *RewriteResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Rewrites a source object to a destination object. Optionally overrides metadata.",
	//   "httpMethod": "POST",
	//   "id": "storage.objects.rewrite",
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
	//     "destinationPredefinedAcl": {
	//       "description": "Apply a predefined set of access controls to the destination object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the destination object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifSourceMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the source object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxBytesRewrittenPerCall": {
	//       "description": "The maximum number of bytes that will be rewritten per rewrite request. Most callers shouldn't need to specify this parameter - it is primarily in place to support testing. If specified the value must be an integral multiple of 1 MiB (1048576). Also, this only applies to requests where the source and destination span locations and/or storage classes. Finally, this value must not change across rewrite calls else you'll get an error that the rewriteToken is invalid.",
	//       "format": "int64",
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
	//     "rewriteToken": {
	//       "description": "Include this field (from the previous rewrite response) on each rewrite request after the first one, until the rewrite response 'done' flag is true. Calls that provide a rewriteToken can omit all other request fields, but if included those fields must match the values provided in the first rewrite request.",
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
	//       "format": "int64",
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
	//   "path": "b/{sourceBucket}/o/{sourceObject}/rewriteTo/b/{destinationBucket}/o/{destinationObject}",
	//   "request": {
	//     "$ref": "Object"
	//   },
	//   "response": {
	//     "$ref": "RewriteResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
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

// Update: Updates an object's metadata.
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
func (c *ObjectsUpdateCall) Generation(generation int64) *ObjectsUpdateCall {
	c.opt_["generation"] = generation
	return c
}

// IfGenerationMatch sets the optional parameter "ifGenerationMatch":
// Makes the operation conditional on whether the object's current
// generation matches the given value.
func (c *ObjectsUpdateCall) IfGenerationMatch(ifGenerationMatch int64) *ObjectsUpdateCall {
	c.opt_["ifGenerationMatch"] = ifGenerationMatch
	return c
}

// IfGenerationNotMatch sets the optional parameter
// "ifGenerationNotMatch": Makes the operation conditional on whether
// the object's current generation does not match the given value.
func (c *ObjectsUpdateCall) IfGenerationNotMatch(ifGenerationNotMatch int64) *ObjectsUpdateCall {
	c.opt_["ifGenerationNotMatch"] = ifGenerationNotMatch
	return c
}

// IfMetagenerationMatch sets the optional parameter
// "ifMetagenerationMatch": Makes the operation conditional on whether
// the object's current metageneration matches the given value.
func (c *ObjectsUpdateCall) IfMetagenerationMatch(ifMetagenerationMatch int64) *ObjectsUpdateCall {
	c.opt_["ifMetagenerationMatch"] = ifMetagenerationMatch
	return c
}

// IfMetagenerationNotMatch sets the optional parameter
// "ifMetagenerationNotMatch": Makes the operation conditional on
// whether the object's current metageneration does not match the given
// value.
func (c *ObjectsUpdateCall) IfMetagenerationNotMatch(ifMetagenerationNotMatch int64) *ObjectsUpdateCall {
	c.opt_["ifMetagenerationNotMatch"] = ifMetagenerationNotMatch
	return c
}

// PredefinedAcl sets the optional parameter "predefinedAcl": Apply a
// predefined set of access controls to this object.
//
// Possible values:
//   "authenticatedRead" - Object owner gets OWNER access, and
// allAuthenticatedUsers get READER access.
//   "bucketOwnerFullControl" - Object owner gets OWNER access, and
// project team owners get OWNER access.
//   "bucketOwnerRead" - Object owner gets OWNER access, and project
// team owners get READER access.
//   "private" - Object owner gets OWNER access.
//   "projectPrivate" - Object owner gets OWNER access, and project team
// members get access according to their roles.
//   "publicRead" - Object owner gets OWNER access, and allUsers get
// READER access.
func (c *ObjectsUpdateCall) PredefinedAcl(predefinedAcl string) *ObjectsUpdateCall {
	c.opt_["predefinedAcl"] = predefinedAcl
	return c
}

// Projection sets the optional parameter "projection": Set of
// properties to return. Defaults to full.
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsUpdateCall) Projection(projection string) *ObjectsUpdateCall {
	c.opt_["projection"] = projection
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsUpdateCall) Fields(s ...googleapi.Field) *ObjectsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["predefinedAcl"]; ok {
		params.Set("predefinedAcl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/{object}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
		"object": c.object,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Object
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an object's metadata.",
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
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifGenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current generation does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration matches the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ifMetagenerationNotMatch": {
	//       "description": "Makes the operation conditional on whether the object's current metageneration does not match the given value.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "object": {
	//       "description": "Name of the object.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "predefinedAcl": {
	//       "description": "Apply a predefined set of access controls to this object.",
	//       "enum": [
	//         "authenticatedRead",
	//         "bucketOwnerFullControl",
	//         "bucketOwnerRead",
	//         "private",
	//         "projectPrivate",
	//         "publicRead"
	//       ],
	//       "enumDescriptions": [
	//         "Object owner gets OWNER access, and allAuthenticatedUsers get READER access.",
	//         "Object owner gets OWNER access, and project team owners get OWNER access.",
	//         "Object owner gets OWNER access, and project team owners get READER access.",
	//         "Object owner gets OWNER access.",
	//         "Object owner gets OWNER access, and project team members get access according to their roles.",
	//         "Object owner gets OWNER access, and allUsers get READER access."
	//       ],
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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
// fewer total results may be returned than requested. The default value
// of this parameter is 1,000 items.
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
//
// Possible values:
//   "full" - Include all properties.
//   "noAcl" - Omit the acl property.
func (c *ObjectsWatchAllCall) Projection(projection string) *ObjectsWatchAllCall {
	c.opt_["projection"] = projection
	return c
}

// Versions sets the optional parameter "versions": If true, lists all
// versions of an object as distinct results. The default is false. For
// more information, see Object Versioning.
func (c *ObjectsWatchAllCall) Versions(versions bool) *ObjectsWatchAllCall {
	c.opt_["versions"] = versions
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ObjectsWatchAllCall) Fields(s ...googleapi.Field) *ObjectsWatchAllCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
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
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "b/{bucket}/o/watch")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"bucket": c.bucket,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Channel
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "Maximum number of items plus prefixes to return. As duplicate prefixes are omitted, fewer total results may be returned than requested. The default value of this parameter is 1,000 items.",
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
	//       "description": "If true, lists all versions of an object as distinct results. The default is false. For more information, see Object Versioning.",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsSubscription": true
	// }

}
