// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package storage contains a Google Cloud Storage client.
//
// This package is experimental and may make backwards-incompatible changes.
package storage // import "cloud.google.com/go/storage"

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"

	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
	raw "google.golang.org/api/storage/v1"
)

var (
	ErrBucketNotExist = errors.New("storage: bucket doesn't exist")
	ErrObjectNotExist = errors.New("storage: object doesn't exist")

	// Done is returned by iterators in this package when they have no more items.
	Done = iterator.Done
)

const userAgent = "gcloud-golang-storage/20151204"

const (
	// ScopeFullControl grants permissions to manage your
	// data and permissions in Google Cloud Storage.
	ScopeFullControl = raw.DevstorageFullControlScope

	// ScopeReadOnly grants permissions to
	// view your data in Google Cloud Storage.
	ScopeReadOnly = raw.DevstorageReadOnlyScope

	// ScopeReadWrite grants permissions to manage your
	// data in Google Cloud Storage.
	ScopeReadWrite = raw.DevstorageReadWriteScope
)

// AdminClient is a client type for performing admin operations on a project's
// buckets.
//
// Deprecated: Client has all of AdminClient's methods.
type AdminClient struct {
	c         *Client
	projectID string
}

// NewAdminClient creates a new AdminClient for a given project.
//
// Deprecated: use NewClient instead.
func NewAdminClient(ctx context.Context, projectID string, opts ...option.ClientOption) (*AdminClient, error) {
	c, err := NewClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &AdminClient{
		c:         c,
		projectID: projectID,
	}, nil
}

// Close closes the AdminClient.
func (c *AdminClient) Close() error {
	return c.c.Close()
}

// Create creates a Bucket in the project.
// If attrs is nil the API defaults will be used.
//
// Deprecated: use BucketHandle.Create instead.
func (c *AdminClient) CreateBucket(ctx context.Context, bucketName string, attrs *BucketAttrs) error {
	return c.c.Bucket(bucketName).Create(ctx, c.projectID, attrs)
}

// Delete deletes a Bucket in the project.
//
// Deprecated: use BucketHandle.Delete instead.
func (c *AdminClient) DeleteBucket(ctx context.Context, bucketName string) error {
	return c.c.Bucket(bucketName).Delete(ctx)
}

// Client is a client for interacting with Google Cloud Storage.
type Client struct {
	hc  *http.Client
	raw *raw.Service
}

// NewClient creates a new Google Cloud Storage client.
// The default scope is ScopeFullControl. To use a different scope, like ScopeReadOnly, use option.WithScopes.
func NewClient(ctx context.Context, opts ...option.ClientOption) (*Client, error) {
	o := []option.ClientOption{
		option.WithScopes(ScopeFullControl),
		option.WithUserAgent(userAgent),
	}
	opts = append(o, opts...)
	hc, _, err := transport.NewHTTPClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	rawService, err := raw.New(hc)
	if err != nil {
		return nil, fmt.Errorf("storage client: %v", err)
	}
	return &Client{
		hc:  hc,
		raw: rawService,
	}, nil
}

// Close closes the Client.
func (c *Client) Close() error {
	c.hc = nil
	return nil
}

// BucketHandle provides operations on a Google Cloud Storage bucket.
// Use Client.Bucket to get a handle.
type BucketHandle struct {
	acl              *ACLHandle
	defaultObjectACL *ACLHandle

	c    *Client
	name string
}

// Bucket returns a BucketHandle, which provides operations on the named bucket.
// This call does not perform any network operations.
//
// name must contain only lowercase letters, numbers, dashes, underscores, and
// dots. The full specification for valid bucket names can be found at:
//   https://cloud.google.com/storage/docs/bucket-naming
func (c *Client) Bucket(name string) *BucketHandle {
	return &BucketHandle{
		c:    c,
		name: name,
		acl: &ACLHandle{
			c:      c,
			bucket: name,
		},
		defaultObjectACL: &ACLHandle{
			c:         c,
			bucket:    name,
			isDefault: true,
		},
	}
}

// SignedURLOptions allows you to restrict the access to the signed URL.
type SignedURLOptions struct {
	// GoogleAccessID represents the authorizer of the signed URL generation.
	// It is typically the Google service account client email address from
	// the Google Developers Console in the form of "xxx@developer.gserviceaccount.com".
	// Required.
	GoogleAccessID string

	// PrivateKey is the Google service account private key. It is obtainable
	// from the Google Developers Console.
	// At https://console.developers.google.com/project/<your-project-id>/apiui/credential,
	// create a service account client ID or reuse one of your existing service account
	// credentials. Click on the "Generate new P12 key" to generate and download
	// a new private key. Once you download the P12 file, use the following command
	// to convert it into a PEM file.
	//
	//    $ openssl pkcs12 -in key.p12 -passin pass:notasecret -out key.pem -nodes
	//
	// Provide the contents of the PEM file as a byte slice.
	// Exactly one of PrivateKey or SignBytes must be non-nil.
	PrivateKey []byte

	// SignBytes is a function for implementing custom signing.
	// If your application is running on Google App Engine, you can use appengine's internal signing function:
	//     ctx := appengine.NewContext(request)
	//     acc, _ := appengine.ServiceAccount(ctx)
	//     url, err := SignedURL("bucket", "object", &SignedURLOptions{
	//     	GoogleAccessID: acc,
	//     	SignBytes: func(b []byte) ([]byte, error) {
	//     		_, signedBytes, err := appengine.SignBytes(ctx, b)
	//     		return signedBytes, err
	//     	},
	//     	// etc.
	//     })
	//
	// Exactly one of PrivateKey or SignBytes must be non-nil.
	SignBytes func([]byte) ([]byte, error)

	// Method is the HTTP method to be used with the signed URL.
	// Signed URLs can be used with GET, HEAD, PUT, and DELETE requests.
	// Required.
	Method string

	// Expires is the expiration time on the signed URL. It must be
	// a datetime in the future.
	// Required.
	Expires time.Time

	// ContentType is the content type header the client must provide
	// to use the generated signed URL.
	// Optional.
	ContentType string

	// Headers is a list of extention headers the client must provide
	// in order to use the generated signed URL.
	// Optional.
	Headers []string

	// MD5 is the base64 encoded MD5 checksum of the file.
	// If provided, the client should provide the exact value on the request
	// header in order to use the signed URL.
	// Optional.
	MD5 []byte
}

// SignedURL returns a URL for the specified object. Signed URLs allow
// the users access to a restricted resource for a limited time without having a
// Google account or signing in. For more information about the signed
// URLs, see https://cloud.google.com/storage/docs/accesscontrol#Signed-URLs.
func SignedURL(bucket, name string, opts *SignedURLOptions) (string, error) {
	if opts == nil {
		return "", errors.New("storage: missing required SignedURLOptions")
	}
	if opts.GoogleAccessID == "" {
		return "", errors.New("storage: missing required GoogleAccessID")
	}
	if (opts.PrivateKey == nil) == (opts.SignBytes == nil) {
		return "", errors.New("storage: exactly one of PrivateKey or SignedBytes must be set")
	}
	if opts.Method == "" {
		return "", errors.New("storage: missing required method option")
	}
	if opts.Expires.IsZero() {
		return "", errors.New("storage: missing required expires option")
	}

	signBytes := opts.SignBytes
	if opts.PrivateKey != nil {
		key, err := parseKey(opts.PrivateKey)
		if err != nil {
			return "", err
		}
		signBytes = func(b []byte) ([]byte, error) {
			sum := sha256.Sum256(b)
			return rsa.SignPKCS1v15(
				rand.Reader,
				key,
				crypto.SHA256,
				sum[:],
			)
		}
	} else {
		signBytes = opts.SignBytes
	}

	u := &url.URL{
		Path: fmt.Sprintf("/%s/%s", bucket, name),
	}

	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "%s\n", opts.Method)
	fmt.Fprintf(buf, "%s\n", opts.MD5)
	fmt.Fprintf(buf, "%s\n", opts.ContentType)
	fmt.Fprintf(buf, "%d\n", opts.Expires.Unix())
	fmt.Fprintf(buf, "%s", strings.Join(opts.Headers, "\n"))
	fmt.Fprintf(buf, "%s", u.String())

	b, err := signBytes(buf.Bytes())
	if err != nil {
		return "", err
	}
	encoded := base64.StdEncoding.EncodeToString(b)
	u.Scheme = "https"
	u.Host = "storage.googleapis.com"
	q := u.Query()
	q.Set("GoogleAccessId", opts.GoogleAccessID)
	q.Set("Expires", fmt.Sprintf("%d", opts.Expires.Unix()))
	q.Set("Signature", string(encoded))
	u.RawQuery = q.Encode()
	return u.String(), nil
}

// ObjectHandle provides operations on an object in a Google Cloud Storage bucket.
// Use BucketHandle.Object to get a handle.
type ObjectHandle struct {
	c      *Client
	bucket string
	object string

	acl   *ACLHandle
	conds []Condition
}

// ACL provides access to the object's access control list.
// This controls who can read and write this object.
// This call does not perform any network operations.
func (o *ObjectHandle) ACL() *ACLHandle {
	return o.acl
}

// WithConditions returns a copy of o using the provided conditions.
func (o *ObjectHandle) WithConditions(conds ...Condition) *ObjectHandle {
	o2 := *o
	o2.conds = conds
	return &o2
}

// Attrs returns meta information about the object.
// ErrObjectNotExist will be returned if the object is not found.
func (o *ObjectHandle) Attrs(ctx context.Context) (*ObjectAttrs, error) {
	if !utf8.ValidString(o.object) {
		return nil, fmt.Errorf("storage: object name %q is not valid UTF-8", o.object)
	}
	call := o.c.raw.Objects.Get(o.bucket, o.object).Projection("full").Context(ctx)
	if err := applyConds("Attrs", o.conds, call); err != nil {
		return nil, err
	}
	obj, err := call.Do()
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return nil, ErrObjectNotExist
	}
	if err != nil {
		return nil, err
	}
	return newObject(obj), nil
}

// Update updates an object with the provided attributes.
// All zero-value attributes are ignored.
// ErrObjectNotExist will be returned if the object is not found.
func (o *ObjectHandle) Update(ctx context.Context, attrs ObjectAttrs) (*ObjectAttrs, error) {
	if !utf8.ValidString(o.object) {
		return nil, fmt.Errorf("storage: object name %q is not valid UTF-8", o.object)
	}
	call := o.c.raw.Objects.Patch(o.bucket, o.object, attrs.toRawObject(o.bucket)).Projection("full").Context(ctx)
	if err := applyConds("Update", o.conds, call); err != nil {
		return nil, err
	}
	obj, err := call.Do()
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return nil, ErrObjectNotExist
	}
	if err != nil {
		return nil, err
	}
	return newObject(obj), nil
}

// Delete deletes the single specified object.
func (o *ObjectHandle) Delete(ctx context.Context) error {
	if !utf8.ValidString(o.object) {
		return fmt.Errorf("storage: object name %q is not valid UTF-8", o.object)
	}
	call := o.c.raw.Objects.Delete(o.bucket, o.object).Context(ctx)
	if err := applyConds("Delete", o.conds, call); err != nil {
		return err
	}
	err := call.Do()
	switch e := err.(type) {
	case nil:
		return nil
	case *googleapi.Error:
		if e.Code == http.StatusNotFound {
			return ErrObjectNotExist
		}
	}
	return err
}

// CopyTo copies the object to the given dst.
// The copied object's attributes are overwritten by attrs if non-nil.
func (o *ObjectHandle) CopyTo(ctx context.Context, dst *ObjectHandle, attrs *ObjectAttrs) (*ObjectAttrs, error) {
	// TODO(djd): move bucket/object name validation to a single helper func.
	if o.bucket == "" || dst.bucket == "" {
		return nil, errors.New("storage: the source and destination bucket names must both be non-empty")
	}
	if o.object == "" || dst.object == "" {
		return nil, errors.New("storage: the source and destination object names must both be non-empty")
	}
	if !utf8.ValidString(o.object) {
		return nil, fmt.Errorf("storage: object name %q is not valid UTF-8", o.object)
	}
	if !utf8.ValidString(dst.object) {
		return nil, fmt.Errorf("storage: dst name %q is not valid UTF-8", dst.object)
	}
	var rawObject *raw.Object
	if attrs != nil {
		attrs.Name = dst.object
		if attrs.ContentType == "" {
			return nil, errors.New("storage: attrs.ContentType must be non-empty")
		}
		rawObject = attrs.toRawObject(dst.bucket)
	}
	call := o.c.raw.Objects.Copy(o.bucket, o.object, dst.bucket, dst.object, rawObject).Projection("full").Context(ctx)
	if err := applyConds("CopyTo destination", dst.conds, call); err != nil {
		return nil, err
	}
	if err := applyConds("CopyTo source", toSourceConds(o.conds), call); err != nil {
		return nil, err
	}
	obj, err := call.Do()
	if err != nil {
		return nil, err
	}
	return newObject(obj), nil
}

// ComposeFrom concatenates the provided slice of source objects into a new
// object whose destination is the receiver. The provided attrs, if not nil,
// are used to set the attributes on the newly-created object. All source
// objects must reside within the same bucket as the destination.
func (o *ObjectHandle) ComposeFrom(ctx context.Context, srcs []*ObjectHandle, attrs *ObjectAttrs) (*ObjectAttrs, error) {
	if o.bucket == "" || o.object == "" {
		return nil, errors.New("storage: the destination bucket and object names must be non-empty")
	}
	if len(srcs) == 0 {
		return nil, errors.New("storage: at least one source object must be specified")
	}

	req := &raw.ComposeRequest{}
	if attrs != nil {
		req.Destination = attrs.toRawObject(o.bucket)
		req.Destination.Name = o.object
	}

	for _, src := range srcs {
		if src.bucket != o.bucket {
			return nil, fmt.Errorf("storage: all source objects must be in bucket %q, found %q", o.bucket, src.bucket)
		}
		if src.object == "" {
			return nil, errors.New("storage: all source object names must be non-empty")
		}
		srcObj := &raw.ComposeRequestSourceObjects{
			Name: src.object,
		}
		if err := applyConds("ComposeFrom source", src.conds, composeSourceObj{srcObj}); err != nil {
			return nil, err
		}
		req.SourceObjects = append(req.SourceObjects, srcObj)
	}

	call := o.c.raw.Objects.Compose(o.bucket, o.object, req).Context(ctx)
	if err := applyConds("ComposeFrom destination", o.conds, call); err != nil {
		return nil, err
	}

	obj, err := call.Do()
	if err != nil {
		return nil, err
	}
	return newObject(obj), nil
}

// NewReader creates a new Reader to read the contents of the
// object.
// ErrObjectNotExist will be returned if the object is not found.
func (o *ObjectHandle) NewReader(ctx context.Context) (*Reader, error) {
	return o.NewRangeReader(ctx, 0, -1)
}

// NewRangeReader reads part of an object, reading at most length bytes
// starting at the given offset.  If length is negative, the object is read
// until the end.
func (o *ObjectHandle) NewRangeReader(ctx context.Context, offset, length int64) (*Reader, error) {
	if !utf8.ValidString(o.object) {
		return nil, fmt.Errorf("storage: object name %q is not valid UTF-8", o.object)
	}
	if offset < 0 {
		return nil, fmt.Errorf("storage: invalid offset %d < 0", offset)
	}
	u := &url.URL{
		Scheme: "https",
		Host:   "storage.googleapis.com",
		Path:   fmt.Sprintf("/%s/%s", o.bucket, o.object),
	}
	verb := "GET"
	if length == 0 {
		verb = "HEAD"
	}
	req, err := http.NewRequest(verb, u.String(), nil)
	if err != nil {
		return nil, err
	}
	if err := applyConds("NewReader", o.conds, objectsGetCall{req}); err != nil {
		return nil, err
	}
	if length < 0 && offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	} else if length > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+length-1))
	}
	res, err := o.c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode == http.StatusNotFound {
		res.Body.Close()
		return nil, ErrObjectNotExist
	}
	if res.StatusCode < 200 || res.StatusCode > 299 {
		body, _ := ioutil.ReadAll(res.Body)
		res.Body.Close()
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
			Body:   string(body),
		}
	}
	if offset > 0 && length != 0 && res.StatusCode != http.StatusPartialContent {
		res.Body.Close()
		return nil, errors.New("storage: partial request not satisfied")
	}
	clHeader := res.Header.Get("X-Goog-Stored-Content-Length")
	cl, err := strconv.ParseInt(clHeader, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("storage: can't parse content length %q: %v", clHeader, err)
	}
	remain := res.ContentLength
	body := res.Body
	if length == 0 {
		remain = 0
		body.Close()
		body = emptyBody
	}
	return &Reader{
		body:        body,
		size:        cl,
		remain:      remain,
		contentType: res.Header.Get("Content-Type"),
	}, nil
}

var emptyBody = ioutil.NopCloser(strings.NewReader(""))

// NewWriter returns a storage Writer that writes to the GCS object
// associated with this ObjectHandle.
//
// A new object will be created if an object with this name already exists.
// Otherwise any previous object with the same name will be replaced.
// The object will not be available (and any previous object will remain)
// until Close has been called.
//
// Attributes can be set on the object by modifying the returned Writer's
// ObjectAttrs field before the first call to Write. If no ContentType
// attribute is specified, the content type will be automatically sniffed
// using net/http.DetectContentType.
//
// It is the caller's responsibility to call Close when writing is done.
func (o *ObjectHandle) NewWriter(ctx context.Context) *Writer {
	return &Writer{
		ctx:         ctx,
		o:           o,
		donec:       make(chan struct{}),
		ObjectAttrs: ObjectAttrs{Name: o.object},
	}
}

// parseKey converts the binary contents of a private key file
// to an *rsa.PrivateKey. It detects whether the private key is in a
// PEM container or not. If so, it extracts the the private key
// from PEM container before conversion. It only supports PEM
// containers with no passphrase.
func parseKey(key []byte) (*rsa.PrivateKey, error) {
	if block, _ := pem.Decode(key); block != nil {
		key = block.Bytes
	}
	parsedKey, err := x509.ParsePKCS8PrivateKey(key)
	if err != nil {
		parsedKey, err = x509.ParsePKCS1PrivateKey(key)
		if err != nil {
			return nil, err
		}
	}
	parsed, ok := parsedKey.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("oauth2: private key is invalid")
	}
	return parsed, nil
}

func toRawObjectACL(oldACL []ACLRule) []*raw.ObjectAccessControl {
	var acl []*raw.ObjectAccessControl
	if len(oldACL) > 0 {
		acl = make([]*raw.ObjectAccessControl, len(oldACL))
		for i, rule := range oldACL {
			acl[i] = &raw.ObjectAccessControl{
				Entity: string(rule.Entity),
				Role:   string(rule.Role),
			}
		}
	}
	return acl
}

// toRawObject copies the editable attributes from o to the raw library's Object type.
func (o ObjectAttrs) toRawObject(bucket string) *raw.Object {
	acl := toRawObjectACL(o.ACL)
	return &raw.Object{
		Bucket:             bucket,
		Name:               o.Name,
		ContentType:        o.ContentType,
		ContentEncoding:    o.ContentEncoding,
		ContentLanguage:    o.ContentLanguage,
		CacheControl:       o.CacheControl,
		ContentDisposition: o.ContentDisposition,
		Acl:                acl,
		Metadata:           o.Metadata,
	}
}

// ObjectAttrs represents the metadata for a Google Cloud Storage (GCS) object.
type ObjectAttrs struct {
	// Bucket is the name of the bucket containing this GCS object.
	// This field is read-only.
	Bucket string

	// Name is the name of the object within the bucket.
	// This field is read-only.
	Name string

	// ContentType is the MIME type of the object's content.
	ContentType string

	// ContentLanguage is the content language of the object's content.
	ContentLanguage string

	// CacheControl is the Cache-Control header to be sent in the response
	// headers when serving the object data.
	CacheControl string

	// ACL is the list of access control rules for the object.
	ACL []ACLRule

	// Owner is the owner of the object. This field is read-only.
	//
	// If non-zero, it is in the form of "user-<userId>".
	Owner string

	// Size is the length of the object's content. This field is read-only.
	Size int64

	// ContentEncoding is the encoding of the object's content.
	ContentEncoding string

	// ContentDisposition is the optional Content-Disposition header of the object
	// sent in the response headers.
	ContentDisposition string

	// MD5 is the MD5 hash of the object's content. This field is read-only.
	MD5 []byte

	// CRC32C is the CRC32 checksum of the object's content using
	// the Castagnoli93 polynomial. This field is read-only.
	CRC32C uint32

	// MediaLink is an URL to the object's content. This field is read-only.
	MediaLink string

	// Metadata represents user-provided metadata, in key/value pairs.
	// It can be nil if no metadata is provided.
	Metadata map[string]string

	// Generation is the generation number of the object's content.
	// This field is read-only.
	Generation int64

	// MetaGeneration is the version of the metadata for this
	// object at this generation. This field is used for preconditions
	// and for detecting changes in metadata. A metageneration number
	// is only meaningful in the context of a particular generation
	// of a particular object. This field is read-only.
	MetaGeneration int64

	// StorageClass is the storage class of the bucket.
	// This value defines how objects in the bucket are stored and
	// determines the SLA and the cost of storage. Typical values are
	// "STANDARD" and "DURABLE_REDUCED_AVAILABILITY".
	// It defaults to "STANDARD". This field is read-only.
	StorageClass string

	// Created is the time the object was created. This field is read-only.
	Created time.Time

	// Deleted is the time the object was deleted.
	// If not deleted, it is the zero value. This field is read-only.
	Deleted time.Time

	// Updated is the creation or modification time of the object.
	// For buckets with versioning enabled, changing an object's
	// metadata does not change this property. This field is read-only.
	Updated time.Time

	// Prefix is set only for ObjectAttrs which represent synthetic "directory
	// entries" when iterating over buckets using Query.Delimiter. See
	// ObjectIterator.Next. When set, no other fields in ObjectAttrs will be
	// populated.
	Prefix string
}

// convertTime converts a time in RFC3339 format to time.Time.
// If any error occurs in parsing, the zero-value time.Time is silently returned.
func convertTime(t string) time.Time {
	var r time.Time
	if t != "" {
		r, _ = time.Parse(time.RFC3339, t)
	}
	return r
}

func newObject(o *raw.Object) *ObjectAttrs {
	if o == nil {
		return nil
	}
	acl := make([]ACLRule, len(o.Acl))
	for i, rule := range o.Acl {
		acl[i] = ACLRule{
			Entity: ACLEntity(rule.Entity),
			Role:   ACLRole(rule.Role),
		}
	}
	owner := ""
	if o.Owner != nil {
		owner = o.Owner.Entity
	}
	md5, _ := base64.StdEncoding.DecodeString(o.Md5Hash)
	var crc32c uint32
	d, err := base64.StdEncoding.DecodeString(o.Crc32c)
	if err == nil && len(d) == 4 {
		crc32c = uint32(d[0])<<24 + uint32(d[1])<<16 + uint32(d[2])<<8 + uint32(d[3])
	}
	return &ObjectAttrs{
		Bucket:          o.Bucket,
		Name:            o.Name,
		ContentType:     o.ContentType,
		ContentLanguage: o.ContentLanguage,
		CacheControl:    o.CacheControl,
		ACL:             acl,
		Owner:           owner,
		ContentEncoding: o.ContentEncoding,
		Size:            int64(o.Size),
		MD5:             md5,
		CRC32C:          crc32c,
		MediaLink:       o.MediaLink,
		Metadata:        o.Metadata,
		Generation:      o.Generation,
		MetaGeneration:  o.Metageneration,
		StorageClass:    o.StorageClass,
		Created:         convertTime(o.TimeCreated),
		Deleted:         convertTime(o.TimeDeleted),
		Updated:         convertTime(o.Updated),
	}
}

// Query represents a query to filter objects from a bucket.
type Query struct {
	// Delimiter returns results in a directory-like fashion.
	// Results will contain only objects whose names, aside from the
	// prefix, do not contain delimiter. Objects whose names,
	// aside from the prefix, contain delimiter will have their name,
	// truncated after the delimiter, returned in prefixes.
	// Duplicate prefixes are omitted.
	// Optional.
	Delimiter string

	// Prefix is the prefix filter to query objects
	// whose names begin with this prefix.
	// Optional.
	Prefix string

	// Versions indicates whether multiple versions of the same
	// object will be included in the results.
	Versions bool

	// Cursor is a previously-returned page token
	// representing part of the larger set of results to view.
	// Optional.
	//
	// Deprecated: Use ObjectIterator.PageInfo().Token instead.
	Cursor string

	// MaxResults is the maximum number of items plus prefixes
	// to return. As duplicate prefixes are omitted,
	// fewer total results may be returned than requested.
	// The default page limit is used if it is negative or zero.
	//
	// Deprecated: Use ObjectIterator.PageInfo().MaxSize instead.
	MaxResults int
}

// contentTyper implements ContentTyper to enable an
// io.ReadCloser to specify its MIME type.
type contentTyper struct {
	io.Reader
	t string
}

func (c *contentTyper) ContentType() string {
	return c.t
}

// A Condition constrains methods to act on specific generations of
// resources.
//
// Not all conditions or combinations of conditions are applicable to
// all methods.
type Condition interface {
	// method is the high-level ObjectHandle method name, for
	// error messages.  call is the call object to modify.
	modifyCall(method string, call interface{}) error
}

// applyConds modifies the provided call using the conditions in conds.
// call is something that quacks like a *raw.WhateverCall.
func applyConds(method string, conds []Condition, call interface{}) error {
	for _, cond := range conds {
		if err := cond.modifyCall(method, call); err != nil {
			return err
		}
	}
	return nil
}

// toSourceConds returns a slice of Conditions derived from Conds that instead
// function on the equivalent Source methods of a call.
func toSourceConds(conds []Condition) []Condition {
	out := make([]Condition, 0, len(conds))
	for _, c := range conds {
		switch c := c.(type) {
		case genCond:
			var m string
			if strings.HasPrefix(c.method, "If") {
				m = "IfSource" + c.method[2:]
			} else {
				m = "Source" + c.method
			}
			out = append(out, genCond{method: m, val: c.val})
		default:
			// NOTE(djd): If the message from unsupportedCond becomes
			// confusing, we'll need to find a way for Conditions to
			// identify themselves.
			out = append(out, unsupportedCond{})
		}
	}
	return out
}

func Generation(gen int64) Condition               { return genCond{"Generation", gen} }
func IfGenerationMatch(gen int64) Condition        { return genCond{"IfGenerationMatch", gen} }
func IfGenerationNotMatch(gen int64) Condition     { return genCond{"IfGenerationNotMatch", gen} }
func IfMetaGenerationMatch(gen int64) Condition    { return genCond{"IfMetagenerationMatch", gen} }
func IfMetaGenerationNotMatch(gen int64) Condition { return genCond{"IfMetagenerationNotMatch", gen} }

type genCond struct {
	method string
	val    int64
}

func (g genCond) modifyCall(srcMethod string, call interface{}) error {
	rv := reflect.ValueOf(call)
	meth := rv.MethodByName(g.method)
	if !meth.IsValid() {
		return fmt.Errorf("%s: condition %s not supported", srcMethod, g.method)
	}
	meth.Call([]reflect.Value{reflect.ValueOf(g.val)})
	return nil
}

type unsupportedCond struct{}

func (unsupportedCond) modifyCall(srcMethod string, call interface{}) error {
	return fmt.Errorf("%s: condition not supported", srcMethod)
}

func appendParam(req *http.Request, k, v string) {
	sep := ""
	if req.URL.RawQuery != "" {
		sep = "&"
	}
	req.URL.RawQuery += sep + url.QueryEscape(k) + "=" + url.QueryEscape(v)
}

// objectsGetCall wraps an *http.Request for an object fetch call, but adds the methods
// that modifyCall searches for by name. (the same names as the raw, auto-generated API)
type objectsGetCall struct{ req *http.Request }

func (c objectsGetCall) Generation(gen int64) {
	appendParam(c.req, "generation", fmt.Sprint(gen))
}
func (c objectsGetCall) IfGenerationMatch(gen int64) {
	appendParam(c.req, "ifGenerationMatch", fmt.Sprint(gen))
}
func (c objectsGetCall) IfGenerationNotMatch(gen int64) {
	appendParam(c.req, "ifGenerationNotMatch", fmt.Sprint(gen))
}
func (c objectsGetCall) IfMetagenerationMatch(gen int64) {
	appendParam(c.req, "ifMetagenerationMatch", fmt.Sprint(gen))
}
func (c objectsGetCall) IfMetagenerationNotMatch(gen int64) {
	appendParam(c.req, "ifMetagenerationNotMatch", fmt.Sprint(gen))
}

// composeSourceObj wraps a *raw.ComposeRequestSourceObjects, but adds the methods
// that modifyCall searches for by name.
type composeSourceObj struct {
	src *raw.ComposeRequestSourceObjects
}

func (c composeSourceObj) Generation(gen int64) {
	c.src.Generation = gen
}

func (c composeSourceObj) IfGenerationMatch(gen int64) {
	// It's safe to overwrite ObjectPreconditions, since its only field is
	// IfGenerationMatch.
	c.src.ObjectPreconditions = &raw.ComposeRequestSourceObjectsObjectPreconditions{
		IfGenerationMatch: gen,
	}
}

// TODO(jbd): Add storage.objects.watch.
