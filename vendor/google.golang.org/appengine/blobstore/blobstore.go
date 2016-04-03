// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package blobstore provides a client for App Engine's persistent blob
// storage service.
package blobstore // import "google.golang.org/appengine/blobstore"

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/internal"

	basepb "google.golang.org/appengine/internal/base"
	blobpb "google.golang.org/appengine/internal/blobstore"
)

const (
	blobInfoKind      = "__BlobInfo__"
	blobFileIndexKind = "__BlobFileIndex__"
	zeroKey           = appengine.BlobKey("")
)

// BlobInfo is the blob metadata that is stored in the datastore.
// Filename may be empty.
type BlobInfo struct {
	BlobKey      appengine.BlobKey
	ContentType  string    `datastore:"content_type"`
	CreationTime time.Time `datastore:"creation"`
	Filename     string    `datastore:"filename"`
	Size         int64     `datastore:"size"`
	MD5          string    `datastore:"md5_hash"`

	// ObjectName is the Google Cloud Storage name for this blob.
	ObjectName string `datastore:"gs_object_name"`
}

// isErrFieldMismatch returns whether err is a datastore.ErrFieldMismatch.
//
// The blobstore stores blob metadata in the datastore. When loading that
// metadata, it may contain fields that we don't care about. datastore.Get will
// return datastore.ErrFieldMismatch in that case, so we ignore that specific
// error.
func isErrFieldMismatch(err error) bool {
	_, ok := err.(*datastore.ErrFieldMismatch)
	return ok
}

// Stat returns the BlobInfo for a provided blobKey. If no blob was found for
// that key, Stat returns datastore.ErrNoSuchEntity.
func Stat(c context.Context, blobKey appengine.BlobKey) (*BlobInfo, error) {
	c, _ = appengine.Namespace(c, "") // Blobstore is always in the empty string namespace
	dskey := datastore.NewKey(c, blobInfoKind, string(blobKey), 0, nil)
	bi := &BlobInfo{
		BlobKey: blobKey,
	}
	if err := datastore.Get(c, dskey, bi); err != nil && !isErrFieldMismatch(err) {
		return nil, err
	}
	return bi, nil
}

// Send sets the headers on response to instruct App Engine to send a blob as
// the response body. This is more efficient than reading and writing it out
// manually and isn't subject to normal response size limits.
func Send(response http.ResponseWriter, blobKey appengine.BlobKey) {
	hdr := response.Header()
	hdr.Set("X-AppEngine-BlobKey", string(blobKey))

	if hdr.Get("Content-Type") == "" {
		// This value is known to dev_appserver to mean automatic.
		// In production this is remapped to the empty value which
		// means automatic.
		hdr.Set("Content-Type", "application/vnd.google.appengine.auto")
	}
}

// UploadURL creates an upload URL for the form that the user will
// fill out, passing the application path to load when the POST of the
// form is completed. These URLs expire and should not be reused. The
// opts parameter may be nil.
func UploadURL(c context.Context, successPath string, opts *UploadURLOptions) (*url.URL, error) {
	req := &blobpb.CreateUploadURLRequest{
		SuccessPath: proto.String(successPath),
	}
	if opts != nil {
		if n := opts.MaxUploadBytes; n != 0 {
			req.MaxUploadSizeBytes = &n
		}
		if n := opts.MaxUploadBytesPerBlob; n != 0 {
			req.MaxUploadSizePerBlobBytes = &n
		}
		if s := opts.StorageBucket; s != "" {
			req.GsBucketName = &s
		}
	}
	res := &blobpb.CreateUploadURLResponse{}
	if err := internal.Call(c, "blobstore", "CreateUploadURL", req, res); err != nil {
		return nil, err
	}
	return url.Parse(*res.Url)
}

// UploadURLOptions are the options to create an upload URL.
type UploadURLOptions struct {
	MaxUploadBytes        int64 // optional
	MaxUploadBytesPerBlob int64 // optional

	// StorageBucket specifies the Google Cloud Storage bucket in which
	// to store the blob.
	// This is required if you use Cloud Storage instead of Blobstore.
	// Your application must have permission to write to the bucket.
	// You may optionally specify a bucket name and path in the format
	// "bucket_name/path", in which case the included path will be the
	// prefix of the uploaded object's name.
	StorageBucket string
}

// Delete deletes a blob.
func Delete(c context.Context, blobKey appengine.BlobKey) error {
	return DeleteMulti(c, []appengine.BlobKey{blobKey})
}

// DeleteMulti deletes multiple blobs.
func DeleteMulti(c context.Context, blobKey []appengine.BlobKey) error {
	s := make([]string, len(blobKey))
	for i, b := range blobKey {
		s[i] = string(b)
	}
	req := &blobpb.DeleteBlobRequest{
		BlobKey: s,
	}
	res := &basepb.VoidProto{}
	if err := internal.Call(c, "blobstore", "DeleteBlob", req, res); err != nil {
		return err
	}
	return nil
}

func errorf(format string, args ...interface{}) error {
	return fmt.Errorf("blobstore: "+format, args...)
}

// ParseUpload parses the synthetic POST request that your app gets from
// App Engine after a user's successful upload of blobs. Given the request,
// ParseUpload returns a map of the blobs received (keyed by HTML form
// element name) and other non-blob POST parameters.
func ParseUpload(req *http.Request) (blobs map[string][]*BlobInfo, other url.Values, err error) {
	_, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
	if err != nil {
		return nil, nil, err
	}
	boundary := params["boundary"]
	if boundary == "" {
		return nil, nil, errorf("did not find MIME multipart boundary")
	}

	blobs = make(map[string][]*BlobInfo)
	other = make(url.Values)

	mreader := multipart.NewReader(io.MultiReader(req.Body, strings.NewReader("\r\n\r\n")), boundary)
	for {
		part, perr := mreader.NextPart()
		if perr == io.EOF {
			break
		}
		if perr != nil {
			return nil, nil, errorf("error reading next mime part with boundary %q (len=%d): %v",
				boundary, len(boundary), perr)
		}

		bi := &BlobInfo{}
		ctype, params, err := mime.ParseMediaType(part.Header.Get("Content-Disposition"))
		if err != nil {
			return nil, nil, err
		}
		bi.Filename = params["filename"]
		formKey := params["name"]

		ctype, params, err = mime.ParseMediaType(part.Header.Get("Content-Type"))
		if err != nil {
			return nil, nil, err
		}
		bi.BlobKey = appengine.BlobKey(params["blob-key"])
		if ctype != "message/external-body" || bi.BlobKey == "" {
			if formKey != "" {
				slurp, serr := ioutil.ReadAll(part)
				if serr != nil {
					return nil, nil, errorf("error reading %q MIME part", formKey)
				}
				other[formKey] = append(other[formKey], string(slurp))
			}
			continue
		}

		// App Engine sends a MIME header as the body of each MIME part.
		tp := textproto.NewReader(bufio.NewReader(part))
		header, mimeerr := tp.ReadMIMEHeader()
		if mimeerr != nil {
			return nil, nil, mimeerr
		}
		bi.Size, err = strconv.ParseInt(header.Get("Content-Length"), 10, 64)
		if err != nil {
			return nil, nil, err
		}
		bi.ContentType = header.Get("Content-Type")

		// Parse the time from the MIME header like:
		// X-AppEngine-Upload-Creation: 2011-03-15 21:38:34.712136
		createDate := header.Get("X-AppEngine-Upload-Creation")
		if createDate == "" {
			return nil, nil, errorf("expected to find an X-AppEngine-Upload-Creation header")
		}
		bi.CreationTime, err = time.Parse("2006-01-02 15:04:05.000000", createDate)
		if err != nil {
			return nil, nil, errorf("error parsing X-AppEngine-Upload-Creation: %s", err)
		}

		if hdr := header.Get("Content-MD5"); hdr != "" {
			md5, err := base64.URLEncoding.DecodeString(hdr)
			if err != nil {
				return nil, nil, errorf("bad Content-MD5 %q: %v", hdr, err)
			}
			bi.MD5 = string(md5)
		}

		// If the GCS object name was provided, record it.
		bi.ObjectName = header.Get("X-AppEngine-Cloud-Storage-Object")

		blobs[formKey] = append(blobs[formKey], bi)
	}
	return
}

// Reader is a blob reader.
type Reader interface {
	io.Reader
	io.ReaderAt
	io.Seeker
}

// NewReader returns a reader for a blob. It always succeeds; if the blob does
// not exist then an error will be reported upon first read.
func NewReader(c context.Context, blobKey appengine.BlobKey) Reader {
	return openBlob(c, blobKey)
}

// BlobKeyForFile returns a BlobKey for a Google Storage file.
// The filename should be of the form "/gs/bucket_name/object_name".
func BlobKeyForFile(c context.Context, filename string) (appengine.BlobKey, error) {
	req := &blobpb.CreateEncodedGoogleStorageKeyRequest{
		Filename: &filename,
	}
	res := &blobpb.CreateEncodedGoogleStorageKeyResponse{}
	if err := internal.Call(c, "blobstore", "CreateEncodedGoogleStorageKey", req, res); err != nil {
		return "", err
	}
	return appengine.BlobKey(*res.BlobKey), nil
}
