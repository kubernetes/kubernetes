// Package gcs provides a storagedriver.StorageDriver implementation to
// store blobs in Google cloud storage.
//
// This package leverages the google.golang.org/cloud/storage client library
//for interfacing with gcs.
//
// Because gcs is a key, value store the Stat call does not support last modification
// time for directories (directories are an abstraction for key, value stores)
//
// Note that the contents of incomplete uploads are not accessible even though
// Stat returns their length
//
// +build include_gcs

package gcs

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/jwt"
	"google.golang.org/api/googleapi"
	"google.golang.org/cloud"
	"google.golang.org/cloud/storage"

	"github.com/Sirupsen/logrus"

	ctx "github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/base"
	"github.com/docker/distribution/registry/storage/driver/factory"
)

const (
	driverName     = "gcs"
	dummyProjectID = "<unknown>"

	uploadSessionContentType = "application/x-docker-upload-session"
	minChunkSize             = 256 * 1024
	defaultChunkSize         = 20 * minChunkSize

	maxTries = 5
)

var rangeHeader = regexp.MustCompile(`^bytes=([0-9])+-([0-9]+)$`)

// driverParameters is a struct that encapsulates all of the driver parameters after all values have been set
type driverParameters struct {
	bucket        string
	config        *jwt.Config
	email         string
	privateKey    []byte
	client        *http.Client
	rootDirectory string
	chunkSize     int
}

func init() {
	factory.Register(driverName, &gcsDriverFactory{})
}

// gcsDriverFactory implements the factory.StorageDriverFactory interface
type gcsDriverFactory struct{}

// Create StorageDriver from parameters
func (factory *gcsDriverFactory) Create(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	return FromParameters(parameters)
}

// driver is a storagedriver.StorageDriver implementation backed by GCS
// Objects are stored at absolute keys in the provided bucket.
type driver struct {
	client        *http.Client
	bucket        string
	email         string
	privateKey    []byte
	rootDirectory string
	chunkSize     int
}

// FromParameters constructs a new Driver with a given parameters map
// Required parameters:
// - bucket
func FromParameters(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	bucket, ok := parameters["bucket"]
	if !ok || fmt.Sprint(bucket) == "" {
		return nil, fmt.Errorf("No bucket parameter provided")
	}

	rootDirectory, ok := parameters["rootdirectory"]
	if !ok {
		rootDirectory = ""
	}

	chunkSize := defaultChunkSize
	chunkSizeParam, ok := parameters["chunksize"]
	if ok {
		switch v := chunkSizeParam.(type) {
		case string:
			vv, err := strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("chunksize parameter must be an integer, %v invalid", chunkSizeParam)
			}
			chunkSize = vv
		case int, uint, int32, uint32, uint64, int64:
			chunkSize = int(reflect.ValueOf(v).Convert(reflect.TypeOf(chunkSize)).Int())
		default:
			return nil, fmt.Errorf("invalid valud for chunksize: %#v", chunkSizeParam)
		}

		if chunkSize < minChunkSize {
			return nil, fmt.Errorf("The chunksize %#v parameter should be a number that is larger than or equal to %d", chunkSize, minChunkSize)
		}

		if chunkSize%minChunkSize != 0 {
			return nil, fmt.Errorf("chunksize should be a multiple of %d", minChunkSize)
		}
	}

	var ts oauth2.TokenSource
	jwtConf := new(jwt.Config)
	if keyfile, ok := parameters["keyfile"]; ok {
		jsonKey, err := ioutil.ReadFile(fmt.Sprint(keyfile))
		if err != nil {
			return nil, err
		}
		jwtConf, err = google.JWTConfigFromJSON(jsonKey, storage.ScopeFullControl)
		if err != nil {
			return nil, err
		}
		ts = jwtConf.TokenSource(context.Background())
	} else {
		var err error
		ts, err = google.DefaultTokenSource(context.Background(), storage.ScopeFullControl)
		if err != nil {
			return nil, err
		}
	}

	params := driverParameters{
		bucket:        fmt.Sprint(bucket),
		rootDirectory: fmt.Sprint(rootDirectory),
		email:         jwtConf.Email,
		privateKey:    jwtConf.PrivateKey,
		client:        oauth2.NewClient(context.Background(), ts),
		chunkSize:     chunkSize,
	}

	return New(params)
}

// New constructs a new driver
func New(params driverParameters) (storagedriver.StorageDriver, error) {
	rootDirectory := strings.Trim(params.rootDirectory, "/")
	if rootDirectory != "" {
		rootDirectory += "/"
	}
	if params.chunkSize <= 0 || params.chunkSize%minChunkSize != 0 {
		return nil, fmt.Errorf("Invalid chunksize: %d is not a positive multiple of %d", params.chunkSize, minChunkSize)
	}
	d := &driver{
		bucket:        params.bucket,
		rootDirectory: rootDirectory,
		email:         params.email,
		privateKey:    params.privateKey,
		client:        params.client,
		chunkSize:     params.chunkSize,
	}

	return &base.Base{
		StorageDriver: d,
	}, nil
}

// Implement the storagedriver.StorageDriver interface

func (d *driver) Name() string {
	return driverName
}

// GetContent retrieves the content stored at "path" as a []byte.
// This should primarily be used for small objects.
func (d *driver) GetContent(context ctx.Context, path string) ([]byte, error) {
	gcsContext := d.context(context)
	name := d.pathToKey(path)
	var rc io.ReadCloser
	err := retry(func() error {
		var err error
		rc, err = storage.NewReader(gcsContext, d.bucket, name)
		return err
	})
	if err == storage.ErrObjectNotExist {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	p, err := ioutil.ReadAll(rc)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// PutContent stores the []byte content at a location designated by "path".
// This should primarily be used for small objects.
func (d *driver) PutContent(context ctx.Context, path string, contents []byte) error {
	return retry(func() error {
		wc := storage.NewWriter(d.context(context), d.bucket, d.pathToKey(path))
		wc.ContentType = "application/octet-stream"
		return putContentsClose(wc, contents)
	})
}

// Reader retrieves an io.ReadCloser for the content stored at "path"
// with a given byte offset.
// May be used to resume reading a stream by providing a nonzero offset.
func (d *driver) Reader(context ctx.Context, path string, offset int64) (io.ReadCloser, error) {
	res, err := getObject(d.client, d.bucket, d.pathToKey(path), offset)
	if err != nil {
		if res != nil {
			if res.StatusCode == http.StatusNotFound {
				res.Body.Close()
				return nil, storagedriver.PathNotFoundError{Path: path}
			}

			if res.StatusCode == http.StatusRequestedRangeNotSatisfiable {
				res.Body.Close()
				obj, err := storageStatObject(d.context(context), d.bucket, d.pathToKey(path))
				if err != nil {
					return nil, err
				}
				if offset == int64(obj.Size) {
					return ioutil.NopCloser(bytes.NewReader([]byte{})), nil
				}
				return nil, storagedriver.InvalidOffsetError{Path: path, Offset: offset}
			}
		}
		return nil, err
	}
	if res.Header.Get("Content-Type") == uploadSessionContentType {
		defer res.Body.Close()
		return nil, storagedriver.PathNotFoundError{Path: path}
	}
	return res.Body, nil
}

func getObject(client *http.Client, bucket string, name string, offset int64) (*http.Response, error) {
	// copied from google.golang.org/cloud/storage#NewReader :
	// to set the additional "Range" header
	u := &url.URL{
		Scheme: "https",
		Host:   "storage.googleapis.com",
		Path:   fmt.Sprintf("/%s/%s", bucket, name),
	}
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%v-", offset))
	}
	var res *http.Response
	err = retry(func() error {
		var err error
		res, err = client.Do(req)
		return err
	})
	if err != nil {
		return nil, err
	}
	return res, googleapi.CheckMediaResponse(res)
}

// Writer returns a FileWriter which will store the content written to it
// at the location designated by "path" after the call to Commit.
func (d *driver) Writer(context ctx.Context, path string, append bool) (storagedriver.FileWriter, error) {
	writer := &writer{
		client: d.client,
		bucket: d.bucket,
		name:   d.pathToKey(path),
		buffer: make([]byte, d.chunkSize),
	}

	if append {
		err := writer.init(path)
		if err != nil {
			return nil, err
		}
	}
	return writer, nil
}

type writer struct {
	client     *http.Client
	bucket     string
	name       string
	size       int64
	offset     int64
	closed     bool
	sessionURI string
	buffer     []byte
	buffSize   int
}

// Cancel removes any written content from this FileWriter.
func (w *writer) Cancel() error {
	err := w.checkClosed()
	if err != nil {
		return err
	}
	w.closed = true
	err = storageDeleteObject(cloud.NewContext(dummyProjectID, w.client), w.bucket, w.name)
	if err != nil {
		if status, ok := err.(*googleapi.Error); ok {
			if status.Code == http.StatusNotFound {
				err = nil
			}
		}
	}
	return err
}

func (w *writer) Close() error {
	if w.closed {
		return nil
	}
	w.closed = true

	err := w.writeChunk()
	if err != nil {
		return err
	}

	// Copy the remaining bytes from the buffer to the upload session
	// Normally buffSize will be smaller than minChunkSize. However, in the
	// unlikely event that the upload session failed to start, this number could be higher.
	// In this case we can safely clip the remaining bytes to the minChunkSize
	if w.buffSize > minChunkSize {
		w.buffSize = minChunkSize
	}

	// commit the writes by updating the upload session
	err = retry(func() error {
		wc := storage.NewWriter(cloud.NewContext(dummyProjectID, w.client), w.bucket, w.name)
		wc.ContentType = uploadSessionContentType
		wc.Metadata = map[string]string{
			"Session-URI": w.sessionURI,
			"Offset":      strconv.FormatInt(w.offset, 10),
		}
		return putContentsClose(wc, w.buffer[0:w.buffSize])
	})
	if err != nil {
		return err
	}
	w.size = w.offset + int64(w.buffSize)
	w.buffSize = 0
	return nil
}

func putContentsClose(wc *storage.Writer, contents []byte) error {
	size := len(contents)
	var nn int
	var err error
	for nn < size {
		n, err := wc.Write(contents[nn:size])
		nn += n
		if err != nil {
			break
		}
	}
	if err != nil {
		wc.CloseWithError(err)
		return err
	}
	return wc.Close()
}

// Commit flushes all content written to this FileWriter and makes it
// available for future calls to StorageDriver.GetContent and
// StorageDriver.Reader.
func (w *writer) Commit() error {

	if err := w.checkClosed(); err != nil {
		return err
	}
	w.closed = true

	// no session started yet just perform a simple upload
	if w.sessionURI == "" {
		err := retry(func() error {
			wc := storage.NewWriter(cloud.NewContext(dummyProjectID, w.client), w.bucket, w.name)
			wc.ContentType = "application/octet-stream"
			return putContentsClose(wc, w.buffer[0:w.buffSize])
		})
		if err != nil {
			return err
		}
		w.size = w.offset + int64(w.buffSize)
		w.buffSize = 0
		return nil
	}
	size := w.offset + int64(w.buffSize)
	var nn int
	// loop must be performed at least once to ensure the file is committed even when
	// the buffer is empty
	for {
		n, err := putChunk(w.client, w.sessionURI, w.buffer[nn:w.buffSize], w.offset, size)
		nn += int(n)
		w.offset += n
		w.size = w.offset
		if err != nil {
			w.buffSize = copy(w.buffer, w.buffer[nn:w.buffSize])
			return err
		}
		if nn == w.buffSize {
			break
		}
	}
	w.buffSize = 0
	return nil
}

func (w *writer) checkClosed() error {
	if w.closed {
		return fmt.Errorf("Writer already closed")
	}
	return nil
}

func (w *writer) writeChunk() error {
	var err error
	// chunks can be uploaded only in multiples of minChunkSize
	// chunkSize is a multiple of minChunkSize less than or equal to buffSize
	chunkSize := w.buffSize - (w.buffSize % minChunkSize)
	if chunkSize == 0 {
		return nil
	}
	// if their is no sessionURI yet, obtain one by starting the session
	if w.sessionURI == "" {
		w.sessionURI, err = startSession(w.client, w.bucket, w.name)
	}
	if err != nil {
		return err
	}
	nn, err := putChunk(w.client, w.sessionURI, w.buffer[0:chunkSize], w.offset, -1)
	w.offset += nn
	if w.offset > w.size {
		w.size = w.offset
	}
	// shift the remaining bytes to the start of the buffer
	w.buffSize = copy(w.buffer, w.buffer[int(nn):w.buffSize])

	return err
}

func (w *writer) Write(p []byte) (int, error) {
	err := w.checkClosed()
	if err != nil {
		return 0, err
	}

	var nn int
	for nn < len(p) {
		n := copy(w.buffer[w.buffSize:], p[nn:])
		w.buffSize += n
		if w.buffSize == cap(w.buffer) {
			err = w.writeChunk()
			if err != nil {
				break
			}
		}
		nn += n
	}
	return nn, err
}

// Size returns the number of bytes written to this FileWriter.
func (w *writer) Size() int64 {
	return w.size
}

func (w *writer) init(path string) error {
	res, err := getObject(w.client, w.bucket, w.name, 0)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.Header.Get("Content-Type") != uploadSessionContentType {
		return storagedriver.PathNotFoundError{Path: path}
	}
	offset, err := strconv.ParseInt(res.Header.Get("X-Goog-Meta-Offset"), 10, 64)
	if err != nil {
		return err
	}
	buffer, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return err
	}
	w.sessionURI = res.Header.Get("X-Goog-Meta-Session-URI")
	w.buffSize = copy(w.buffer, buffer)
	w.offset = offset
	w.size = offset + int64(w.buffSize)
	return nil
}

type request func() error

func retry(req request) error {
	backoff := time.Second
	var err error
	for i := 0; i < maxTries; i++ {
		err = req()
		if err == nil {
			return nil
		}

		status, ok := err.(*googleapi.Error)
		if !ok || (status.Code != 429 && status.Code < http.StatusInternalServerError) {
			return err
		}

		time.Sleep(backoff - time.Second + (time.Duration(rand.Int31n(1000)) * time.Millisecond))
		if i <= 4 {
			backoff = backoff * 2
		}
	}
	return err
}

// Stat retrieves the FileInfo for the given path, including the current
// size in bytes and the creation time.
func (d *driver) Stat(context ctx.Context, path string) (storagedriver.FileInfo, error) {
	var fi storagedriver.FileInfoFields
	//try to get as file
	gcsContext := d.context(context)
	obj, err := storageStatObject(gcsContext, d.bucket, d.pathToKey(path))
	if err == nil {
		if obj.ContentType == uploadSessionContentType {
			return nil, storagedriver.PathNotFoundError{Path: path}
		}
		fi = storagedriver.FileInfoFields{
			Path:    path,
			Size:    obj.Size,
			ModTime: obj.Updated,
			IsDir:   false,
		}
		return storagedriver.FileInfoInternal{FileInfoFields: fi}, nil
	}
	//try to get as folder
	dirpath := d.pathToDirKey(path)

	var query *storage.Query
	query = &storage.Query{}
	query.Prefix = dirpath
	query.MaxResults = 1

	objects, err := storageListObjects(gcsContext, d.bucket, query)
	if err != nil {
		return nil, err
	}
	if len(objects.Results) < 1 {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}
	fi = storagedriver.FileInfoFields{
		Path:  path,
		IsDir: true,
	}
	obj = objects.Results[0]
	if obj.Name == dirpath {
		fi.Size = obj.Size
		fi.ModTime = obj.Updated
	}
	return storagedriver.FileInfoInternal{FileInfoFields: fi}, nil
}

// List returns a list of the objects that are direct descendants of the
//given path.
func (d *driver) List(context ctx.Context, path string) ([]string, error) {
	var query *storage.Query
	query = &storage.Query{}
	query.Delimiter = "/"
	query.Prefix = d.pathToDirKey(path)
	list := make([]string, 0, 64)
	for {
		objects, err := storageListObjects(d.context(context), d.bucket, query)
		if err != nil {
			return nil, err
		}
		for _, object := range objects.Results {
			// GCS does not guarantee strong consistency between
			// DELETE and LIST operations. Check that the object is not deleted,
			// and filter out any objects with a non-zero time-deleted
			if object.Deleted.IsZero() && object.ContentType != uploadSessionContentType {
				list = append(list, d.keyToPath(object.Name))
			}
		}
		for _, subpath := range objects.Prefixes {
			subpath = d.keyToPath(subpath)
			list = append(list, subpath)
		}
		query = objects.Next
		if query == nil {
			break
		}
	}
	if path != "/" && len(list) == 0 {
		// Treat empty response as missing directory, since we don't actually
		// have directories in Google Cloud Storage.
		return nil, storagedriver.PathNotFoundError{Path: path}
	}
	return list, nil
}

// Move moves an object stored at sourcePath to destPath, removing the
// original object.
func (d *driver) Move(context ctx.Context, sourcePath string, destPath string) error {
	gcsContext := d.context(context)
	_, err := storageCopyObject(gcsContext, d.bucket, d.pathToKey(sourcePath), d.bucket, d.pathToKey(destPath), nil)
	if err != nil {
		if status, ok := err.(*googleapi.Error); ok {
			if status.Code == http.StatusNotFound {
				return storagedriver.PathNotFoundError{Path: sourcePath}
			}
		}
		return err
	}
	err = storageDeleteObject(gcsContext, d.bucket, d.pathToKey(sourcePath))
	// if deleting the file fails, log the error, but do not fail; the file was successfully copied,
	// and the original should eventually be cleaned when purging the uploads folder.
	if err != nil {
		logrus.Infof("error deleting file: %v due to %v", sourcePath, err)
	}
	return nil
}

// listAll recursively lists all names of objects stored at "prefix" and its subpaths.
func (d *driver) listAll(context context.Context, prefix string) ([]string, error) {
	list := make([]string, 0, 64)
	query := &storage.Query{}
	query.Prefix = prefix
	query.Versions = false
	for {
		objects, err := storageListObjects(d.context(context), d.bucket, query)
		if err != nil {
			return nil, err
		}
		for _, obj := range objects.Results {
			// GCS does not guarantee strong consistency between
			// DELETE and LIST operations. Check that the object is not deleted,
			// and filter out any objects with a non-zero time-deleted
			if obj.Deleted.IsZero() {
				list = append(list, obj.Name)
			}
		}
		query = objects.Next
		if query == nil {
			break
		}
	}
	return list, nil
}

// Delete recursively deletes all objects stored at "path" and its subpaths.
func (d *driver) Delete(context ctx.Context, path string) error {
	prefix := d.pathToDirKey(path)
	gcsContext := d.context(context)
	keys, err := d.listAll(gcsContext, prefix)
	if err != nil {
		return err
	}
	if len(keys) > 0 {
		sort.Sort(sort.Reverse(sort.StringSlice(keys)))
		for _, key := range keys {
			err := storageDeleteObject(gcsContext, d.bucket, key)
			// GCS only guarantees eventual consistency, so listAll might return
			// paths that no longer exist. If this happens, just ignore any not
			// found error
			if status, ok := err.(*googleapi.Error); ok {
				if status.Code == http.StatusNotFound {
					err = nil
				}
			}
			if err != nil {
				return err
			}
		}
		return nil
	}
	err = storageDeleteObject(gcsContext, d.bucket, d.pathToKey(path))
	if err != nil {
		if status, ok := err.(*googleapi.Error); ok {
			if status.Code == http.StatusNotFound {
				return storagedriver.PathNotFoundError{Path: path}
			}
		}
	}
	return err
}

func storageDeleteObject(context context.Context, bucket string, name string) error {
	return retry(func() error {
		return storage.DeleteObject(context, bucket, name)
	})
}

func storageStatObject(context context.Context, bucket string, name string) (*storage.Object, error) {
	var obj *storage.Object
	err := retry(func() error {
		var err error
		obj, err = storage.StatObject(context, bucket, name)
		return err
	})
	return obj, err
}

func storageListObjects(context context.Context, bucket string, q *storage.Query) (*storage.Objects, error) {
	var objs *storage.Objects
	err := retry(func() error {
		var err error
		objs, err = storage.ListObjects(context, bucket, q)
		return err
	})
	return objs, err
}

func storageCopyObject(context context.Context, srcBucket, srcName string, destBucket, destName string, attrs *storage.ObjectAttrs) (*storage.Object, error) {
	var obj *storage.Object
	err := retry(func() error {
		var err error
		obj, err = storage.CopyObject(context, srcBucket, srcName, destBucket, destName, attrs)
		return err
	})
	return obj, err
}

// URLFor returns a URL which may be used to retrieve the content stored at
// the given path, possibly using the given options.
// Returns ErrUnsupportedMethod if this driver has no privateKey
func (d *driver) URLFor(context ctx.Context, path string, options map[string]interface{}) (string, error) {
	if d.privateKey == nil {
		return "", storagedriver.ErrUnsupportedMethod{}
	}

	name := d.pathToKey(path)
	methodString := "GET"
	method, ok := options["method"]
	if ok {
		methodString, ok = method.(string)
		if !ok || (methodString != "GET" && methodString != "HEAD") {
			return "", storagedriver.ErrUnsupportedMethod{}
		}
	}

	expiresTime := time.Now().Add(20 * time.Minute)
	expires, ok := options["expiry"]
	if ok {
		et, ok := expires.(time.Time)
		if ok {
			expiresTime = et
		}
	}

	opts := &storage.SignedURLOptions{
		GoogleAccessID: d.email,
		PrivateKey:     d.privateKey,
		Method:         methodString,
		Expires:        expiresTime,
	}
	return storage.SignedURL(d.bucket, name, opts)
}

func startSession(client *http.Client, bucket string, name string) (uri string, err error) {
	u := &url.URL{
		Scheme:   "https",
		Host:     "www.googleapis.com",
		Path:     fmt.Sprintf("/upload/storage/v1/b/%v/o", bucket),
		RawQuery: fmt.Sprintf("uploadType=resumable&name=%v", name),
	}
	err = retry(func() error {
		req, err := http.NewRequest("POST", u.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("X-Upload-Content-Type", "application/octet-stream")
		req.Header.Set("Content-Length", "0")
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		err = googleapi.CheckMediaResponse(resp)
		if err != nil {
			return err
		}
		uri = resp.Header.Get("Location")
		return nil
	})
	return uri, err
}

func putChunk(client *http.Client, sessionURI string, chunk []byte, from int64, totalSize int64) (int64, error) {
	bytesPut := int64(0)
	err := retry(func() error {
		req, err := http.NewRequest("PUT", sessionURI, bytes.NewReader(chunk))
		if err != nil {
			return err
		}
		length := int64(len(chunk))
		to := from + length - 1
		size := "*"
		if totalSize >= 0 {
			size = strconv.FormatInt(totalSize, 10)
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		if from == to+1 {
			req.Header.Set("Content-Range", fmt.Sprintf("bytes */%v", size))
		} else {
			req.Header.Set("Content-Range", fmt.Sprintf("bytes %v-%v/%v", from, to, size))
		}
		req.Header.Set("Content-Length", strconv.FormatInt(length, 10))

		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if totalSize < 0 && resp.StatusCode == 308 {
			groups := rangeHeader.FindStringSubmatch(resp.Header.Get("Range"))
			end, err := strconv.ParseInt(groups[2], 10, 64)
			if err != nil {
				return err
			}
			bytesPut = end - from + 1
			return nil
		}
		err = googleapi.CheckMediaResponse(resp)
		if err != nil {
			return err
		}
		bytesPut = to - from + 1
		return nil
	})
	return bytesPut, err
}

func (d *driver) context(context ctx.Context) context.Context {
	return cloud.WithContext(context, dummyProjectID, d.client)
}

func (d *driver) pathToKey(path string) string {
	return strings.TrimRight(d.rootDirectory+strings.TrimLeft(path, "/"), "/")
}

func (d *driver) pathToDirKey(path string) string {
	return d.pathToKey(path) + "/"
}

func (d *driver) keyToPath(key string) string {
	return "/" + strings.Trim(strings.TrimPrefix(key, d.rootDirectory), "/")
}
