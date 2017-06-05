// Package oss provides a storagedriver.StorageDriver implementation to
// store blobs in Aliyun OSS cloud storage.
//
// This package leverages the denverdino/aliyungo client library for interfacing with
// oss.
//
// Because OSS is a key, value store the Stat call does not support last modification
// time for directories (directories are an abstraction for key, value stores)
//
// +build include_oss

package oss

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/docker/distribution/context"

	"github.com/Sirupsen/logrus"
	"github.com/denverdino/aliyungo/oss"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/base"
	"github.com/docker/distribution/registry/storage/driver/factory"
)

const driverName = "oss"

// minChunkSize defines the minimum multipart upload chunk size
// OSS API requires multipart upload chunks to be at least 5MB
const minChunkSize = 5 << 20

const defaultChunkSize = 2 * minChunkSize
const defaultTimeout = 2 * time.Minute // 2 minute timeout per chunk

// listMax is the largest amount of objects you can request from OSS in a list call
const listMax = 1000

//DriverParameters A struct that encapsulates all of the driver parameters after all values have been set
type DriverParameters struct {
	AccessKeyID     string
	AccessKeySecret string
	Bucket          string
	Region          oss.Region
	Internal        bool
	Encrypt         bool
	Secure          bool
	ChunkSize       int64
	RootDirectory   string
	Endpoint        string
}

func init() {
	factory.Register(driverName, &ossDriverFactory{})
}

// ossDriverFactory implements the factory.StorageDriverFactory interface
type ossDriverFactory struct{}

func (factory *ossDriverFactory) Create(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	return FromParameters(parameters)
}

type driver struct {
	Client        *oss.Client
	Bucket        *oss.Bucket
	ChunkSize     int64
	Encrypt       bool
	RootDirectory string
}

type baseEmbed struct {
	base.Base
}

// Driver is a storagedriver.StorageDriver implementation backed by Aliyun OSS
// Objects are stored at absolute keys in the provided bucket.
type Driver struct {
	baseEmbed
}

// FromParameters constructs a new Driver with a given parameters map
// Required parameters:
// - accesskey
// - secretkey
// - region
// - bucket
// - encrypt
func FromParameters(parameters map[string]interface{}) (*Driver, error) {
	// Providing no values for these is valid in case the user is authenticating

	accessKey, ok := parameters["accesskeyid"]
	if !ok {
		return nil, fmt.Errorf("No accesskeyid parameter provided")
	}
	secretKey, ok := parameters["accesskeysecret"]
	if !ok {
		return nil, fmt.Errorf("No accesskeysecret parameter provided")
	}

	regionName, ok := parameters["region"]
	if !ok || fmt.Sprint(regionName) == "" {
		return nil, fmt.Errorf("No region parameter provided")
	}

	bucket, ok := parameters["bucket"]
	if !ok || fmt.Sprint(bucket) == "" {
		return nil, fmt.Errorf("No bucket parameter provided")
	}

	internalBool := false
	internal, ok := parameters["internal"]
	if ok {
		internalBool, ok = internal.(bool)
		if !ok {
			return nil, fmt.Errorf("The internal parameter should be a boolean")
		}
	}

	encryptBool := false
	encrypt, ok := parameters["encrypt"]
	if ok {
		encryptBool, ok = encrypt.(bool)
		if !ok {
			return nil, fmt.Errorf("The encrypt parameter should be a boolean")
		}
	}

	secureBool := true
	secure, ok := parameters["secure"]
	if ok {
		secureBool, ok = secure.(bool)
		if !ok {
			return nil, fmt.Errorf("The secure parameter should be a boolean")
		}
	}

	chunkSize := int64(defaultChunkSize)
	chunkSizeParam, ok := parameters["chunksize"]
	if ok {
		switch v := chunkSizeParam.(type) {
		case string:
			vv, err := strconv.ParseInt(v, 0, 64)
			if err != nil {
				return nil, fmt.Errorf("chunksize parameter must be an integer, %v invalid", chunkSizeParam)
			}
			chunkSize = vv
		case int64:
			chunkSize = v
		case int, uint, int32, uint32, uint64:
			chunkSize = reflect.ValueOf(v).Convert(reflect.TypeOf(chunkSize)).Int()
		default:
			return nil, fmt.Errorf("invalid valud for chunksize: %#v", chunkSizeParam)
		}

		if chunkSize < minChunkSize {
			return nil, fmt.Errorf("The chunksize %#v parameter should be a number that is larger than or equal to %d", chunkSize, minChunkSize)
		}
	}

	rootDirectory, ok := parameters["rootdirectory"]
	if !ok {
		rootDirectory = ""
	}

	endpoint, ok := parameters["endpoint"]
	if !ok {
		endpoint = ""
	}

	params := DriverParameters{
		AccessKeyID:     fmt.Sprint(accessKey),
		AccessKeySecret: fmt.Sprint(secretKey),
		Bucket:          fmt.Sprint(bucket),
		Region:          oss.Region(fmt.Sprint(regionName)),
		ChunkSize:       chunkSize,
		RootDirectory:   fmt.Sprint(rootDirectory),
		Encrypt:         encryptBool,
		Secure:          secureBool,
		Internal:        internalBool,
		Endpoint:        fmt.Sprint(endpoint),
	}

	return New(params)
}

// New constructs a new Driver with the given Aliyun credentials, region, encryption flag, and
// bucketName
func New(params DriverParameters) (*Driver, error) {

	client := oss.NewOSSClient(params.Region, params.Internal, params.AccessKeyID, params.AccessKeySecret, params.Secure)
	client.SetEndpoint(params.Endpoint)
	bucket := client.Bucket(params.Bucket)
	client.SetDebug(false)

	// Validate that the given credentials have at least read permissions in the
	// given bucket scope.
	if _, err := bucket.List(strings.TrimRight(params.RootDirectory, "/"), "", "", 1); err != nil {
		return nil, err
	}

	// TODO(tg123): Currently multipart uploads have no timestamps, so this would be unwise
	// if you initiated a new OSS client while another one is running on the same bucket.

	d := &driver{
		Client:        client,
		Bucket:        bucket,
		ChunkSize:     params.ChunkSize,
		Encrypt:       params.Encrypt,
		RootDirectory: params.RootDirectory,
	}

	return &Driver{
		baseEmbed: baseEmbed{
			Base: base.Base{
				StorageDriver: d,
			},
		},
	}, nil
}

// Implement the storagedriver.StorageDriver interface

func (d *driver) Name() string {
	return driverName
}

// GetContent retrieves the content stored at "path" as a []byte.
func (d *driver) GetContent(ctx context.Context, path string) ([]byte, error) {
	content, err := d.Bucket.Get(d.ossPath(path))
	if err != nil {
		return nil, parseError(path, err)
	}
	return content, nil
}

// PutContent stores the []byte content at a location designated by "path".
func (d *driver) PutContent(ctx context.Context, path string, contents []byte) error {
	return parseError(path, d.Bucket.Put(d.ossPath(path), contents, d.getContentType(), getPermissions(), d.getOptions()))
}

// Reader retrieves an io.ReadCloser for the content stored at "path" with a
// given byte offset.
func (d *driver) Reader(ctx context.Context, path string, offset int64) (io.ReadCloser, error) {
	headers := make(http.Header)
	headers.Add("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")

	resp, err := d.Bucket.GetResponseWithHeaders(d.ossPath(path), headers)
	if err != nil {
		return nil, parseError(path, err)
	}

	// Due to Aliyun OSS API, status 200 and whole object will be return instead of an
	// InvalidRange error when range is invalid.
	//
	// OSS sever will always return http.StatusPartialContent if range is acceptable.
	if resp.StatusCode != http.StatusPartialContent {
		resp.Body.Close()
		return ioutil.NopCloser(bytes.NewReader(nil)), nil
	}

	return resp.Body, nil
}

// Writer returns a FileWriter which will store the content written to it
// at the location designated by "path" after the call to Commit.
func (d *driver) Writer(ctx context.Context, path string, append bool) (storagedriver.FileWriter, error) {
	key := d.ossPath(path)
	if !append {
		// TODO (brianbland): cancel other uploads at this path
		multi, err := d.Bucket.InitMulti(key, d.getContentType(), getPermissions(), d.getOptions())
		if err != nil {
			return nil, err
		}
		return d.newWriter(key, multi, nil), nil
	}
	multis, _, err := d.Bucket.ListMulti(key, "")
	if err != nil {
		return nil, parseError(path, err)
	}
	for _, multi := range multis {
		if key != multi.Key {
			continue
		}
		parts, err := multi.ListParts()
		if err != nil {
			return nil, parseError(path, err)
		}
		var multiSize int64
		for _, part := range parts {
			multiSize += part.Size
		}
		return d.newWriter(key, multi, parts), nil
	}
	return nil, storagedriver.PathNotFoundError{Path: path}
}

// Stat retrieves the FileInfo for the given path, including the current size
// in bytes and the creation time.
func (d *driver) Stat(ctx context.Context, path string) (storagedriver.FileInfo, error) {
	listResponse, err := d.Bucket.List(d.ossPath(path), "", "", 1)
	if err != nil {
		return nil, err
	}

	fi := storagedriver.FileInfoFields{
		Path: path,
	}

	if len(listResponse.Contents) == 1 {
		if listResponse.Contents[0].Key != d.ossPath(path) {
			fi.IsDir = true
		} else {
			fi.IsDir = false
			fi.Size = listResponse.Contents[0].Size

			timestamp, err := time.Parse(time.RFC3339Nano, listResponse.Contents[0].LastModified)
			if err != nil {
				return nil, err
			}
			fi.ModTime = timestamp
		}
	} else if len(listResponse.CommonPrefixes) == 1 {
		fi.IsDir = true
	} else {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}

	return storagedriver.FileInfoInternal{FileInfoFields: fi}, nil
}

// List returns a list of the objects that are direct descendants of the given path.
func (d *driver) List(ctx context.Context, opath string) ([]string, error) {
	path := opath
	if path != "/" && opath[len(path)-1] != '/' {
		path = path + "/"
	}

	// This is to cover for the cases when the rootDirectory of the driver is either "" or "/".
	// In those cases, there is no root prefix to replace and we must actually add a "/" to all
	// results in order to keep them as valid paths as recognized by storagedriver.PathRegexp
	prefix := ""
	if d.ossPath("") == "" {
		prefix = "/"
	}

	listResponse, err := d.Bucket.List(d.ossPath(path), "/", "", listMax)
	if err != nil {
		return nil, parseError(opath, err)
	}

	files := []string{}
	directories := []string{}

	for {
		for _, key := range listResponse.Contents {
			files = append(files, strings.Replace(key.Key, d.ossPath(""), prefix, 1))
		}

		for _, commonPrefix := range listResponse.CommonPrefixes {
			directories = append(directories, strings.Replace(commonPrefix[0:len(commonPrefix)-1], d.ossPath(""), prefix, 1))
		}

		if listResponse.IsTruncated {
			listResponse, err = d.Bucket.List(d.ossPath(path), "/", listResponse.NextMarker, listMax)
			if err != nil {
				return nil, err
			}
		} else {
			break
		}
	}

	if opath != "/" {
		if len(files) == 0 && len(directories) == 0 {
			// Treat empty response as missing directory, since we don't actually
			// have directories in s3.
			return nil, storagedriver.PathNotFoundError{Path: opath}
		}
	}

	return append(files, directories...), nil
}

// Move moves an object stored at sourcePath to destPath, removing the original
// object.
func (d *driver) Move(ctx context.Context, sourcePath string, destPath string) error {
	logrus.Infof("Move from %s to %s", d.ossPath(sourcePath), d.ossPath(destPath))

	err := d.Bucket.CopyLargeFile(d.ossPath(sourcePath), d.ossPath(destPath),
		d.getContentType(),
		getPermissions(),
		oss.Options{})
	if err != nil {
		logrus.Errorf("Failed for move from %s to %s: %v", d.ossPath(sourcePath), d.ossPath(destPath), err)
		return parseError(sourcePath, err)
	}

	return d.Delete(ctx, sourcePath)
}

// Delete recursively deletes all objects stored at "path" and its subpaths.
func (d *driver) Delete(ctx context.Context, path string) error {
	listResponse, err := d.Bucket.List(d.ossPath(path), "", "", listMax)
	if err != nil || len(listResponse.Contents) == 0 {
		return storagedriver.PathNotFoundError{Path: path}
	}

	ossObjects := make([]oss.Object, listMax)

	for len(listResponse.Contents) > 0 {
		for index, key := range listResponse.Contents {
			ossObjects[index].Key = key.Key
		}

		err := d.Bucket.DelMulti(oss.Delete{Quiet: false, Objects: ossObjects[0:len(listResponse.Contents)]})
		if err != nil {
			return nil
		}

		listResponse, err = d.Bucket.List(d.ossPath(path), "", "", listMax)
		if err != nil {
			return err
		}
	}

	return nil
}

// URLFor returns a URL which may be used to retrieve the content stored at the given path.
// May return an UnsupportedMethodErr in certain StorageDriver implementations.
func (d *driver) URLFor(ctx context.Context, path string, options map[string]interface{}) (string, error) {
	methodString := "GET"
	method, ok := options["method"]
	if ok {
		methodString, ok = method.(string)
		if !ok || (methodString != "GET") {
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
	logrus.Infof("methodString: %s, expiresTime: %v", methodString, expiresTime)
	signedURL := d.Bucket.SignedURLWithMethod(methodString, d.ossPath(path), expiresTime, nil, nil)
	logrus.Infof("signed URL: %s", signedURL)
	return signedURL, nil
}

func (d *driver) ossPath(path string) string {
	return strings.TrimLeft(strings.TrimRight(d.RootDirectory, "/")+path, "/")
}

func parseError(path string, err error) error {
	if ossErr, ok := err.(*oss.Error); ok && ossErr.StatusCode == http.StatusNotFound && (ossErr.Code == "NoSuchKey" || ossErr.Code == "") {
		return storagedriver.PathNotFoundError{Path: path}
	}

	return err
}

func hasCode(err error, code string) bool {
	ossErr, ok := err.(*oss.Error)
	return ok && ossErr.Code == code
}

func (d *driver) getOptions() oss.Options {
	return oss.Options{ServerSideEncryption: d.Encrypt}
}

func getPermissions() oss.ACL {
	return oss.Private
}

func (d *driver) getContentType() string {
	return "application/octet-stream"
}

// writer attempts to upload parts to S3 in a buffered fashion where the last
// part is at least as large as the chunksize, so the multipart upload could be
// cleanly resumed in the future. This is violated if Close is called after less
// than a full chunk is written.
type writer struct {
	driver      *driver
	key         string
	multi       *oss.Multi
	parts       []oss.Part
	size        int64
	readyPart   []byte
	pendingPart []byte
	closed      bool
	committed   bool
	cancelled   bool
}

func (d *driver) newWriter(key string, multi *oss.Multi, parts []oss.Part) storagedriver.FileWriter {
	var size int64
	for _, part := range parts {
		size += part.Size
	}
	return &writer{
		driver: d,
		key:    key,
		multi:  multi,
		parts:  parts,
		size:   size,
	}
}

func (w *writer) Write(p []byte) (int, error) {
	if w.closed {
		return 0, fmt.Errorf("already closed")
	} else if w.committed {
		return 0, fmt.Errorf("already committed")
	} else if w.cancelled {
		return 0, fmt.Errorf("already cancelled")
	}

	// If the last written part is smaller than minChunkSize, we need to make a
	// new multipart upload :sadface:
	if len(w.parts) > 0 && int(w.parts[len(w.parts)-1].Size) < minChunkSize {
		err := w.multi.Complete(w.parts)
		if err != nil {
			w.multi.Abort()
			return 0, err
		}

		multi, err := w.driver.Bucket.InitMulti(w.key, w.driver.getContentType(), getPermissions(), w.driver.getOptions())
		if err != nil {
			return 0, err
		}
		w.multi = multi

		// If the entire written file is smaller than minChunkSize, we need to make
		// a new part from scratch :double sad face:
		if w.size < minChunkSize {
			contents, err := w.driver.Bucket.Get(w.key)
			if err != nil {
				return 0, err
			}
			w.parts = nil
			w.readyPart = contents
		} else {
			// Otherwise we can use the old file as the new first part
			_, part, err := multi.PutPartCopy(1, oss.CopyOptions{}, w.driver.Bucket.Name+"/"+w.key)
			if err != nil {
				return 0, err
			}
			w.parts = []oss.Part{part}
		}
	}

	var n int

	for len(p) > 0 {
		// If no parts are ready to write, fill up the first part
		if neededBytes := int(w.driver.ChunkSize) - len(w.readyPart); neededBytes > 0 {
			if len(p) >= neededBytes {
				w.readyPart = append(w.readyPart, p[:neededBytes]...)
				n += neededBytes
				p = p[neededBytes:]
			} else {
				w.readyPart = append(w.readyPart, p...)
				n += len(p)
				p = nil
			}
		}

		if neededBytes := int(w.driver.ChunkSize) - len(w.pendingPart); neededBytes > 0 {
			if len(p) >= neededBytes {
				w.pendingPart = append(w.pendingPart, p[:neededBytes]...)
				n += neededBytes
				p = p[neededBytes:]
				err := w.flushPart()
				if err != nil {
					w.size += int64(n)
					return n, err
				}
			} else {
				w.pendingPart = append(w.pendingPart, p...)
				n += len(p)
				p = nil
			}
		}
	}
	w.size += int64(n)
	return n, nil
}

func (w *writer) Size() int64 {
	return w.size
}

func (w *writer) Close() error {
	if w.closed {
		return fmt.Errorf("already closed")
	}
	w.closed = true
	return w.flushPart()
}

func (w *writer) Cancel() error {
	if w.closed {
		return fmt.Errorf("already closed")
	} else if w.committed {
		return fmt.Errorf("already committed")
	}
	w.cancelled = true
	err := w.multi.Abort()
	return err
}

func (w *writer) Commit() error {
	if w.closed {
		return fmt.Errorf("already closed")
	} else if w.committed {
		return fmt.Errorf("already committed")
	} else if w.cancelled {
		return fmt.Errorf("already cancelled")
	}
	err := w.flushPart()
	if err != nil {
		return err
	}
	w.committed = true
	err = w.multi.Complete(w.parts)
	if err != nil {
		w.multi.Abort()
		return err
	}
	return nil
}

// flushPart flushes buffers to write a part to S3.
// Only called by Write (with both buffers full) and Close/Commit (always)
func (w *writer) flushPart() error {
	if len(w.readyPart) == 0 && len(w.pendingPart) == 0 {
		// nothing to write
		return nil
	}
	if len(w.pendingPart) < int(w.driver.ChunkSize) {
		// closing with a small pending part
		// combine ready and pending to avoid writing a small part
		w.readyPart = append(w.readyPart, w.pendingPart...)
		w.pendingPart = nil
	}

	part, err := w.multi.PutPart(len(w.parts)+1, bytes.NewReader(w.readyPart))
	if err != nil {
		return err
	}
	w.parts = append(w.parts, part)
	w.readyPart = w.pendingPart
	w.pendingPart = nil
	return nil
}
