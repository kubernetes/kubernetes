// Package azure provides a storagedriver.StorageDriver implementation to
// store blobs in Microsoft Azure Blob Storage Service.
package azure

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/base"
	"github.com/docker/distribution/registry/storage/driver/factory"

	azure "github.com/Azure/azure-sdk-for-go/storage"
)

const driverName = "azure"

const (
	paramAccountName = "accountname"
	paramAccountKey  = "accountkey"
	paramContainer   = "container"
	paramRealm       = "realm"
	maxChunkSize     = 4 * 1024 * 1024
)

type driver struct {
	client    azure.BlobStorageClient
	container string
}

type baseEmbed struct{ base.Base }

// Driver is a storagedriver.StorageDriver implementation backed by
// Microsoft Azure Blob Storage Service.
type Driver struct{ baseEmbed }

func init() {
	factory.Register(driverName, &azureDriverFactory{})
}

type azureDriverFactory struct{}

func (factory *azureDriverFactory) Create(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	return FromParameters(parameters)
}

// FromParameters constructs a new Driver with a given parameters map.
func FromParameters(parameters map[string]interface{}) (*Driver, error) {
	accountName, ok := parameters[paramAccountName]
	if !ok || fmt.Sprint(accountName) == "" {
		return nil, fmt.Errorf("No %s parameter provided", paramAccountName)
	}

	accountKey, ok := parameters[paramAccountKey]
	if !ok || fmt.Sprint(accountKey) == "" {
		return nil, fmt.Errorf("No %s parameter provided", paramAccountKey)
	}

	container, ok := parameters[paramContainer]
	if !ok || fmt.Sprint(container) == "" {
		return nil, fmt.Errorf("No %s parameter provided", paramContainer)
	}

	realm, ok := parameters[paramRealm]
	if !ok || fmt.Sprint(realm) == "" {
		realm = azure.DefaultBaseURL
	}

	return New(fmt.Sprint(accountName), fmt.Sprint(accountKey), fmt.Sprint(container), fmt.Sprint(realm))
}

// New constructs a new Driver with the given Azure Storage Account credentials
func New(accountName, accountKey, container, realm string) (*Driver, error) {
	api, err := azure.NewClient(accountName, accountKey, realm, azure.DefaultAPIVersion, true)
	if err != nil {
		return nil, err
	}

	blobClient := api.GetBlobService()

	// Create registry container
	if _, err = blobClient.CreateContainerIfNotExists(container, azure.ContainerAccessTypePrivate); err != nil {
		return nil, err
	}

	d := &driver{
		client:    blobClient,
		container: container}
	return &Driver{baseEmbed: baseEmbed{Base: base.Base{StorageDriver: d}}}, nil
}

// Implement the storagedriver.StorageDriver interface.
func (d *driver) Name() string {
	return driverName
}

// GetContent retrieves the content stored at "path" as a []byte.
func (d *driver) GetContent(ctx context.Context, path string) ([]byte, error) {
	blob, err := d.client.GetBlob(d.container, path)
	if err != nil {
		if is404(err) {
			return nil, storagedriver.PathNotFoundError{Path: path}
		}
		return nil, err
	}

	return ioutil.ReadAll(blob)
}

// PutContent stores the []byte content at a location designated by "path".
func (d *driver) PutContent(ctx context.Context, path string, contents []byte) error {
	if _, err := d.client.DeleteBlobIfExists(d.container, path); err != nil {
		return err
	}
	writer, err := d.Writer(ctx, path, false)
	if err != nil {
		return err
	}
	defer writer.Close()
	_, err = writer.Write(contents)
	if err != nil {
		return err
	}
	return writer.Commit()
}

// Reader retrieves an io.ReadCloser for the content stored at "path" with a
// given byte offset.
func (d *driver) Reader(ctx context.Context, path string, offset int64) (io.ReadCloser, error) {
	if ok, err := d.client.BlobExists(d.container, path); err != nil {
		return nil, err
	} else if !ok {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}

	info, err := d.client.GetBlobProperties(d.container, path)
	if err != nil {
		return nil, err
	}

	size := int64(info.ContentLength)
	if offset >= size {
		return ioutil.NopCloser(bytes.NewReader(nil)), nil
	}

	bytesRange := fmt.Sprintf("%v-", offset)
	resp, err := d.client.GetBlobRange(d.container, path, bytesRange)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// Writer returns a FileWriter which will store the content written to it
// at the location designated by "path" after the call to Commit.
func (d *driver) Writer(ctx context.Context, path string, append bool) (storagedriver.FileWriter, error) {
	blobExists, err := d.client.BlobExists(d.container, path)
	if err != nil {
		return nil, err
	}
	var size int64
	if blobExists {
		if append {
			blobProperties, err := d.client.GetBlobProperties(d.container, path)
			if err != nil {
				return nil, err
			}
			size = blobProperties.ContentLength
		} else {
			err := d.client.DeleteBlob(d.container, path)
			if err != nil {
				return nil, err
			}
		}
	} else {
		if append {
			return nil, storagedriver.PathNotFoundError{Path: path}
		}
		err := d.client.PutAppendBlob(d.container, path, nil)
		if err != nil {
			return nil, err
		}
	}

	return d.newWriter(path, size), nil
}

// Stat retrieves the FileInfo for the given path, including the current size
// in bytes and the creation time.
func (d *driver) Stat(ctx context.Context, path string) (storagedriver.FileInfo, error) {
	// Check if the path is a blob
	if ok, err := d.client.BlobExists(d.container, path); err != nil {
		return nil, err
	} else if ok {
		blob, err := d.client.GetBlobProperties(d.container, path)
		if err != nil {
			return nil, err
		}

		mtim, err := time.Parse(http.TimeFormat, blob.LastModified)
		if err != nil {
			return nil, err
		}

		return storagedriver.FileInfoInternal{FileInfoFields: storagedriver.FileInfoFields{
			Path:    path,
			Size:    int64(blob.ContentLength),
			ModTime: mtim,
			IsDir:   false,
		}}, nil
	}

	// Check if path is a virtual container
	virtContainerPath := path
	if !strings.HasSuffix(virtContainerPath, "/") {
		virtContainerPath += "/"
	}
	blobs, err := d.client.ListBlobs(d.container, azure.ListBlobsParameters{
		Prefix:     virtContainerPath,
		MaxResults: 1,
	})
	if err != nil {
		return nil, err
	}
	if len(blobs.Blobs) > 0 {
		// path is a virtual container
		return storagedriver.FileInfoInternal{FileInfoFields: storagedriver.FileInfoFields{
			Path:  path,
			IsDir: true,
		}}, nil
	}

	// path is not a blob or virtual container
	return nil, storagedriver.PathNotFoundError{Path: path}
}

// List returns a list of the objects that are direct descendants of the given
// path.
func (d *driver) List(ctx context.Context, path string) ([]string, error) {
	if path == "/" {
		path = ""
	}

	blobs, err := d.listBlobs(d.container, path)
	if err != nil {
		return blobs, err
	}

	list := directDescendants(blobs, path)
	if path != "" && len(list) == 0 {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}
	return list, nil
}

// Move moves an object stored at sourcePath to destPath, removing the original
// object.
func (d *driver) Move(ctx context.Context, sourcePath string, destPath string) error {
	sourceBlobURL := d.client.GetBlobURL(d.container, sourcePath)
	err := d.client.CopyBlob(d.container, destPath, sourceBlobURL)
	if err != nil {
		if is404(err) {
			return storagedriver.PathNotFoundError{Path: sourcePath}
		}
		return err
	}

	return d.client.DeleteBlob(d.container, sourcePath)
}

// Delete recursively deletes all objects stored at "path" and its subpaths.
func (d *driver) Delete(ctx context.Context, path string) error {
	ok, err := d.client.DeleteBlobIfExists(d.container, path)
	if err != nil {
		return err
	}
	if ok {
		return nil // was a blob and deleted, return
	}

	// Not a blob, see if path is a virtual container with blobs
	blobs, err := d.listBlobs(d.container, path)
	if err != nil {
		return err
	}

	for _, b := range blobs {
		if err = d.client.DeleteBlob(d.container, b); err != nil {
			return err
		}
	}

	if len(blobs) == 0 {
		return storagedriver.PathNotFoundError{Path: path}
	}
	return nil
}

// URLFor returns a publicly accessible URL for the blob stored at given path
// for specified duration by making use of Azure Storage Shared Access Signatures (SAS).
// See https://msdn.microsoft.com/en-us/library/azure/ee395415.aspx for more info.
func (d *driver) URLFor(ctx context.Context, path string, options map[string]interface{}) (string, error) {
	expiresTime := time.Now().UTC().Add(20 * time.Minute) // default expiration
	expires, ok := options["expiry"]
	if ok {
		t, ok := expires.(time.Time)
		if ok {
			expiresTime = t
		}
	}
	return d.client.GetBlobSASURI(d.container, path, expiresTime, "r")
}

// directDescendants will find direct descendants (blobs or virtual containers)
// of from list of blob paths and will return their full paths. Elements in blobs
// list must be prefixed with a "/" and
//
// Example: direct descendants of "/" in {"/foo", "/bar/1", "/bar/2"} is
// {"/foo", "/bar"} and direct descendants of "bar" is {"/bar/1", "/bar/2"}
func directDescendants(blobs []string, prefix string) []string {
	if !strings.HasPrefix(prefix, "/") { // add trailing '/'
		prefix = "/" + prefix
	}
	if !strings.HasSuffix(prefix, "/") { // containerify the path
		prefix += "/"
	}

	out := make(map[string]bool)
	for _, b := range blobs {
		if strings.HasPrefix(b, prefix) {
			rel := b[len(prefix):]
			c := strings.Count(rel, "/")
			if c == 0 {
				out[b] = true
			} else {
				out[prefix+rel[:strings.Index(rel, "/")]] = true
			}
		}
	}

	var keys []string
	for k := range out {
		keys = append(keys, k)
	}
	return keys
}

func (d *driver) listBlobs(container, virtPath string) ([]string, error) {
	if virtPath != "" && !strings.HasSuffix(virtPath, "/") { // containerify the path
		virtPath += "/"
	}

	out := []string{}
	marker := ""
	for {
		resp, err := d.client.ListBlobs(d.container, azure.ListBlobsParameters{
			Marker: marker,
			Prefix: virtPath,
		})

		if err != nil {
			return out, err
		}

		for _, b := range resp.Blobs {
			out = append(out, b.Name)
		}

		if len(resp.Blobs) == 0 || resp.NextMarker == "" {
			break
		}
		marker = resp.NextMarker
	}
	return out, nil
}

func is404(err error) bool {
	statusCodeErr, ok := err.(azure.AzureStorageServiceError)
	return ok && statusCodeErr.StatusCode == http.StatusNotFound
}

type writer struct {
	driver    *driver
	path      string
	size      int64
	bw        *bufio.Writer
	closed    bool
	committed bool
	cancelled bool
}

func (d *driver) newWriter(path string, size int64) storagedriver.FileWriter {
	return &writer{
		driver: d,
		path:   path,
		size:   size,
		bw: bufio.NewWriterSize(&blockWriter{
			client:    d.client,
			container: d.container,
			path:      path,
		}, maxChunkSize),
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

	n, err := w.bw.Write(p)
	w.size += int64(n)
	return n, err
}

func (w *writer) Size() int64 {
	return w.size
}

func (w *writer) Close() error {
	if w.closed {
		return fmt.Errorf("already closed")
	}
	w.closed = true
	return w.bw.Flush()
}

func (w *writer) Cancel() error {
	if w.closed {
		return fmt.Errorf("already closed")
	} else if w.committed {
		return fmt.Errorf("already committed")
	}
	w.cancelled = true
	return w.driver.client.DeleteBlob(w.driver.container, w.path)
}

func (w *writer) Commit() error {
	if w.closed {
		return fmt.Errorf("already closed")
	} else if w.committed {
		return fmt.Errorf("already committed")
	} else if w.cancelled {
		return fmt.Errorf("already cancelled")
	}
	w.committed = true
	return w.bw.Flush()
}

type blockWriter struct {
	client    azure.BlobStorageClient
	container string
	path      string
}

func (bw *blockWriter) Write(p []byte) (int, error) {
	n := 0
	for offset := 0; offset < len(p); offset += maxChunkSize {
		chunkSize := maxChunkSize
		if offset+chunkSize > len(p) {
			chunkSize = len(p) - offset
		}
		err := bw.client.AppendBlock(bw.container, bw.path, p[offset:offset+chunkSize])
		if err != nil {
			return n, err
		}

		n += chunkSize
	}

	return n, nil
}
