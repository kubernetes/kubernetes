// Package s3 provides a storagedriver.StorageDriver implementation to
// store blobs in Amazon S3 cloud storage.
//
// This package leverages the official aws client library for interfacing with
// S3.
//
// Because S3 is a key, value store the Stat call does not support last modification
// time for directories (directories are an abstraction for key, value stores)
//
// Keep in mind that S3 guarantees only read-after-write consistency for new
// objects, but no read-after-update or list-after-write consistency.
package s3

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

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/client/transport"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/base"
	"github.com/docker/distribution/registry/storage/driver/factory"
)

const driverName = "s3aws"

// minChunkSize defines the minimum multipart upload chunk size
// S3 API requires multipart upload chunks to be at least 5MB
const minChunkSize = 5 << 20

const defaultChunkSize = 2 * minChunkSize

// listMax is the largest amount of objects you can request from S3 in a list call
const listMax = 1000

// validRegions maps known s3 region identifiers to region descriptors
var validRegions = map[string]struct{}{}

//DriverParameters A struct that encapsulates all of the driver parameters after all values have been set
type DriverParameters struct {
	AccessKey      string
	SecretKey      string
	Bucket         string
	Region         string
	RegionEndpoint string
	Encrypt        bool
	KeyID          string
	Secure         bool
	ChunkSize      int64
	RootDirectory  string
	StorageClass   string
	UserAgent      string
}

func init() {
	for _, region := range []string{
		"us-east-1",
		"us-west-1",
		"us-west-2",
		"eu-west-1",
		"eu-central-1",
		"ap-southeast-1",
		"ap-southeast-2",
		"ap-northeast-1",
		"ap-northeast-2",
		"sa-east-1",
	} {
		validRegions[region] = struct{}{}
	}

	// Register this as the default s3 driver in addition to s3aws
	factory.Register("s3", &s3DriverFactory{})
	factory.Register(driverName, &s3DriverFactory{})
}

// s3DriverFactory implements the factory.StorageDriverFactory interface
type s3DriverFactory struct{}

func (factory *s3DriverFactory) Create(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	return FromParameters(parameters)
}

type driver struct {
	S3            *s3.S3
	Bucket        string
	ChunkSize     int64
	Encrypt       bool
	KeyID         string
	RootDirectory string
	StorageClass  string
}

type baseEmbed struct {
	base.Base
}

// Driver is a storagedriver.StorageDriver implementation backed by Amazon S3
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
	// with an IAM on an ec2 instance (in which case the instance credentials will
	// be summoned when GetAuth is called)
	accessKey := parameters["accesskey"]
	if accessKey == nil {
		accessKey = ""
	}
	secretKey := parameters["secretkey"]
	if secretKey == nil {
		secretKey = ""
	}

	regionEndpoint := parameters["regionendpoint"]
	if regionEndpoint == nil {
		regionEndpoint = ""
	}

	regionName, ok := parameters["region"]
	if regionName == nil || fmt.Sprint(regionName) == "" {
		return nil, fmt.Errorf("No region parameter provided")
	}
	region := fmt.Sprint(regionName)
	// Don't check the region value if a custom endpoint is provided.
	if regionEndpoint == "" {
		if _, ok = validRegions[region]; !ok {
			return nil, fmt.Errorf("Invalid region provided: %v", region)
		}
	}

	bucket := parameters["bucket"]
	if bucket == nil || fmt.Sprint(bucket) == "" {
		return nil, fmt.Errorf("No bucket parameter provided")
	}

	encryptBool := false
	encrypt := parameters["encrypt"]
	switch encrypt := encrypt.(type) {
	case string:
		b, err := strconv.ParseBool(encrypt)
		if err != nil {
			return nil, fmt.Errorf("The encrypt parameter should be a boolean")
		}
		encryptBool = b
	case bool:
		encryptBool = encrypt
	case nil:
		// do nothing
	default:
		return nil, fmt.Errorf("The encrypt parameter should be a boolean")
	}

	secureBool := true
	secure := parameters["secure"]
	switch secure := secure.(type) {
	case string:
		b, err := strconv.ParseBool(secure)
		if err != nil {
			return nil, fmt.Errorf("The secure parameter should be a boolean")
		}
		secureBool = b
	case bool:
		secureBool = secure
	case nil:
		// do nothing
	default:
		return nil, fmt.Errorf("The secure parameter should be a boolean")
	}

	keyID := parameters["keyid"]
	if keyID == nil {
		keyID = ""
	}

	chunkSize := int64(defaultChunkSize)
	chunkSizeParam := parameters["chunksize"]
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
	case nil:
		// do nothing
	default:
		return nil, fmt.Errorf("invalid value for chunksize: %#v", chunkSizeParam)
	}

	if chunkSize < minChunkSize {
		return nil, fmt.Errorf("The chunksize %#v parameter should be a number that is larger than or equal to %d", chunkSize, minChunkSize)
	}

	rootDirectory := parameters["rootdirectory"]
	if rootDirectory == nil {
		rootDirectory = ""
	}

	storageClass := s3.StorageClassStandard
	storageClassParam := parameters["storageclass"]
	if storageClassParam != nil {
		storageClassString, ok := storageClassParam.(string)
		if !ok {
			return nil, fmt.Errorf("The storageclass parameter must be one of %v, %v invalid", []string{s3.StorageClassStandard, s3.StorageClassReducedRedundancy}, storageClassParam)
		}
		// All valid storage class parameters are UPPERCASE, so be a bit more flexible here
		storageClassString = strings.ToUpper(storageClassString)
		if storageClassString != s3.StorageClassStandard && storageClassString != s3.StorageClassReducedRedundancy {
			return nil, fmt.Errorf("The storageclass parameter must be one of %v, %v invalid", []string{s3.StorageClassStandard, s3.StorageClassReducedRedundancy}, storageClassParam)
		}
		storageClass = storageClassString
	}

	userAgent := parameters["useragent"]
	if userAgent == nil {
		userAgent = ""
	}

	params := DriverParameters{
		fmt.Sprint(accessKey),
		fmt.Sprint(secretKey),
		fmt.Sprint(bucket),
		region,
		fmt.Sprint(regionEndpoint),
		encryptBool,
		fmt.Sprint(keyID),
		secureBool,
		chunkSize,
		fmt.Sprint(rootDirectory),
		storageClass,
		fmt.Sprint(userAgent),
	}

	return New(params)
}

// New constructs a new Driver with the given AWS credentials, region, encryption flag, and
// bucketName
func New(params DriverParameters) (*Driver, error) {
	awsConfig := aws.NewConfig()
	var creds *credentials.Credentials
	if params.RegionEndpoint == "" {
		creds = credentials.NewChainCredentials([]credentials.Provider{
			&credentials.StaticProvider{
				Value: credentials.Value{
					AccessKeyID:     params.AccessKey,
					SecretAccessKey: params.SecretKey,
				},
			},
			&credentials.EnvProvider{},
			&credentials.SharedCredentialsProvider{},
			&ec2rolecreds.EC2RoleProvider{Client: ec2metadata.New(session.New())},
		})

	} else {
		creds = credentials.NewChainCredentials([]credentials.Provider{
			&credentials.StaticProvider{
				Value: credentials.Value{
					AccessKeyID:     params.AccessKey,
					SecretAccessKey: params.SecretKey,
				},
			},
			&credentials.EnvProvider{},
		})
		awsConfig.WithS3ForcePathStyle(true)
		awsConfig.WithEndpoint(params.RegionEndpoint)
	}

	awsConfig.WithCredentials(creds)
	awsConfig.WithRegion(params.Region)
	awsConfig.WithDisableSSL(!params.Secure)

	if params.UserAgent != "" {
		awsConfig.WithHTTPClient(&http.Client{
			Transport: transport.NewTransport(http.DefaultTransport, transport.NewHeaderRequestModifier(http.Header{http.CanonicalHeaderKey("User-Agent"): []string{params.UserAgent}})),
		})
	}

	s3obj := s3.New(session.New(awsConfig))

	// TODO Currently multipart uploads have no timestamps, so this would be unwise
	// if you initiated a new s3driver while another one is running on the same bucket.
	// multis, _, err := bucket.ListMulti("", "")
	// if err != nil {
	// 	return nil, err
	// }

	// for _, multi := range multis {
	// 	err := multi.Abort()
	// 	//TODO appropriate to do this error checking?
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// }

	d := &driver{
		S3:            s3obj,
		Bucket:        params.Bucket,
		ChunkSize:     params.ChunkSize,
		Encrypt:       params.Encrypt,
		KeyID:         params.KeyID,
		RootDirectory: params.RootDirectory,
		StorageClass:  params.StorageClass,
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
	reader, err := d.Reader(ctx, path, 0)
	if err != nil {
		return nil, err
	}
	return ioutil.ReadAll(reader)
}

// PutContent stores the []byte content at a location designated by "path".
func (d *driver) PutContent(ctx context.Context, path string, contents []byte) error {
	_, err := d.S3.PutObject(&s3.PutObjectInput{
		Bucket:               aws.String(d.Bucket),
		Key:                  aws.String(d.s3Path(path)),
		ContentType:          d.getContentType(),
		ACL:                  d.getACL(),
		ServerSideEncryption: d.getEncryptionMode(),
		SSEKMSKeyId:          d.getSSEKMSKeyID(),
		StorageClass:         d.getStorageClass(),
		Body:                 bytes.NewReader(contents),
	})
	return parseError(path, err)
}

// Reader retrieves an io.ReadCloser for the content stored at "path" with a
// given byte offset.
func (d *driver) Reader(ctx context.Context, path string, offset int64) (io.ReadCloser, error) {
	resp, err := d.S3.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(d.Bucket),
		Key:    aws.String(d.s3Path(path)),
		Range:  aws.String("bytes=" + strconv.FormatInt(offset, 10) + "-"),
	})

	if err != nil {
		if s3Err, ok := err.(awserr.Error); ok && s3Err.Code() == "InvalidRange" {
			return ioutil.NopCloser(bytes.NewReader(nil)), nil
		}

		return nil, parseError(path, err)
	}
	return resp.Body, nil
}

// Writer returns a FileWriter which will store the content written to it
// at the location designated by "path" after the call to Commit.
func (d *driver) Writer(ctx context.Context, path string, append bool) (storagedriver.FileWriter, error) {
	key := d.s3Path(path)
	if !append {
		// TODO (brianbland): cancel other uploads at this path
		resp, err := d.S3.CreateMultipartUpload(&s3.CreateMultipartUploadInput{
			Bucket:               aws.String(d.Bucket),
			Key:                  aws.String(key),
			ContentType:          d.getContentType(),
			ACL:                  d.getACL(),
			ServerSideEncryption: d.getEncryptionMode(),
			SSEKMSKeyId:          d.getSSEKMSKeyID(),
			StorageClass:         d.getStorageClass(),
		})
		if err != nil {
			return nil, err
		}
		return d.newWriter(key, *resp.UploadId, nil), nil
	}
	resp, err := d.S3.ListMultipartUploads(&s3.ListMultipartUploadsInput{
		Bucket: aws.String(d.Bucket),
		Prefix: aws.String(key),
	})
	if err != nil {
		return nil, parseError(path, err)
	}

	for _, multi := range resp.Uploads {
		if key != *multi.Key {
			continue
		}
		resp, err := d.S3.ListParts(&s3.ListPartsInput{
			Bucket:   aws.String(d.Bucket),
			Key:      aws.String(key),
			UploadId: multi.UploadId,
		})
		if err != nil {
			return nil, parseError(path, err)
		}
		var multiSize int64
		for _, part := range resp.Parts {
			multiSize += *part.Size
		}
		return d.newWriter(key, *multi.UploadId, resp.Parts), nil
	}
	return nil, storagedriver.PathNotFoundError{Path: path}
}

// Stat retrieves the FileInfo for the given path, including the current size
// in bytes and the creation time.
func (d *driver) Stat(ctx context.Context, path string) (storagedriver.FileInfo, error) {
	resp, err := d.S3.ListObjects(&s3.ListObjectsInput{
		Bucket:  aws.String(d.Bucket),
		Prefix:  aws.String(d.s3Path(path)),
		MaxKeys: aws.Int64(1),
	})
	if err != nil {
		return nil, err
	}

	fi := storagedriver.FileInfoFields{
		Path: path,
	}

	if len(resp.Contents) == 1 {
		if *resp.Contents[0].Key != d.s3Path(path) {
			fi.IsDir = true
		} else {
			fi.IsDir = false
			fi.Size = *resp.Contents[0].Size
			fi.ModTime = *resp.Contents[0].LastModified
		}
	} else if len(resp.CommonPrefixes) == 1 {
		fi.IsDir = true
	} else {
		return nil, storagedriver.PathNotFoundError{Path: path}
	}

	return storagedriver.FileInfoInternal{FileInfoFields: fi}, nil
}

// List returns a list of the objects that are direct descendants of the given path.
func (d *driver) List(ctx context.Context, opath string) ([]string, error) {
	path := opath
	if path != "/" && path[len(path)-1] != '/' {
		path = path + "/"
	}

	// This is to cover for the cases when the rootDirectory of the driver is either "" or "/".
	// In those cases, there is no root prefix to replace and we must actually add a "/" to all
	// results in order to keep them as valid paths as recognized by storagedriver.PathRegexp
	prefix := ""
	if d.s3Path("") == "" {
		prefix = "/"
	}

	resp, err := d.S3.ListObjects(&s3.ListObjectsInput{
		Bucket:    aws.String(d.Bucket),
		Prefix:    aws.String(d.s3Path(path)),
		Delimiter: aws.String("/"),
		MaxKeys:   aws.Int64(listMax),
	})
	if err != nil {
		return nil, parseError(opath, err)
	}

	files := []string{}
	directories := []string{}

	for {
		for _, key := range resp.Contents {
			files = append(files, strings.Replace(*key.Key, d.s3Path(""), prefix, 1))
		}

		for _, commonPrefix := range resp.CommonPrefixes {
			commonPrefix := *commonPrefix.Prefix
			directories = append(directories, strings.Replace(commonPrefix[0:len(commonPrefix)-1], d.s3Path(""), prefix, 1))
		}

		if *resp.IsTruncated {
			resp, err = d.S3.ListObjects(&s3.ListObjectsInput{
				Bucket:    aws.String(d.Bucket),
				Prefix:    aws.String(d.s3Path(path)),
				Delimiter: aws.String("/"),
				MaxKeys:   aws.Int64(listMax),
				Marker:    resp.NextMarker,
			})
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
	/* This is terrible, but aws doesn't have an actual move. */
	_, err := d.S3.CopyObject(&s3.CopyObjectInput{
		Bucket:               aws.String(d.Bucket),
		Key:                  aws.String(d.s3Path(destPath)),
		ContentType:          d.getContentType(),
		ACL:                  d.getACL(),
		ServerSideEncryption: d.getEncryptionMode(),
		SSEKMSKeyId:          d.getSSEKMSKeyID(),
		StorageClass:         d.getStorageClass(),
		CopySource:           aws.String(d.Bucket + "/" + d.s3Path(sourcePath)),
	})
	if err != nil {
		return parseError(sourcePath, err)
	}

	return d.Delete(ctx, sourcePath)
}

// Delete recursively deletes all objects stored at "path" and its subpaths.
func (d *driver) Delete(ctx context.Context, path string) error {
	resp, err := d.S3.ListObjects(&s3.ListObjectsInput{
		Bucket: aws.String(d.Bucket),
		Prefix: aws.String(d.s3Path(path)),
	})
	if err != nil || len(resp.Contents) == 0 {
		return storagedriver.PathNotFoundError{Path: path}
	}

	s3Objects := make([]*s3.ObjectIdentifier, 0, listMax)

	for len(resp.Contents) > 0 {
		for _, key := range resp.Contents {
			s3Objects = append(s3Objects, &s3.ObjectIdentifier{
				Key: key.Key,
			})
		}

		_, err := d.S3.DeleteObjects(&s3.DeleteObjectsInput{
			Bucket: aws.String(d.Bucket),
			Delete: &s3.Delete{
				Objects: s3Objects,
				Quiet:   aws.Bool(false),
			},
		})
		if err != nil {
			return nil
		}

		resp, err = d.S3.ListObjects(&s3.ListObjectsInput{
			Bucket: aws.String(d.Bucket),
			Prefix: aws.String(d.s3Path(path)),
		})
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
		if !ok || (methodString != "GET" && methodString != "HEAD") {
			return "", storagedriver.ErrUnsupportedMethod{}
		}
	}

	expiresIn := 20 * time.Minute
	expires, ok := options["expiry"]
	if ok {
		et, ok := expires.(time.Time)
		if ok {
			expiresIn = et.Sub(time.Now())
		}
	}

	var req *request.Request

	switch methodString {
	case "GET":
		req, _ = d.S3.GetObjectRequest(&s3.GetObjectInput{
			Bucket: aws.String(d.Bucket),
			Key:    aws.String(d.s3Path(path)),
		})
	case "HEAD":
		req, _ = d.S3.HeadObjectRequest(&s3.HeadObjectInput{
			Bucket: aws.String(d.Bucket),
			Key:    aws.String(d.s3Path(path)),
		})
	default:
		panic("unreachable")
	}

	return req.Presign(expiresIn)
}

func (d *driver) s3Path(path string) string {
	return strings.TrimLeft(strings.TrimRight(d.RootDirectory, "/")+path, "/")
}

// S3BucketKey returns the s3 bucket key for the given storage driver path.
func (d *Driver) S3BucketKey(path string) string {
	return d.StorageDriver.(*driver).s3Path(path)
}

func parseError(path string, err error) error {
	if s3Err, ok := err.(awserr.Error); ok && s3Err.Code() == "NoSuchKey" {
		return storagedriver.PathNotFoundError{Path: path}
	}

	return err
}

func (d *driver) getEncryptionMode() *string {
	if !d.Encrypt {
		return nil
	}
	if d.KeyID == "" {
		return aws.String("AES256")
	}
	return aws.String("aws:kms")
}

func (d *driver) getSSEKMSKeyID() *string {
	if d.KeyID != "" {
		return aws.String(d.KeyID)
	}
	return nil
}

func (d *driver) getContentType() *string {
	return aws.String("application/octet-stream")
}

func (d *driver) getACL() *string {
	return aws.String("private")
}

func (d *driver) getStorageClass() *string {
	return aws.String(d.StorageClass)
}

// writer attempts to upload parts to S3 in a buffered fashion where the last
// part is at least as large as the chunksize, so the multipart upload could be
// cleanly resumed in the future. This is violated if Close is called after less
// than a full chunk is written.
type writer struct {
	driver      *driver
	key         string
	uploadID    string
	parts       []*s3.Part
	size        int64
	readyPart   []byte
	pendingPart []byte
	closed      bool
	committed   bool
	cancelled   bool
}

func (d *driver) newWriter(key, uploadID string, parts []*s3.Part) storagedriver.FileWriter {
	var size int64
	for _, part := range parts {
		size += *part.Size
	}
	return &writer{
		driver:   d,
		key:      key,
		uploadID: uploadID,
		parts:    parts,
		size:     size,
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
	if len(w.parts) > 0 && int(*w.parts[len(w.parts)-1].Size) < minChunkSize {
		var completedParts []*s3.CompletedPart
		for _, part := range w.parts {
			completedParts = append(completedParts, &s3.CompletedPart{
				ETag:       part.ETag,
				PartNumber: part.PartNumber,
			})
		}
		_, err := w.driver.S3.CompleteMultipartUpload(&s3.CompleteMultipartUploadInput{
			Bucket:   aws.String(w.driver.Bucket),
			Key:      aws.String(w.key),
			UploadId: aws.String(w.uploadID),
			MultipartUpload: &s3.CompletedMultipartUpload{
				Parts: completedParts,
			},
		})
		if err != nil {
			w.driver.S3.AbortMultipartUpload(&s3.AbortMultipartUploadInput{
				Bucket:   aws.String(w.driver.Bucket),
				Key:      aws.String(w.key),
				UploadId: aws.String(w.uploadID),
			})
			return 0, err
		}

		resp, err := w.driver.S3.CreateMultipartUpload(&s3.CreateMultipartUploadInput{
			Bucket:               aws.String(w.driver.Bucket),
			Key:                  aws.String(w.key),
			ContentType:          w.driver.getContentType(),
			ACL:                  w.driver.getACL(),
			ServerSideEncryption: w.driver.getEncryptionMode(),
			StorageClass:         w.driver.getStorageClass(),
		})
		if err != nil {
			return 0, err
		}
		w.uploadID = *resp.UploadId

		// If the entire written file is smaller than minChunkSize, we need to make
		// a new part from scratch :double sad face:
		if w.size < minChunkSize {
			resp, err := w.driver.S3.GetObject(&s3.GetObjectInput{
				Bucket: aws.String(w.driver.Bucket),
				Key:    aws.String(w.key),
			})
			defer resp.Body.Close()
			if err != nil {
				return 0, err
			}
			w.parts = nil
			w.readyPart, err = ioutil.ReadAll(resp.Body)
			if err != nil {
				return 0, err
			}
		} else {
			// Otherwise we can use the old file as the new first part
			copyPartResp, err := w.driver.S3.UploadPartCopy(&s3.UploadPartCopyInput{
				Bucket:     aws.String(w.driver.Bucket),
				CopySource: aws.String(w.driver.Bucket + "/" + w.key),
				Key:        aws.String(w.key),
				PartNumber: aws.Int64(1),
				UploadId:   resp.UploadId,
			})
			if err != nil {
				return 0, err
			}
			w.parts = []*s3.Part{
				{
					ETag:       copyPartResp.CopyPartResult.ETag,
					PartNumber: aws.Int64(1),
					Size:       aws.Int64(w.size),
				},
			}
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
	_, err := w.driver.S3.AbortMultipartUpload(&s3.AbortMultipartUploadInput{
		Bucket:   aws.String(w.driver.Bucket),
		Key:      aws.String(w.key),
		UploadId: aws.String(w.uploadID),
	})
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
	var completedParts []*s3.CompletedPart
	for _, part := range w.parts {
		completedParts = append(completedParts, &s3.CompletedPart{
			ETag:       part.ETag,
			PartNumber: part.PartNumber,
		})
	}
	_, err = w.driver.S3.CompleteMultipartUpload(&s3.CompleteMultipartUploadInput{
		Bucket:   aws.String(w.driver.Bucket),
		Key:      aws.String(w.key),
		UploadId: aws.String(w.uploadID),
		MultipartUpload: &s3.CompletedMultipartUpload{
			Parts: completedParts,
		},
	})
	if err != nil {
		w.driver.S3.AbortMultipartUpload(&s3.AbortMultipartUploadInput{
			Bucket:   aws.String(w.driver.Bucket),
			Key:      aws.String(w.key),
			UploadId: aws.String(w.uploadID),
		})
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

	partNumber := aws.Int64(int64(len(w.parts) + 1))
	resp, err := w.driver.S3.UploadPart(&s3.UploadPartInput{
		Bucket:     aws.String(w.driver.Bucket),
		Key:        aws.String(w.key),
		PartNumber: partNumber,
		UploadId:   aws.String(w.uploadID),
		Body:       bytes.NewReader(w.readyPart),
	})
	if err != nil {
		return err
	}
	w.parts = append(w.parts, &s3.Part{
		ETag:       resp.ETag,
		PartNumber: partNumber,
		Size:       aws.Int64(int64(len(w.readyPart))),
	})
	w.readyPart = w.pendingPart
	w.pendingPart = nil
	return nil
}
