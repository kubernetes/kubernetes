package s3

import (
	"io/ioutil"
	"os"
	"strconv"
	"testing"

	"github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/testsuites"
	"github.com/docker/goamz/aws"
	"github.com/docker/goamz/s3"

	"gopkg.in/check.v1"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

var s3DriverConstructor func(rootDirectory string, storageClass s3.StorageClass) (*Driver, error)
var skipS3 func() string

func init() {
	accessKey := os.Getenv("AWS_ACCESS_KEY")
	secretKey := os.Getenv("AWS_SECRET_KEY")
	bucket := os.Getenv("S3_BUCKET")
	encrypt := os.Getenv("S3_ENCRYPT")
	secure := os.Getenv("S3_SECURE")
	v4auth := os.Getenv("S3_USE_V4_AUTH")
	region := os.Getenv("AWS_REGION")
	root, err := ioutil.TempDir("", "driver-")
	if err != nil {
		panic(err)
	}
	defer os.Remove(root)

	s3DriverConstructor = func(rootDirectory string, storageClass s3.StorageClass) (*Driver, error) {
		encryptBool := false
		if encrypt != "" {
			encryptBool, err = strconv.ParseBool(encrypt)
			if err != nil {
				return nil, err
			}
		}

		secureBool := true
		if secure != "" {
			secureBool, err = strconv.ParseBool(secure)
			if err != nil {
				return nil, err
			}
		}

		v4AuthBool := false
		if v4auth != "" {
			v4AuthBool, err = strconv.ParseBool(v4auth)
			if err != nil {
				return nil, err
			}
		}

		parameters := DriverParameters{
			accessKey,
			secretKey,
			bucket,
			aws.GetRegion(region),
			encryptBool,
			secureBool,
			v4AuthBool,
			minChunkSize,
			rootDirectory,
			storageClass,
			driverName + "-test",
		}

		return New(parameters)
	}

	// Skip S3 storage driver tests if environment variable parameters are not provided
	skipS3 = func() string {
		if accessKey == "" || secretKey == "" || region == "" || bucket == "" || encrypt == "" {
			return "Must set AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET, and S3_ENCRYPT to run S3 tests"
		}
		return ""
	}

	testsuites.RegisterSuite(func() (storagedriver.StorageDriver, error) {
		return s3DriverConstructor(root, s3.StandardStorage)
	}, skipS3)
}

func TestEmptyRootList(t *testing.T) {
	if skipS3() != "" {
		t.Skip(skipS3())
	}

	validRoot, err := ioutil.TempDir("", "driver-")
	if err != nil {
		t.Fatalf("unexpected error creating temporary directory: %v", err)
	}
	defer os.Remove(validRoot)

	rootedDriver, err := s3DriverConstructor(validRoot, s3.StandardStorage)
	if err != nil {
		t.Fatalf("unexpected error creating rooted driver: %v", err)
	}

	emptyRootDriver, err := s3DriverConstructor("", s3.StandardStorage)
	if err != nil {
		t.Fatalf("unexpected error creating empty root driver: %v", err)
	}

	slashRootDriver, err := s3DriverConstructor("/", s3.StandardStorage)
	if err != nil {
		t.Fatalf("unexpected error creating slash root driver: %v", err)
	}

	filename := "/test"
	contents := []byte("contents")
	ctx := context.Background()
	err = rootedDriver.PutContent(ctx, filename, contents)
	if err != nil {
		t.Fatalf("unexpected error creating content: %v", err)
	}
	defer rootedDriver.Delete(ctx, filename)

	keys, err := emptyRootDriver.List(ctx, "/")
	for _, path := range keys {
		if !storagedriver.PathRegexp.MatchString(path) {
			t.Fatalf("unexpected string in path: %q != %q", path, storagedriver.PathRegexp)
		}
	}

	keys, err = slashRootDriver.List(ctx, "/")
	for _, path := range keys {
		if !storagedriver.PathRegexp.MatchString(path) {
			t.Fatalf("unexpected string in path: %q != %q", path, storagedriver.PathRegexp)
		}
	}
}

func TestStorageClass(t *testing.T) {
	if skipS3() != "" {
		t.Skip(skipS3())
	}

	rootDir, err := ioutil.TempDir("", "driver-")
	if err != nil {
		t.Fatalf("unexpected error creating temporary directory: %v", err)
	}
	defer os.Remove(rootDir)

	standardDriver, err := s3DriverConstructor(rootDir, s3.StandardStorage)
	if err != nil {
		t.Fatalf("unexpected error creating driver with standard storage: %v", err)
	}

	rrDriver, err := s3DriverConstructor(rootDir, s3.ReducedRedundancy)
	if err != nil {
		t.Fatalf("unexpected error creating driver with reduced redundancy storage: %v", err)
	}

	standardFilename := "/test-standard"
	rrFilename := "/test-rr"
	contents := []byte("contents")
	ctx := context.Background()

	err = standardDriver.PutContent(ctx, standardFilename, contents)
	if err != nil {
		t.Fatalf("unexpected error creating content: %v", err)
	}
	defer standardDriver.Delete(ctx, standardFilename)

	err = rrDriver.PutContent(ctx, rrFilename, contents)
	if err != nil {
		t.Fatalf("unexpected error creating content: %v", err)
	}
	defer rrDriver.Delete(ctx, rrFilename)

	standardDriverUnwrapped := standardDriver.Base.StorageDriver.(*driver)
	resp, err := standardDriverUnwrapped.Bucket.GetResponse(standardDriverUnwrapped.s3Path(standardFilename))
	if err != nil {
		t.Fatalf("unexpected error retrieving standard storage file: %v", err)
	}
	defer resp.Body.Close()
	// Amazon only populates this header value for non-standard storage classes
	if storageClass := resp.Header.Get("x-amz-storage-class"); storageClass != "" {
		t.Fatalf("unexpected storage class for standard file: %v", storageClass)
	}

	rrDriverUnwrapped := rrDriver.Base.StorageDriver.(*driver)
	resp, err = rrDriverUnwrapped.Bucket.GetResponse(rrDriverUnwrapped.s3Path(rrFilename))
	if err != nil {
		t.Fatalf("unexpected error retrieving reduced-redundancy storage file: %v", err)
	}
	defer resp.Body.Close()
	if storageClass := resp.Header.Get("x-amz-storage-class"); storageClass != string(s3.ReducedRedundancy) {
		t.Fatalf("unexpected storage class for reduced-redundancy file: %v", storageClass)
	}
}
