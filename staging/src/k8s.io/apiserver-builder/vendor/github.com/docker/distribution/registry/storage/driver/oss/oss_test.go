// +build include_oss

package oss

import (
	"io/ioutil"

	alioss "github.com/denverdino/aliyungo/oss"
	"github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/testsuites"
	//"log"
	"os"
	"strconv"
	"testing"

	"gopkg.in/check.v1"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

var ossDriverConstructor func(rootDirectory string) (*Driver, error)

var skipCheck func() string

func init() {
	accessKey := os.Getenv("ALIYUN_ACCESS_KEY_ID")
	secretKey := os.Getenv("ALIYUN_ACCESS_KEY_SECRET")
	bucket := os.Getenv("OSS_BUCKET")
	region := os.Getenv("OSS_REGION")
	internal := os.Getenv("OSS_INTERNAL")
	encrypt := os.Getenv("OSS_ENCRYPT")
	secure := os.Getenv("OSS_SECURE")
	endpoint := os.Getenv("OSS_ENDPOINT")
	root, err := ioutil.TempDir("", "driver-")
	if err != nil {
		panic(err)
	}
	defer os.Remove(root)

	ossDriverConstructor = func(rootDirectory string) (*Driver, error) {
		encryptBool := false
		if encrypt != "" {
			encryptBool, err = strconv.ParseBool(encrypt)
			if err != nil {
				return nil, err
			}
		}

		secureBool := false
		if secure != "" {
			secureBool, err = strconv.ParseBool(secure)
			if err != nil {
				return nil, err
			}
		}

		internalBool := false
		if internal != "" {
			internalBool, err = strconv.ParseBool(internal)
			if err != nil {
				return nil, err
			}
		}

		parameters := DriverParameters{
			AccessKeyID:     accessKey,
			AccessKeySecret: secretKey,
			Bucket:          bucket,
			Region:          alioss.Region(region),
			Internal:        internalBool,
			ChunkSize:       minChunkSize,
			RootDirectory:   rootDirectory,
			Encrypt:         encryptBool,
			Secure:          secureBool,
			Endpoint:        endpoint,
		}

		return New(parameters)
	}

	// Skip OSS storage driver tests if environment variable parameters are not provided
	skipCheck = func() string {
		if accessKey == "" || secretKey == "" || region == "" || bucket == "" || encrypt == "" {
			return "Must set ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, OSS_REGION, OSS_BUCKET, and OSS_ENCRYPT to run OSS tests"
		}
		return ""
	}

	testsuites.RegisterSuite(func() (storagedriver.StorageDriver, error) {
		return ossDriverConstructor(root)
	}, skipCheck)
}

func TestEmptyRootList(t *testing.T) {
	if skipCheck() != "" {
		t.Skip(skipCheck())
	}

	validRoot, err := ioutil.TempDir("", "driver-")
	if err != nil {
		t.Fatalf("unexpected error creating temporary directory: %v", err)
	}
	defer os.Remove(validRoot)

	rootedDriver, err := ossDriverConstructor(validRoot)
	if err != nil {
		t.Fatalf("unexpected error creating rooted driver: %v", err)
	}

	emptyRootDriver, err := ossDriverConstructor("")
	if err != nil {
		t.Fatalf("unexpected error creating empty root driver: %v", err)
	}

	slashRootDriver, err := ossDriverConstructor("/")
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
