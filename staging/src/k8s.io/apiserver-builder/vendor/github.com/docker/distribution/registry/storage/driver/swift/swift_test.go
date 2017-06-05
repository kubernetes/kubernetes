package swift

import (
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/ncw/swift/swifttest"

	"github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/testsuites"

	"gopkg.in/check.v1"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

var swiftDriverConstructor func(prefix string) (*Driver, error)

func init() {
	var (
		username           string
		password           string
		authURL            string
		tenant             string
		tenantID           string
		domain             string
		domainID           string
		trustID            string
		container          string
		region             string
		insecureSkipVerify bool
		secretKey          string
		accessKey          string
		containerKey       bool
		tempURLMethods     []string

		swiftServer *swifttest.SwiftServer
		err         error
	)
	username = os.Getenv("SWIFT_USERNAME")
	password = os.Getenv("SWIFT_PASSWORD")
	authURL = os.Getenv("SWIFT_AUTH_URL")
	tenant = os.Getenv("SWIFT_TENANT_NAME")
	tenantID = os.Getenv("SWIFT_TENANT_ID")
	domain = os.Getenv("SWIFT_DOMAIN_NAME")
	domainID = os.Getenv("SWIFT_DOMAIN_ID")
	trustID = os.Getenv("SWIFT_TRUST_ID")
	container = os.Getenv("SWIFT_CONTAINER_NAME")
	region = os.Getenv("SWIFT_REGION_NAME")
	insecureSkipVerify, _ = strconv.ParseBool(os.Getenv("SWIFT_INSECURESKIPVERIFY"))
	secretKey = os.Getenv("SWIFT_SECRET_KEY")
	accessKey = os.Getenv("SWIFT_ACCESS_KEY")
	containerKey, _ = strconv.ParseBool(os.Getenv("SWIFT_TEMPURL_CONTAINERKEY"))
	tempURLMethods = strings.Split(os.Getenv("SWIFT_TEMPURL_METHODS"), ",")

	if username == "" || password == "" || authURL == "" || container == "" {
		if swiftServer, err = swifttest.NewSwiftServer("localhost"); err != nil {
			panic(err)
		}
		username = "swifttest"
		password = "swifttest"
		authURL = swiftServer.AuthURL
		container = "test"
	}

	prefix, err := ioutil.TempDir("", "driver-")
	if err != nil {
		panic(err)
	}
	defer os.Remove(prefix)

	swiftDriverConstructor = func(root string) (*Driver, error) {
		parameters := Parameters{
			username,
			password,
			authURL,
			tenant,
			tenantID,
			domain,
			domainID,
			trustID,
			region,
			container,
			root,
			insecureSkipVerify,
			defaultChunkSize,
			secretKey,
			accessKey,
			containerKey,
			tempURLMethods,
		}

		return New(parameters)
	}

	driverConstructor := func() (storagedriver.StorageDriver, error) {
		return swiftDriverConstructor(prefix)
	}

	testsuites.RegisterSuite(driverConstructor, testsuites.NeverSkip)
}

func TestEmptyRootList(t *testing.T) {
	validRoot, err := ioutil.TempDir("", "driver-")
	if err != nil {
		t.Fatalf("unexpected error creating temporary directory: %v", err)
	}
	defer os.Remove(validRoot)

	rootedDriver, err := swiftDriverConstructor(validRoot)
	if err != nil {
		t.Fatalf("unexpected error creating rooted driver: %v", err)
	}

	emptyRootDriver, err := swiftDriverConstructor("")
	if err != nil {
		t.Fatalf("unexpected error creating empty root driver: %v", err)
	}

	slashRootDriver, err := swiftDriverConstructor("/")
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

	// Create an object with a path nested under the existing object
	err = rootedDriver.PutContent(ctx, filename+"/file1", contents)
	if err != nil {
		t.Fatalf("unexpected error creating content: %v", err)
	}

	err = rootedDriver.Delete(ctx, filename)
	if err != nil {
		t.Fatalf("failed to delete: %v", err)
	}

	keys, err = rootedDriver.List(ctx, "/")
	if err != nil {
		t.Fatalf("failed to list objects after deletion: %v", err)
	}

	if len(keys) != 0 {
		t.Fatal("delete did not remove nested objects")
	}
}
