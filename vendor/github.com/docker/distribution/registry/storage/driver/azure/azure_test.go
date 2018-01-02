package azure

import (
	"fmt"
	"os"
	"strings"
	"testing"

	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/testsuites"
	. "gopkg.in/check.v1"
)

const (
	envAccountName = "AZURE_STORAGE_ACCOUNT_NAME"
	envAccountKey  = "AZURE_STORAGE_ACCOUNT_KEY"
	envContainer   = "AZURE_STORAGE_CONTAINER"
	envRealm       = "AZURE_STORAGE_REALM"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { TestingT(t) }

func init() {
	var (
		accountName string
		accountKey  string
		container   string
		realm       string
	)

	config := []struct {
		env   string
		value *string
	}{
		{envAccountName, &accountName},
		{envAccountKey, &accountKey},
		{envContainer, &container},
		{envRealm, &realm},
	}

	missing := []string{}
	for _, v := range config {
		*v.value = os.Getenv(v.env)
		if *v.value == "" {
			missing = append(missing, v.env)
		}
	}

	azureDriverConstructor := func() (storagedriver.StorageDriver, error) {
		return New(accountName, accountKey, container, realm)
	}

	// Skip Azure storage driver tests if environment variable parameters are not provided
	skipCheck := func() string {
		if len(missing) > 0 {
			return fmt.Sprintf("Must set %s environment variables to run Azure tests", strings.Join(missing, ", "))
		}
		return ""
	}

	testsuites.RegisterSuite(azureDriverConstructor, skipCheck)
}
