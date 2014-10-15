package osutil

import (
	"fmt"
	"github.com/rackspace/gophercloud"
	"os"
	"strings"
)

var (
	nilOptions = gophercloud.AuthOptions{}

	// ErrNoAuthUrl errors occur when the value of the OS_AUTH_URL environment variable cannot be determined.
	ErrNoAuthUrl = fmt.Errorf("Environment variable OS_AUTH_URL needs to be set.")

	// ErrNoUsername errors occur when the value of the OS_USERNAME environment variable cannot be determined.
	ErrNoUsername = fmt.Errorf("Environment variable OS_USERNAME needs to be set.")

	// ErrNoPassword errors occur when the value of the OS_PASSWORD environment variable cannot be determined.
	ErrNoPassword = fmt.Errorf("Environment variable OS_PASSWORD or OS_API_KEY needs to be set.")
)

// AuthOptions fills out a gophercloud.AuthOptions structure with the settings found on the various OpenStack
// OS_* environment variables.  The following variables provide sources of truth: OS_AUTH_URL, OS_USERNAME,
// OS_PASSWORD, OS_TENANT_ID, and OS_TENANT_NAME.  Of these, OS_USERNAME, OS_PASSWORD, and OS_AUTH_URL must
// have settings, or an error will result.  OS_TENANT_ID and OS_TENANT_NAME are optional.
//
// The value of OS_AUTH_URL will be returned directly to the caller, for subsequent use in
// gophercloud.Authenticate()'s Provider parameter.  This function will not interpret the value of OS_AUTH_URL,
// so as a convenient extention, you may set OS_AUTH_URL to, e.g., "rackspace-uk", or any other Gophercloud-recognized
// provider shortcuts.  For broad compatibility, especially with local installations, you should probably
// avoid the temptation to do this.
func AuthOptions() (string, gophercloud.AuthOptions, error) {
	provider := os.Getenv("OS_AUTH_URL")
	username := os.Getenv("OS_USERNAME")
	password := os.Getenv("OS_PASSWORD")
	tenantId := os.Getenv("OS_TENANT_ID")
	tenantName := os.Getenv("OS_TENANT_NAME")

	if provider == "" {
		return "", nilOptions, ErrNoAuthUrl
	}

	if username == "" {
		return "", nilOptions, ErrNoUsername
	}

	if password == "" {
		return "", nilOptions, ErrNoPassword
	}

	ao := gophercloud.AuthOptions{
		Username:   username,
		Password:   password,
		TenantId:   tenantId,
		TenantName: tenantName,
	}

	if !strings.HasSuffix(provider, "/tokens") {
		provider += "/tokens"
	}

	return provider, ao, nil
}
