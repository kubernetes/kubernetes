package openstack

import (
	"fmt"
	"os"

	"github.com/rackspace/gophercloud"
)

var nilOptions = gophercloud.AuthOptions{}

// ErrNoAuthUrl, ErrNoUsername, and ErrNoPassword errors indicate of the required OS_AUTH_URL, OS_USERNAME, or OS_PASSWORD
// environment variables, respectively, remain undefined.  See the AuthOptions() function for more details.
var (
	ErrNoAuthURL  = fmt.Errorf("Environment variable OS_AUTH_URL needs to be set.")
	ErrNoUsername = fmt.Errorf("Environment variable OS_USERNAME, OS_USERID, or OS_TOKEN needs to be set.")
	ErrNoPassword = fmt.Errorf("Environment variable OS_PASSWORD or OS_TOKEN needs to be set.")
)

// AuthOptionsFromEnv fills out an AuthOptions structure from the environment
// variables: OS_AUTH_URL, OS_USERNAME, OS_USERID, OS_PASSWORD, OS_TENANT_ID,
// OS_TENANT_NAME, OS_DOMAIN_ID, OS_DOMAIN_NAME, OS_TOKEN.  It checks that
// (1) OS_AUTH_URL is set, (2) OS_USERNAME, OS_USERID, or OS_TOKEN is set,
// (3) OS_PASSWORD or OS_TOKEN is set.
func AuthOptionsFromEnv() (gophercloud.AuthOptions, error) {
	authURL := os.Getenv("OS_AUTH_URL")
	username := os.Getenv("OS_USERNAME")
	userID := os.Getenv("OS_USERID")
	password := os.Getenv("OS_PASSWORD")
	tenantID := os.Getenv("OS_TENANT_ID")
	tenantName := os.Getenv("OS_TENANT_NAME")
	domainID := os.Getenv("OS_DOMAIN_ID")
	domainName := os.Getenv("OS_DOMAIN_NAME")
	tokenID := os.Getenv("OS_TOKEN")

	if authURL == "" {
		return nilOptions, ErrNoAuthURL
	}

	if username == "" && userID == "" && tokenID == "" {
		return nilOptions, ErrNoUsername
	}

	if password == "" && tokenID == "" {
		return nilOptions, ErrNoPassword
	}

	ao := gophercloud.AuthOptions{
		IdentityEndpoint: authURL,
		UserID:           userID,
		Username:         username,
		Password:         password,
		TenantID:         tenantID,
		TenantName:       tenantName,
		DomainID:         domainID,
		DomainName:       domainName,
		TokenID:          tokenID,
	}

	return ao, nil
}
