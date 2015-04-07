package rackspace

import (
	"fmt"
	"os"

	"github.com/rackspace/gophercloud"
)

var nilOptions = gophercloud.AuthOptions{}

// ErrNoAuthUrl, ErrNoUsername, and ErrNoPassword errors indicate of the
// required RS_AUTH_URL, RS_USERNAME, or RS_PASSWORD environment variables,
// respectively, remain undefined.  See the AuthOptions() function for more details.
var (
	ErrNoAuthURL  = fmt.Errorf("Environment variable RS_AUTH_URL or OS_AUTH_URL need to be set.")
	ErrNoUsername = fmt.Errorf("Environment variable RS_USERNAME or OS_USERNAME need to be set.")
	ErrNoPassword = fmt.Errorf("Environment variable RS_API_KEY or RS_PASSWORD needs to be set.")
)

func prefixedEnv(base string) string {
	value := os.Getenv("RS_" + base)
	if value == "" {
		value = os.Getenv("OS_" + base)
	}
	return value
}

// AuthOptionsFromEnv fills out an identity.AuthOptions structure with the
// settings found on the various Rackspace RS_* environment variables.
func AuthOptionsFromEnv() (gophercloud.AuthOptions, error) {
	authURL := prefixedEnv("AUTH_URL")
	username := prefixedEnv("USERNAME")
	password := prefixedEnv("PASSWORD")
	apiKey := prefixedEnv("API_KEY")

	if authURL == "" {
		return nilOptions, ErrNoAuthURL
	}

	if username == "" {
		return nilOptions, ErrNoUsername
	}

	if password == "" && apiKey == "" {
		return nilOptions, ErrNoPassword
	}

	ao := gophercloud.AuthOptions{
		IdentityEndpoint: authURL,
		Username:         username,
		Password:         password,
		APIKey:           apiKey,
	}

	return ao, nil
}
