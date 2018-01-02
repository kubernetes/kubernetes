// +build example,go18

package main

import (
	"encoding/json"
	"os"

	"github.com/pkg/errors"
)

// Example plugin that will retrieve credentials from a JSON file that the
// "PLUGIN_CREDS_FILE" environment variable points to
//
// Build with:
//   go build -tags example -o plugin.so -buildmode=plugin plugin.go
func main() {}

var myCredProvider provider

func init() {
	// Initialize a mock credential provider with stubs
	myCredProvider = provider{Filename: os.Getenv("PLUGIN_CREDS_FILE")}
}

// GetAWSSDKCredentialProvider is the symbol SDK will lookup and use to
// get the credential provider's retrieve and isExpired functions.
func GetAWSSDKCredentialProvider() (func() (key, secret, token string, err error), func() bool) {
	return myCredProvider.Retrieve, myCredProvider.IsExpired
}

// mock implementation of a type that returns retrieves credentials and
// returns if they have expired.
type provider struct {
	Filename string

	loaded bool
}

func (p *provider) Retrieve() (key, secret, token string, err error) {
	f, err := os.Open(p.Filename)
	if err != nil {
		return "", "", "", errors.Wrapf(err, "failed to open credentials file, %q", p.Filename)
	}
	decoder := json.NewDecoder(f)

	creds := struct {
		Key, Secret, Token string
	}{}

	if err := decoder.Decode(&creds); err != nil {
		return "", "", "", errors.Wrap(err, "failed to decode credentials file")
	}

	p.loaded = true
	return creds.Key, creds.Secret, creds.Token, nil
}

func (p *provider) IsExpired() bool {
	return !p.loaded
}
