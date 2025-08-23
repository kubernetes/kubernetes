package credentials

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"
	"sync/atomic"

	"github.com/docker/cli/cli/config/types"
)

type store interface {
	Save() error
	GetAuthConfigs() map[string]types.AuthConfig
	GetFilename() string
}

// fileStore implements a credentials store using
// the docker configuration file to keep the credentials in plain text.
type fileStore struct {
	file store
}

// NewFileStore creates a new file credentials store.
func NewFileStore(file store) Store {
	return &fileStore{file: file}
}

// Erase removes the given credentials from the file store.This function is
// idempotent and does not update the file if credentials did not change.
func (c *fileStore) Erase(serverAddress string) error {
	if _, exists := c.file.GetAuthConfigs()[serverAddress]; !exists {
		// nothing to do; no credentials found for the given serverAddress
		return nil
	}
	delete(c.file.GetAuthConfigs(), serverAddress)
	return c.file.Save()
}

// Get retrieves credentials for a specific server from the file store.
func (c *fileStore) Get(serverAddress string) (types.AuthConfig, error) {
	authConfig, ok := c.file.GetAuthConfigs()[serverAddress]
	if !ok {
		// Maybe they have a legacy config file, we will iterate the keys converting
		// them to the new format and testing
		for r, ac := range c.file.GetAuthConfigs() {
			if serverAddress == ConvertToHostname(r) {
				return ac, nil
			}
		}

		authConfig = types.AuthConfig{}
	}
	return authConfig, nil
}

func (c *fileStore) GetAll() (map[string]types.AuthConfig, error) {
	return c.file.GetAuthConfigs(), nil
}

// unencryptedWarning warns the user when using an insecure credential storage.
// After a deprecation period, user will get prompted if stdin and stderr are a terminal.
// Otherwise, we'll assume they want it (sadly), because people may have been scripting
// insecure logins and we don't want to break them. Maybe they'll see the warning in their
// logs and fix things.
const unencryptedWarning = `
WARNING! Your credentials are stored unencrypted in '%s'.
Configure a credential helper to remove this warning. See
https://docs.docker.com/go/credential-store/
`

// alreadyPrinted ensures that we only print the unencryptedWarning once per
// CLI invocation (no need to warn the user multiple times per command).
var alreadyPrinted atomic.Bool

// Store saves the given credentials in the file store. This function is
// idempotent and does not update the file if credentials did not change.
func (c *fileStore) Store(authConfig types.AuthConfig) error {
	authConfigs := c.file.GetAuthConfigs()
	if oldAuthConfig, ok := authConfigs[authConfig.ServerAddress]; ok && oldAuthConfig == authConfig {
		// Credentials didn't change, so skip updating the configuration file.
		return nil
	}
	authConfigs[authConfig.ServerAddress] = authConfig
	if err := c.file.Save(); err != nil {
		return err
	}

	if !alreadyPrinted.Load() && authConfig.Password != "" {
		// Display a warning if we're storing the users password (not a token).
		//
		// FIXME(thaJeztah): make output configurable instead of hardcoding to os.Stderr
		_, _ = fmt.Fprintln(os.Stderr, fmt.Sprintf(unencryptedWarning, c.file.GetFilename()))
		alreadyPrinted.Store(true)
	}

	return nil
}

// ConvertToHostname converts a registry url which has http|https prepended
// to just an hostname.
// Copied from github.com/docker/docker/registry.ConvertToHostname to reduce dependencies.
func ConvertToHostname(maybeURL string) string {
	stripped := maybeURL
	if strings.Contains(stripped, "://") {
		u, err := url.Parse(stripped)
		if err == nil && u.Hostname() != "" {
			if u.Port() == "" {
				return u.Hostname()
			}
			return net.JoinHostPort(u.Hostname(), u.Port())
		}
	}
	hostName, _, _ := strings.Cut(stripped, "/")
	return hostName
}
