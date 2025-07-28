package credentials

import "os/exec"

// DetectDefaultStore return the default credentials store for the platform if
// no user-defined store is passed, and the store executable is available.
func DetectDefaultStore(store string) string {
	if store != "" {
		// use user-defined
		return store
	}

	platformDefault := defaultCredentialsStore()
	if platformDefault == "" {
		return ""
	}

	if _, err := exec.LookPath(remoteCredentialsPrefix + platformDefault); err != nil {
		return ""
	}
	return platformDefault
}
