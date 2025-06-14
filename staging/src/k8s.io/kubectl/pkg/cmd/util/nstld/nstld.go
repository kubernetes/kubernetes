package nstld

import (
	"fmt"
	"os"
	"strings"
)

// CommonTLDs contains a set of common top-level domains that should be
// avoided when naming Kubernetes namespaces to prevent DNS resolution issues.
// This list can be extended by setting the KUBECTL_ADDITIONAL_TLDS environment
// variable with a comma-separated list of additional TLDs.
var CommonTLDs = map[string]struct{}{
	"com":  {},
	"org":  {},
	"net":  {},
	"edu":  {},
	"gov":  {},
	"dev":  {},
	"io": {},
}

func init() {
	// Allow extending the TLD list via environment variable
	if additionalTLDs := os.Getenv("KUBECTL_ADDITIONAL_TLDS"); additionalTLDs != "" {
		for _, tld := range strings.Split(additionalTLDs, ",") {
			tld = strings.TrimSpace(tld)
			if tld != "" {
				CommonTLDs[tld] = struct{}{}
			}
		}
	}
}

// IsTLD checks if the provided namespace name is a common top-level domain.
// Returns true if the namespace name matches a known TLD.
func IsTLD(nsName string) bool {
	_, isTLD := CommonTLDs[nsName]
	return isTLD
}

// GetTLDWarningMessage returns a warning message if the namespace name
// is a common top-level domain. If the name is not a TLD, returns an empty string.
func GetTLDWarningMessage(nsName string) string {
	if IsTLD(nsName) {
		return fmt.Sprintf("Warning: Namespace name '%s' is a top-level domain (TLD). This may cause issues with DNS resolution.\n", nsName)
	}
	return ""
}
