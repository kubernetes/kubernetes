package nstld

import (
    "fmt"
)

// CommonTDLs contains a set of common top-level domains that should be
// avoided when naming Kubernetes namespaces to prevent DNS resolution issues.
var CommonTDLs = map[string]struct{}{
    "com":  {},
    "org":  {},
    "net":  {},
    "edu":  {},
    "gov":  {},
    "dev":  {},
    "co":   {},
    "info": {},
    "biz":  {},
}

// IsTLD checks if the provided namespace name is a common top-level domain.
// Returns true if the namespace name matches a known TLD.
func IsTLD(nsName string) bool {
    _, isTLD := CommonTDLs[nsName]
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