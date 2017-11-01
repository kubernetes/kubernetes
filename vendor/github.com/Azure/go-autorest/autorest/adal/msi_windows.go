// +build windows

package adal

import (
	"os"
	"strings"
)

// msiPath is the path to the MSI Extension settings file (to discover the endpoint)
var msiPath = strings.Join([]string{os.Getenv("SystemDrive"), "WindowsAzure/Config/ManagedIdentity-Settings"}, "/")
