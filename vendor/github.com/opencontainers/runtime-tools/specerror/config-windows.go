package specerror

import (
	"fmt"

	rfc2119 "github.com/opencontainers/runtime-tools/error"
)

// define error codes
const (
	// WindowsLayerFoldersRequired represents "`layerFolders` MUST contain at least one entry."
	WindowsLayerFoldersRequired Code = 0xd001 + iota
	// WindowsHyperVPresent represents "If present, the container MUST be run with Hyper-V isolation."
	WindowsHyperVPresent
	// WindowsHyperVOmit represents "If omitted, the container MUST be run as a Windows Server container."
	WindowsHyperVOmit
)

var (
	layerfoldersRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-windows.md#layerfolders"), nil
	}
	hypervRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-windows.md#hyperv"), nil
	}
)

func init() {
	register(WindowsLayerFoldersRequired, rfc2119.Must, layerfoldersRef)
	register(WindowsHyperVPresent, rfc2119.Must, hypervRef)
	register(WindowsHyperVOmit, rfc2119.Must, hypervRef)
}
