package operatingsystem // import "github.com/docker/docker/pkg/parsers/operatingsystem"

import (
	"fmt"

	"golang.org/x/sys/windows/registry"
)

// GetOperatingSystem gets the name of the current operating system.
func GetOperatingSystem() (string, error) {

	// Default return value
	ret := "Unknown Operating System"

	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion`, registry.QUERY_VALUE)
	if err != nil {
		return ret, err
	}
	defer k.Close()

	pn, _, err := k.GetStringValue("ProductName")
	if err != nil {
		return ret, err
	}
	ret = pn

	ri, _, err := k.GetStringValue("ReleaseId")
	if err != nil {
		return ret, err
	}
	ret = fmt.Sprintf("%s Version %s", ret, ri)

	cbn, _, err := k.GetStringValue("CurrentBuildNumber")
	if err != nil {
		return ret, err
	}

	ubr, _, err := k.GetIntegerValue("UBR")
	if err != nil {
		return ret, err
	}
	ret = fmt.Sprintf("%s (OS Build %s.%d)", ret, cbn, ubr)

	return ret, nil
}

// IsContainerized returns true if we are running inside a container.
// No-op on Windows, always returns false.
func IsContainerized() (bool, error) {
	return false, nil
}
