// +build linux darwin

package sequences

import (
	"fmt"
)

func EnableVirtualTerminalProcessing(stream uintptr, enable bool) error {
	return fmt.Errorf("windows only package")
}
