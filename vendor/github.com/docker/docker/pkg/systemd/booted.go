package systemd

import (
	"os"
)

// Conversion to Go of systemd's sd_booted()
func SdBooted() bool {
	s, err := os.Stat("/run/systemd/system")
	if err != nil {
		return false
	}

	return s.IsDir()
}
