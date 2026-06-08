package apparmor

import (
	"os"
	"sync"
)

var (
	appArmorEnabled bool
	checkAppArmor   sync.Once
)

// IsEnabled returns true if apparmor is enabled for the host.
func IsEnabled() bool {
	checkAppArmor.Do(func() {
		if _, err := os.Stat("/sys/kernel/security/apparmor"); err == nil {
			buf, err := os.ReadFile("/sys/module/apparmor/parameters/enabled")
			appArmorEnabled = err == nil && len(buf) > 1 && buf[0] == 'Y'
		}
	})
	return appArmorEnabled
}
