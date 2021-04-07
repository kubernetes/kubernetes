package apparmor

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/opencontainers/runc/libcontainer/utils"
)

// IsEnabled returns true if apparmor is enabled for the host.
func IsEnabled() bool {
	if _, err := os.Stat("/sys/kernel/security/apparmor"); err == nil {
		buf, err := ioutil.ReadFile("/sys/module/apparmor/parameters/enabled")
		return err == nil && bytes.HasPrefix(buf, []byte("Y"))
	}
	return false
}

func setProcAttr(attr, value string) error {
	// Under AppArmor you can only change your own attr, so use /proc/self/
	// instead of /proc/<tid>/ like libapparmor does
	f, err := os.OpenFile("/proc/self/attr/"+attr, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := utils.EnsureProcHandle(f); err != nil {
		return err
	}

	_, err = f.WriteString(value)
	return err
}

// changeOnExec reimplements aa_change_onexec from libapparmor in Go
func changeOnExec(name string) error {
	if err := setProcAttr("exec", "exec "+name); err != nil {
		return fmt.Errorf("apparmor failed to apply profile: %s", err)
	}
	return nil
}

// ApplyProfile will apply the profile with the specified name to the process after
// the next exec.
func ApplyProfile(name string) error {
	if name == "" {
		return nil
	}

	return changeOnExec(name)
}
