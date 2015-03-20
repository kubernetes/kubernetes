package apparmor

import (
	"fmt"
	"os"
	"os/exec"
	"path"
)

const (
	DefaultProfilePath = "/etc/apparmor.d/docker"
)

func InstallDefaultProfile() error {
	if !IsEnabled() {
		return nil
	}

	// Make sure /etc/apparmor.d exists
	if err := os.MkdirAll(path.Dir(DefaultProfilePath), 0755); err != nil {
		return err
	}

	f, err := os.OpenFile(DefaultProfilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if err := generateProfile(f); err != nil {
		f.Close()
		return err
	}
	f.Close()

	cmd := exec.Command("/sbin/apparmor_parser", "-r", "-W", "docker")
	// to use the parser directly we have to make sure we are in the correct
	// dir with the profile
	cmd.Dir = "/etc/apparmor.d"

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Error loading docker apparmor profile: %s (%s)", err, output)
	}
	return nil
}
