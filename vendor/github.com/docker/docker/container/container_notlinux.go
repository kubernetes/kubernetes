// +build solaris freebsd

package container

import (
	"golang.org/x/sys/unix"
)

func detachMounted(path string) error {
	//Solaris and FreeBSD do not support the lazy unmount or MNT_DETACH feature.
	// Therefore there are separate definitions for this.
	return unix.Unmount(path, 0)
}

// SecretMounts returns the mounts for the secret path
func (container *Container) SecretMounts() []Mount {
	return nil
}

// UnmountSecrets unmounts the fs for secrets
func (container *Container) UnmountSecrets() error {
	return nil
}
