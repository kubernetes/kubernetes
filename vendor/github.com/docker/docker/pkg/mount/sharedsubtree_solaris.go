// +build solaris

package mount

// MakeShared ensures a mounted filesystem has the SHARED mount option enabled.
// See the supported options in flags.go for further reference.
func MakeShared(mountPoint string) error {
	return ensureMountedAs(mountPoint, "shared")
}

// MakeRShared ensures a mounted filesystem has the RSHARED mount option enabled.
// See the supported options in flags.go for further reference.
func MakeRShared(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rshared")
}

// MakePrivate ensures a mounted filesystem has the PRIVATE mount option enabled.
// See the supported options in flags.go for further reference.
func MakePrivate(mountPoint string) error {
	return ensureMountedAs(mountPoint, "private")
}

// MakeRPrivate ensures a mounted filesystem has the RPRIVATE mount option
// enabled. See the supported options in flags.go for further reference.
func MakeRPrivate(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rprivate")
}

// MakeSlave ensures a mounted filesystem has the SLAVE mount option enabled.
// See the supported options in flags.go for further reference.
func MakeSlave(mountPoint string) error {
	return ensureMountedAs(mountPoint, "slave")
}

// MakeRSlave ensures a mounted filesystem has the RSLAVE mount option enabled.
// See the supported options in flags.go for further reference.
func MakeRSlave(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rslave")
}

// MakeUnbindable ensures a mounted filesystem has the UNBINDABLE mount option
// enabled. See the supported options in flags.go for further reference.
func MakeUnbindable(mountPoint string) error {
	return ensureMountedAs(mountPoint, "unbindable")
}

// MakeRUnbindable ensures a mounted filesystem has the RUNBINDABLE mount
// option enabled. See the supported options in flags.go for further reference.
func MakeRUnbindable(mountPoint string) error {
	return ensureMountedAs(mountPoint, "runbindable")
}

func ensureMountedAs(mountPoint, options string) error {
	// TODO: Solaris does not support bind mounts.
	// Evaluate lofs and also look at the relevant
	// mount flags to be supported.
	return nil
}
