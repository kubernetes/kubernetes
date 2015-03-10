// +build linux

package mount

func MakeShared(mountPoint string) error {
	return ensureMountedAs(mountPoint, "shared")
}

func MakeRShared(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rshared")
}

func MakePrivate(mountPoint string) error {
	return ensureMountedAs(mountPoint, "private")
}

func MakeRPrivate(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rprivate")
}

func MakeSlave(mountPoint string) error {
	return ensureMountedAs(mountPoint, "slave")
}

func MakeRSlave(mountPoint string) error {
	return ensureMountedAs(mountPoint, "rslave")
}

func MakeUnbindable(mountPoint string) error {
	return ensureMountedAs(mountPoint, "unbindable")
}

func MakeRUnbindable(mountPoint string) error {
	return ensureMountedAs(mountPoint, "runbindable")
}

func ensureMountedAs(mountPoint, options string) error {
	mounted, err := Mounted(mountPoint)
	if err != nil {
		return err
	}

	if !mounted {
		if err := Mount(mountPoint, mountPoint, "none", "bind,rw"); err != nil {
			return err
		}
	}
	mounted, err = Mounted(mountPoint)
	if err != nil {
		return err
	}

	return ForceMount("", mountPoint, "none", options)
}
