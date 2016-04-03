// +build windows

package builder

func fixPermissions(source, destination string, uid, gid int, destExisted bool) error {
	// chown is not supported on Windows
	return nil
}
