//go:build !windows

package internal

func HasPrivilegesForSymlink() bool {
	return true
}
