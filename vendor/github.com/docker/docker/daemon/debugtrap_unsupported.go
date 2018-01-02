// +build !linux,!darwin,!freebsd,!windows,!solaris

package daemon

func (d *Daemon) setupDumpStackTrap(_ string) {
	return
}
