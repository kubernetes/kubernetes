// +build !linux,!darwin,!freebsd,!windows

package daemon

func (d *Daemon) setupDumpStackTrap(_ string) {
	return
}
