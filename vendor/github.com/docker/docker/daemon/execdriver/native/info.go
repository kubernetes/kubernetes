// +build linux,cgo

package native

type info struct {
	ID     string
	driver *driver
}

// IsRunning is determined by looking for the
// pid file for a container.  If the file exists then the
// container is currently running
func (i *info) IsRunning() bool {
	_, ok := i.driver.activeContainers[i.ID]
	return ok
}
