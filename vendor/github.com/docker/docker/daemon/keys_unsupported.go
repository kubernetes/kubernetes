// +build !linux

package daemon

// ModifyRootKeyLimit is a noop on unsupported platforms.
func ModifyRootKeyLimit() error {
	return nil
}
