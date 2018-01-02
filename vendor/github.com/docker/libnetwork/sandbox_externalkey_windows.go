// +build windows

package libnetwork

import (
	"io"
	"net"

	"github.com/docker/libnetwork/types"
)

// processSetKeyReexec is a private function that must be called only on an reexec path
// It expects 3 args { [0] = "libnetwork-setkey", [1] = <container-id>, [2] = <controller-id> }
// It also expects configs.HookState as a json string in <stdin>
// Refer to https://github.com/opencontainers/runc/pull/160/ for more information
func processSetKeyReexec() {
}

// SetExternalKey provides a convenient way to set an External key to a sandbox
func SetExternalKey(controllerID string, containerID string, key string) error {
	return types.NotImplementedErrorf("SetExternalKey isn't supported on non linux systems")
}

func sendKey(c net.Conn, data setKeyData) error {
	return types.NotImplementedErrorf("sendKey isn't supported on non linux systems")
}

func processReturn(r io.Reader) error {
	return types.NotImplementedErrorf("processReturn isn't supported on non linux systems")
}

// no-op on non linux systems
func (c *controller) startExternalKeyListener() error {
	return nil
}

func (c *controller) acceptClientConnections(sock string, l net.Listener) {
}

func (c *controller) processExternalKey(conn net.Conn) error {
	return types.NotImplementedErrorf("processExternalKey isn't supported on non linux systems")
}

func (c *controller) stopExternalKeyListener() {
}
