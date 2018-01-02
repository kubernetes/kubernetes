// +build solaris

package bridge

import (
	"bytes"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"

	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

var (
	defaultBindingIP = net.IPv4(0, 0, 0, 0)
)

const (
	maxAllocatePortAttempts = 10
)

func addPFRules(epid, bindIntf string, bs []types.PortBinding) {
	var id string

	if len(epid) > 12 {
		id = epid[:12]
	} else {
		id = epid
	}

	fname := "/var/lib/docker/network/files/pf." + id

	f, err := os.OpenFile(fname,
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		logrus.Warn("cannot open temp pf file")
		return
	}
	for _, b := range bs {
		r := fmt.Sprintf(
			"pass in on %s proto %s from any to (%s) "+
				"port %d rdr-to %s port %d\n", bindIntf,
			b.Proto.String(), bindIntf, b.HostPort,
			b.IP.String(), b.Port)
		_, err = f.WriteString(r)
		if err != nil {
			logrus.Warnf("cannot write firewall rules to %s: %v", fname, err)
		}
	}
	f.Close()

	anchor := fmt.Sprintf("_auto/docker/ep%s", id)
	err = exec.Command("/usr/sbin/pfctl", "-a", anchor, "-f", fname).Run()
	if err != nil {
		logrus.Warnf("failed to add firewall rules: %v", err)
	}
	os.Remove(fname)
}

func removePFRules(epid string) {
	var id string

	if len(epid) > 12 {
		id = epid[:12]
	} else {
		id = epid
	}

	anchor := fmt.Sprintf("_auto/docker/ep%s", id)
	err := exec.Command("/usr/sbin/pfctl", "-a", anchor, "-F", "all").Run()
	if err != nil {
		logrus.Warnf("failed to remove firewall rules: %v", err)
	}
}

func (n *bridgeNetwork) allocatePorts(ep *bridgeEndpoint, bindIntf string, reqDefBindIP net.IP, ulPxyEnabled bool) ([]types.PortBinding, error) {
	if ep.extConnConfig == nil || ep.extConnConfig.PortBindings == nil {
		return nil, nil
	}

	defHostIP := defaultBindingIP
	if reqDefBindIP != nil {
		defHostIP = reqDefBindIP
	}

	bs, err := n.allocatePortsInternal(ep.extConnConfig.PortBindings, bindIntf, ep.addr.IP, defHostIP, ulPxyEnabled)
	if err != nil {
		return nil, err
	}

	// Add PF rules for port bindings, if any
	if len(bs) > 0 {
		addPFRules(ep.id, bindIntf, bs)
	}

	return bs, err
}

func (n *bridgeNetwork) allocatePortsInternal(bindings []types.PortBinding, bindIntf string, containerIP, defHostIP net.IP, ulPxyEnabled bool) ([]types.PortBinding, error) {
	bs := make([]types.PortBinding, 0, len(bindings))
	for _, c := range bindings {
		b := c.GetCopy()
		if err := n.allocatePort(&b, containerIP, defHostIP); err != nil {
			// On allocation failure,release previously
			// allocated ports. On cleanup error, just log
			// a warning message
			if cuErr := n.releasePortsInternal(bs); cuErr != nil {
				logrus.Warnf("Upon allocation failure "+
					"for %v, failed to clear previously "+
					"allocated port bindings: %v", b, cuErr)
			}
			return nil, err
		}
		bs = append(bs, b)
	}
	return bs, nil
}

func (n *bridgeNetwork) allocatePort(bnd *types.PortBinding, containerIP, defHostIP net.IP) error {
	var (
		host net.Addr
		err  error
	)

	// Store the container interface address in the operational binding
	bnd.IP = containerIP

	// Adjust the host address in the operational binding
	if len(bnd.HostIP) == 0 {
		bnd.HostIP = defHostIP
	}

	// Adjust HostPortEnd if this is not a range.
	if bnd.HostPortEnd == 0 {
		bnd.HostPortEnd = bnd.HostPort
	}

	// Construct the container side transport address
	container, err := bnd.ContainerAddr()
	if err != nil {
		return err
	}

	// Try up to maxAllocatePortAttempts times to get a port that's
	// not already allocated.
	for i := 0; i < maxAllocatePortAttempts; i++ {
		if host, err = n.portMapper.MapRange(container, bnd.HostIP,
			int(bnd.HostPort), int(bnd.HostPortEnd), false); err == nil {
			break
		}
		// There is no point in immediately retrying to map an
		// explicitly chosen port.
		if bnd.HostPort != 0 {
			logrus.Warnf(
				"Failed to allocate and map port %d-%d: %s",
				bnd.HostPort, bnd.HostPortEnd, err)
			break
		}
		logrus.Warnf("Failed to allocate and map port: %s, retry: %d",
			err, i+1)
	}
	if err != nil {
		return err
	}

	// Save the host port (regardless it was or not specified in the
	// binding)
	switch netAddr := host.(type) {
	case *net.TCPAddr:
		bnd.HostPort = uint16(host.(*net.TCPAddr).Port)
		return nil
	case *net.UDPAddr:
		bnd.HostPort = uint16(host.(*net.UDPAddr).Port)
		return nil
	default:
		// For completeness
		return ErrUnsupportedAddressType(fmt.Sprintf("%T", netAddr))
	}
}

func (n *bridgeNetwork) releasePorts(ep *bridgeEndpoint) error {
	err := n.releasePortsInternal(ep.portMapping)
	if err != nil {
		return nil
	}

	// remove rules if there are any port mappings
	if len(ep.portMapping) > 0 {
		removePFRules(ep.id)
	}

	return nil

}

func (n *bridgeNetwork) releasePortsInternal(bindings []types.PortBinding) error {
	var errorBuf bytes.Buffer

	// Attempt to release all port bindings, do not stop on failure
	for _, m := range bindings {
		if err := n.releasePort(m); err != nil {
			errorBuf.WriteString(
				fmt.Sprintf(
					"\ncould not release %v because of %v",
					m, err))
		}
	}

	if errorBuf.Len() != 0 {
		return errors.New(errorBuf.String())
	}
	return nil
}

func (n *bridgeNetwork) releasePort(bnd types.PortBinding) error {
	// Construct the host side transport address
	host, err := bnd.HostAddr()
	if err != nil {
		return err
	}
	return n.portMapper.Unmap(host)
}
