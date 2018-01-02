package libnetwork

import (
	"fmt"
	"strings"

	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	gwEPlen = 12
)

var procGwNetwork = make(chan (bool), 1)

/*
   libnetwork creates a bridge network "docker_gw_bridge" for providing
   default gateway for the containers if none of the container's endpoints
   have GW set by the driver. ICC is set to false for the GW_bridge network.

   If a driver can't provide external connectivity it can choose to not set
   the GW IP for the endpoint.

   endpoint on the GW_bridge network is managed dynamically by libnetwork.
   ie:
   - its created when an endpoint without GW joins the container
   - its deleted when an endpoint with GW joins the container
*/

func (sb *sandbox) setupDefaultGW() error {

	// check if the container already has a GW endpoint
	if ep := sb.getEndpointInGWNetwork(); ep != nil {
		return nil
	}

	c := sb.controller

	// Look for default gw network. In case of error (includes not found),
	// retry and create it if needed in a serialized execution.
	n, err := c.NetworkByName(libnGWNetwork)
	if err != nil {
		if n, err = c.defaultGwNetwork(); err != nil {
			return err
		}
	}

	createOptions := []EndpointOption{CreateOptionAnonymous()}

	eplen := gwEPlen
	if len(sb.containerID) < gwEPlen {
		eplen = len(sb.containerID)
	}

	sbLabels := sb.Labels()

	if sbLabels[netlabel.PortMap] != nil {
		createOptions = append(createOptions, CreateOptionPortMapping(sbLabels[netlabel.PortMap].([]types.PortBinding)))
	}

	if sbLabels[netlabel.ExposedPorts] != nil {
		createOptions = append(createOptions, CreateOptionExposedPorts(sbLabels[netlabel.ExposedPorts].([]types.TransportPort)))
	}

	epOption := getPlatformOption()
	if epOption != nil {
		createOptions = append(createOptions, epOption)
	}

	newEp, err := n.CreateEndpoint("gateway_"+sb.containerID[0:eplen], createOptions...)
	if err != nil {
		return fmt.Errorf("container %s: endpoint create on GW Network failed: %v", sb.containerID, err)
	}

	defer func() {
		if err != nil {
			if err2 := newEp.Delete(true); err2 != nil {
				logrus.Warnf("Failed to remove gw endpoint for container %s after failing to join the gateway network: %v",
					sb.containerID, err2)
			}
		}
	}()

	epLocal := newEp.(*endpoint)

	if err = epLocal.sbJoin(sb); err != nil {
		return fmt.Errorf("container %s: endpoint join on GW Network failed: %v", sb.containerID, err)
	}

	return nil
}

// If present, detach and remove the endpoint connecting the sandbox to the default gw network.
func (sb *sandbox) clearDefaultGW() error {
	var ep *endpoint

	if ep = sb.getEndpointInGWNetwork(); ep == nil {
		return nil
	}
	if err := ep.sbLeave(sb, false); err != nil {
		return fmt.Errorf("container %s: endpoint leaving GW Network failed: %v", sb.containerID, err)
	}
	if err := ep.Delete(false); err != nil {
		return fmt.Errorf("container %s: deleting endpoint on GW Network failed: %v", sb.containerID, err)
	}
	return nil
}

// Evaluate whether the sandbox requires a default gateway based
// on the endpoints to which it is connected. It does not account
// for the default gateway network endpoint.

func (sb *sandbox) needDefaultGW() bool {
	var needGW bool

	for _, ep := range sb.getConnectedEndpoints() {
		if ep.endpointInGWNetwork() {
			continue
		}
		if ep.getNetwork().Type() == "null" || ep.getNetwork().Type() == "host" {
			continue
		}
		if ep.getNetwork().Internal() {
			continue
		}
		// During stale sandbox cleanup, joinInfo may be nil
		if ep.joinInfo != nil && ep.joinInfo.disableGatewayService {
			continue
		}
		// TODO v6 needs to be handled.
		if len(ep.Gateway()) > 0 {
			return false
		}
		for _, r := range ep.StaticRoutes() {
			if r.Destination != nil && r.Destination.String() == "0.0.0.0/0" {
				return false
			}
		}
		needGW = true
	}

	return needGW
}

func (sb *sandbox) getEndpointInGWNetwork() *endpoint {
	for _, ep := range sb.getConnectedEndpoints() {
		if ep.getNetwork().name == libnGWNetwork && strings.HasPrefix(ep.Name(), "gateway_") {
			return ep
		}
	}
	return nil
}

func (ep *endpoint) endpointInGWNetwork() bool {
	if ep.getNetwork().name == libnGWNetwork && strings.HasPrefix(ep.Name(), "gateway_") {
		return true
	}
	return false
}

func (sb *sandbox) getEPwithoutGateway() *endpoint {
	for _, ep := range sb.getConnectedEndpoints() {
		if ep.getNetwork().Type() == "null" || ep.getNetwork().Type() == "host" {
			continue
		}
		if len(ep.Gateway()) == 0 {
			return ep
		}
	}
	return nil
}

// Looks for the default gw network and creates it if not there.
// Parallel executions are serialized.
func (c *controller) defaultGwNetwork() (Network, error) {
	procGwNetwork <- true
	defer func() { <-procGwNetwork }()

	n, err := c.NetworkByName(libnGWNetwork)
	if err != nil {
		if _, ok := err.(types.NotFoundError); ok {
			n, err = c.createGWNetwork()
		}
	}
	return n, err
}

// Returns the endpoint which is providing external connectivity to the sandbox
func (sb *sandbox) getGatewayEndpoint() *endpoint {
	for _, ep := range sb.getConnectedEndpoints() {
		if ep.getNetwork().Type() == "null" || ep.getNetwork().Type() == "host" {
			continue
		}
		if len(ep.Gateway()) != 0 {
			return ep
		}
	}
	return nil
}
