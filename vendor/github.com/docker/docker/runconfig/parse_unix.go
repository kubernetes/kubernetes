// +build !windows

package runconfig

import (
	"fmt"
	"strings"
)

func parseNetMode(netMode string) (NetworkMode, error) {
	parts := strings.Split(netMode, ":")
	switch mode := parts[0]; mode {
	case "default", "bridge", "none", "host":
	case "container":
		if len(parts) < 2 || parts[1] == "" {
			return "", fmt.Errorf("invalid container format container:<name|id>")
		}
	default:
		return "", fmt.Errorf("invalid --net: %s", netMode)
	}
	return NetworkMode(netMode), nil
}

func validateNetMode(vals *validateNM) error {

	if (vals.netMode.IsHost() || vals.netMode.IsContainer()) && *vals.flHostname != "" {
		return ErrConflictNetworkHostname
	}

	if vals.netMode.IsHost() && vals.flLinks.Len() > 0 {
		return ErrConflictHostNetworkAndLinks
	}

	if vals.netMode.IsContainer() && vals.flLinks.Len() > 0 {
		return ErrConflictContainerNetworkAndLinks
	}

	if (vals.netMode.IsHost() || vals.netMode.IsContainer()) && vals.flDns.Len() > 0 {
		return ErrConflictNetworkAndDns
	}

	if (vals.netMode.IsContainer() || vals.netMode.IsHost()) && vals.flExtraHosts.Len() > 0 {
		return ErrConflictNetworkHosts
	}

	if (vals.netMode.IsContainer() || vals.netMode.IsHost()) && *vals.flMacAddress != "" {
		return ErrConflictContainerNetworkAndMac
	}

	if vals.netMode.IsContainer() && (vals.flPublish.Len() > 0 || *vals.flPublishAll == true) {
		return ErrConflictNetworkPublishPorts
	}

	if vals.netMode.IsContainer() && vals.flExpose.Len() > 0 {
		return ErrConflictNetworkExposePorts
	}
	return nil
}
