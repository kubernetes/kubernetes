// +build linux

package daemon

import (
	"errors"
	"fmt"
	"strings"

	"github.com/docker/docker/runconfig"
	"github.com/opencontainers/runc/libcontainer/selinux"
)

func selinuxSetDisabled() {
	selinux.SetDisabled()
}

func selinuxFreeLxcContexts(label string) {
	selinux.FreeLxcContexts(label)
}

func selinuxEnabled() bool {
	return selinux.SelinuxEnabled()
}

func mergeLxcConfIntoOptions(hostConfig *runconfig.HostConfig) ([]string, error) {
	if hostConfig == nil {
		return nil, nil
	}

	out := []string{}

	// merge in the lxc conf options into the generic config map
	if lxcConf := hostConfig.LxcConf; lxcConf != nil {
		lxSlice := lxcConf.Slice()
		for _, pair := range lxSlice {
			// because lxc conf gets the driver name lxc.XXXX we need to trim it off
			// and let the lxc driver add it back later if needed
			if !strings.Contains(pair.Key, ".") {
				return nil, errors.New("Illegal Key passed into LXC Configurations")
			}
			parts := strings.SplitN(pair.Key, ".", 2)
			out = append(out, fmt.Sprintf("%s=%s", parts[1], pair.Value))
		}
	}

	return out, nil
}
