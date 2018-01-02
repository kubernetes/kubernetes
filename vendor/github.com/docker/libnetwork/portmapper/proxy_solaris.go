package portmapper

import (
	"net"
	"os/exec"
	"strconv"
)

func newProxyCommand(proto string, hostIP net.IP, hostPort int, containerIP net.IP, containerPort int, proxyPath string) (userlandProxy, error) {
	path := proxyPath
	if proxyPath == "" {
		cmd, err := exec.LookPath(userlandProxyCommandName)
		if err != nil {
			return nil, err
		}
		path = cmd
	}

	args := []string{
		path,
		"-proto", proto,
		"-host-ip", hostIP.String(),
		"-host-port", strconv.Itoa(hostPort),
		"-container-ip", containerIP.String(),
		"-container-port", strconv.Itoa(containerPort),
	}

	return &proxyCommand{
		cmd: &exec.Cmd{
			Path: path,
			Args: args,
		},
	}, nil
}
