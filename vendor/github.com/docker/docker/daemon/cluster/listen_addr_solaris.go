package cluster

import (
	"bufio"
	"fmt"
	"net"
	"os/exec"
	"strings"
)

func (c *Cluster) resolveSystemAddr() (net.IP, error) {
	defRouteCmd := "/usr/sbin/ipadm show-addr -p -o addr " +
		"`/usr/sbin/route get default | /usr/bin/grep interface | " +
		"/usr/bin/awk '{print $2}'`"
	out, err := exec.Command("/usr/bin/bash", "-c", defRouteCmd).Output()
	if err != nil {
		return nil, fmt.Errorf("cannot get default route: %v", err)
	}

	defInterface := strings.SplitN(string(out), "/", 2)
	defInterfaceIP := net.ParseIP(defInterface[0])

	return defInterfaceIP, nil
}

func listSystemIPs() []net.IP {
	var systemAddrs []net.IP
	cmd := exec.Command("/usr/sbin/ipadm", "show-addr", "-p", "-o", "addr")
	cmdReader, err := cmd.StdoutPipe()
	if err != nil {
		return nil
	}

	if err := cmd.Start(); err != nil {
		return nil
	}

	scanner := bufio.NewScanner(cmdReader)
	go func() {
		for scanner.Scan() {
			text := scanner.Text()
			nameAddrPair := strings.SplitN(text, "/", 2)
			// Let go of loopback interfaces and docker interfaces
			systemAddrs = append(systemAddrs, net.ParseIP(nameAddrPair[0]))
		}
	}()

	if err := scanner.Err(); err != nil {
		fmt.Printf("scan underwent err: %+v\n", err)
	}

	if err := cmd.Wait(); err != nil {
		fmt.Printf("run command wait: %+v\n", err)
	}

	return systemAddrs
}
