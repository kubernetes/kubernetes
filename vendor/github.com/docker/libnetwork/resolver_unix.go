// +build !windows

package libnetwork

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"runtime"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork/iptables"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netns"
)

func init() {
	reexec.Register("setup-resolver", reexecSetupResolver)
}

const (
	// outputChain used for docker embed dns
	outputChain = "DOCKER_OUTPUT"
	//postroutingchain used for docker embed dns
	postroutingchain = "DOCKER_POSTROUTING"
)

func reexecSetupResolver() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if len(os.Args) < 4 {
		logrus.Error("invalid number of arguments..")
		os.Exit(1)
	}

	resolverIP, ipPort, _ := net.SplitHostPort(os.Args[2])
	_, tcpPort, _ := net.SplitHostPort(os.Args[3])
	rules := [][]string{
		{"-t", "nat", "-I", outputChain, "-d", resolverIP, "-p", "udp", "--dport", dnsPort, "-j", "DNAT", "--to-destination", os.Args[2]},
		{"-t", "nat", "-I", postroutingchain, "-s", resolverIP, "-p", "udp", "--sport", ipPort, "-j", "SNAT", "--to-source", ":" + dnsPort},
		{"-t", "nat", "-I", outputChain, "-d", resolverIP, "-p", "tcp", "--dport", dnsPort, "-j", "DNAT", "--to-destination", os.Args[3]},
		{"-t", "nat", "-I", postroutingchain, "-s", resolverIP, "-p", "tcp", "--sport", tcpPort, "-j", "SNAT", "--to-source", ":" + dnsPort},
	}

	f, err := os.OpenFile(os.Args[1], os.O_RDONLY, 0)
	if err != nil {
		logrus.Errorf("failed get network namespace %q: %v", os.Args[1], err)
		os.Exit(2)
	}
	defer f.Close()

	nsFD := f.Fd()
	if err = netns.Set(netns.NsHandle(nsFD)); err != nil {
		logrus.Errorf("setting into container net ns %v failed, %v", os.Args[1], err)
		os.Exit(3)
	}

	// insert outputChain and postroutingchain
	err = iptables.RawCombinedOutputNative("-t", "nat", "-C", "OUTPUT", "-d", resolverIP, "-j", outputChain)
	if err == nil {
		iptables.RawCombinedOutputNative("-t", "nat", "-F", outputChain)
	} else {
		iptables.RawCombinedOutputNative("-t", "nat", "-N", outputChain)
		iptables.RawCombinedOutputNative("-t", "nat", "-I", "OUTPUT", "-d", resolverIP, "-j", outputChain)
	}

	err = iptables.RawCombinedOutputNative("-t", "nat", "-C", "POSTROUTING", "-d", resolverIP, "-j", postroutingchain)
	if err == nil {
		iptables.RawCombinedOutputNative("-t", "nat", "-F", postroutingchain)
	} else {
		iptables.RawCombinedOutputNative("-t", "nat", "-N", postroutingchain)
		iptables.RawCombinedOutputNative("-t", "nat", "-I", "POSTROUTING", "-d", resolverIP, "-j", postroutingchain)
	}

	for _, rule := range rules {
		if iptables.RawCombinedOutputNative(rule...) != nil {
			logrus.Errorf("setting up rule failed, %v", rule)
		}
	}
}

func (r *resolver) setupIPTable() error {
	if r.err != nil {
		return r.err
	}
	laddr := r.conn.LocalAddr().String()
	ltcpaddr := r.tcpListen.Addr().String()

	cmd := &exec.Cmd{
		Path:   reexec.Self(),
		Args:   append([]string{"setup-resolver"}, r.resolverKey, laddr, ltcpaddr),
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("reexec failed: %v", err)
	}
	return nil
}
