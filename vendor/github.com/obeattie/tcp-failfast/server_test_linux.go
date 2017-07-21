// +build linux

package tcpfailfast

import (
	"fmt"
	"os/exec"

	"github.com/songgao/water"
)

func setupTUN(iface *water.Interface) {
	err := exec.Command("ip", "addr", "add", "10.1.0.10/24", "dev", iface.Name()).Run()
	if err != nil {
		panic(fmt.Sprintf("error adding TUN IP: %v", err))
	}
	err = exec.Command("ip", "link", "set", "dev", iface.Name(), "up").Run()
	if err != nil {
		panic(fmt.Sprintf("error setting TUN interface up: %v", err))
	}
}
