// +build darwin

package tcpfailfast

import (
	"fmt"
	"os/exec"

	"github.com/songgao/water"
)

func setupTUN(iface *water.Interface) {
	err := exec.Command("ifconfig", iface.Name(), "10.1.0.10", "10.1.0.20", "up").Run()
	if err != nil {
		panic(fmt.Sprintf("error setting TUN interface up: %v", err))
	}

	b, err := exec.Command("ifconfig", iface.Name()).CombinedOutput()
	if err != nil {
		panic(err)
	}
	fmt.Println(string(b))
}
