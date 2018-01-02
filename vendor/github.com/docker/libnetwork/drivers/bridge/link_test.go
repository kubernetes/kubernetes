package bridge

import (
	"testing"

	"github.com/docker/libnetwork/types"
)

func getPorts() []types.TransportPort {
	return []types.TransportPort{
		{Proto: types.TCP, Port: uint16(5000)},
		{Proto: types.UDP, Port: uint16(400)},
		{Proto: types.TCP, Port: uint16(600)},
	}
}

func TestLinkNew(t *testing.T) {
	ports := getPorts()

	link := newLink("172.0.17.3", "172.0.17.2", ports, "docker0")

	if link == nil {
		t.FailNow()
	}
	if link.parentIP != "172.0.17.3" {
		t.Fail()
	}
	if link.childIP != "172.0.17.2" {
		t.Fail()
	}
	for i, p := range link.ports {
		if p != ports[i] {
			t.Fail()
		}
	}
	if link.bridge != "docker0" {
		t.Fail()
	}
}
