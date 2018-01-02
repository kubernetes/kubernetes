package nat

import (
	"fmt"
	"reflect"
	"testing"
)

func TestSortUniquePorts(t *testing.T) {
	ports := []Port{
		Port("6379/tcp"),
		Port("22/tcp"),
	}

	Sort(ports, func(ip, jp Port) bool {
		return ip.Int() < jp.Int() || (ip.Int() == jp.Int() && ip.Proto() == "tcp")
	})

	first := ports[0]
	if fmt.Sprint(first) != "22/tcp" {
		t.Log(fmt.Sprint(first))
		t.Fail()
	}
}

func TestSortSamePortWithDifferentProto(t *testing.T) {
	ports := []Port{
		Port("8888/tcp"),
		Port("8888/udp"),
		Port("6379/tcp"),
		Port("6379/udp"),
	}

	Sort(ports, func(ip, jp Port) bool {
		return ip.Int() < jp.Int() || (ip.Int() == jp.Int() && ip.Proto() == "tcp")
	})

	first := ports[0]
	if fmt.Sprint(first) != "6379/tcp" {
		t.Fail()
	}
}

func TestSortPortMap(t *testing.T) {
	ports := []Port{
		Port("22/tcp"),
		Port("22/udp"),
		Port("8000/tcp"),
		Port("6379/tcp"),
		Port("9999/tcp"),
	}

	portMap := PortMap{
		Port("22/tcp"): []PortBinding{
			{},
		},
		Port("8000/tcp"): []PortBinding{
			{},
		},
		Port("6379/tcp"): []PortBinding{
			{},
			{HostIP: "0.0.0.0", HostPort: "32749"},
		},
		Port("9999/tcp"): []PortBinding{
			{HostIP: "0.0.0.0", HostPort: "40000"},
		},
	}

	SortPortMap(ports, portMap)
	if !reflect.DeepEqual(ports, []Port{
		Port("9999/tcp"),
		Port("6379/tcp"),
		Port("8000/tcp"),
		Port("22/tcp"),
		Port("22/udp"),
	}) {
		t.Errorf("failed to prioritize port with explicit mappings, got %v", ports)
	}
	if pm := portMap[Port("6379/tcp")]; !reflect.DeepEqual(pm, []PortBinding{
		{HostIP: "0.0.0.0", HostPort: "32749"},
		{},
	}) {
		t.Errorf("failed to prioritize bindings with explicit mappings, got %v", pm)
	}
}
