package main

import (
	"fmt"
	"log"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
)

func main() {
	if reexec.Init() {
		return
	}

	// Select and configure the network driver
	networkType := "bridge"

	// Create a new controller instance
	driverOptions := options.Generic{}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = driverOptions
	controller, err := libnetwork.New(config.OptionDriverConfig(networkType, genericOption))
	if err != nil {
		log.Fatalf("libnetwork.New: %s", err)
	}

	// Create a network for containers to join.
	// NewNetwork accepts Variadic optional arguments that libnetwork and Drivers can use.
	network, err := controller.NewNetwork(networkType, "network1", "")
	if err != nil {
		log.Fatalf("controller.NewNetwork: %s", err)
	}

	// For each new container: allocate IP and interfaces. The returned network
	// settings will be used for container infos (inspect and such), as well as
	// iptables rules for port publishing. This info is contained or accessible
	// from the returned endpoint.
	ep, err := network.CreateEndpoint("Endpoint1")
	if err != nil {
		log.Fatalf("network.CreateEndpoint: %s", err)
	}

	// Create the sandbox for the container.
	// NewSandbox accepts Variadic optional arguments which libnetwork can use.
	sbx, err := controller.NewSandbox("container1",
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"))
	if err != nil {
		log.Fatalf("controller.NewSandbox: %s", err)
	}

	// A sandbox can join the endpoint via the join api.
	err = ep.Join(sbx)
	if err != nil {
		log.Fatalf("ep.Join: %s", err)
	}

	// libnetwork client can check the endpoint's operational data via the Info() API
	epInfo, err := ep.DriverInfo()
	if err != nil {
		log.Fatalf("ep.DriverInfo: %s", err)
	}

	macAddress, ok := epInfo[netlabel.MacAddress]
	if !ok {
		log.Fatal("failed to get mac address from endpoint info")
	}

	fmt.Printf("Joined endpoint %s (%s) to sandbox %s (%s)\n", ep.Name(), macAddress, sbx.ContainerID(), sbx.Key())
}
