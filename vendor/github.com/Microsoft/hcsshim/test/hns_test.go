package hcsshimtest

import (
	"os"
	"testing"

	"github.com/microsoft/hcsshim"
)

const (
	NatTestNetworkName  string = "GoTestNat"
	NatTestEndpointName string = "GoTestNatEndpoint"
)

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func CreateTestNetwork() (*hcsshim.HNSNetwork, error) {
	network := &hcsshim.HNSNetwork{
		Type: "NAT",
		Name: NatTestNetworkName,
		Subnets: []hcsshim.Subnet{
			hcsshim.Subnet{
				AddressPrefix:  "192.168.100.0/24",
				GatewayAddress: "192.168.100.1",
			},
		},
	}

	return network.Create()
}

func TestEndpoint(t *testing.T) {

	network, err := CreateTestNetwork()
	if err != nil {
		t.Error(err)
	}

	Endpoint := &hcsshim.HNSEndpoint{
		Name: NatTestEndpointName,
	}

	Endpoint, err = network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	err = Endpoint.HostAttach(1)
	if err != nil {
		t.Error(err)
	}

	err = Endpoint.HostDetach()
	if err != nil {
		t.Error(err)
	}

	_, err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}

	_, err = network.Delete()
	if err != nil {
		t.Error(err)
	}
}

func TestEndpointGetAll(t *testing.T) {
	_, err := hcsshim.HNSListEndpointRequest()
	if err != nil {
		t.Error(err)
	}
}

func TestNetworkGetAll(t *testing.T) {
	_, err := hcsshim.HNSListNetworkRequest("GET", "", "")
	if err != nil {
		t.Error(err)
	}
}

func TestNetwork(t *testing.T) {
	network, err := CreateTestNetwork()
	if err != nil {
		t.Error(err)
	}
	_, err = network.Delete()
	if err != nil {
		t.Error(err)
	}
}
