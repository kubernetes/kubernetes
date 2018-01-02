package bridge

import (
	"os"
	"testing"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func TestMain(m *testing.M) {
	if reexec.Init() {
		return
	}
	os.Exit(m.Run())
}

func TestPortMappingConfig(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()
	d := newDriver()

	config := &configuration{
		EnableIPTables: true,
	}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	binding1 := types.PortBinding{Proto: types.UDP, Port: uint16(400), HostPort: uint16(54000)}
	binding2 := types.PortBinding{Proto: types.TCP, Port: uint16(500), HostPort: uint16(65000)}
	portBindings := []types.PortBinding{binding1, binding2}

	sbOptions := make(map[string]interface{})
	sbOptions[netlabel.PortMap] = portBindings

	netConfig := &networkConfiguration{
		BridgeName: DefaultBridgeName,
	}
	netOptions := make(map[string]interface{})
	netOptions[netlabel.GenericData] = netConfig

	ipdList := getIPv4Data(t, "")
	err := d.CreateNetwork("dummy", netOptions, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	te := newTestEndpoint(ipdList[0].Pool, 11)
	err = d.CreateEndpoint("dummy", "ep1", te.Interface(), nil)
	if err != nil {
		t.Fatalf("Failed to create the endpoint: %s", err.Error())
	}

	if err = d.Join("dummy", "ep1", "sbox", te, sbOptions); err != nil {
		t.Fatalf("Failed to join the endpoint: %v", err)
	}

	if err = d.ProgramExternalConnectivity("dummy", "ep1", sbOptions); err != nil {
		t.Fatalf("Failed to program external connectivity: %v", err)
	}

	network, ok := d.networks["dummy"]
	if !ok {
		t.Fatalf("Cannot find network %s inside driver", "dummy")
	}
	ep, _ := network.endpoints["ep1"]
	if len(ep.portMapping) != 2 {
		t.Fatalf("Failed to store the port bindings into the sandbox info. Found: %v", ep.portMapping)
	}
	if ep.portMapping[0].Proto != binding1.Proto || ep.portMapping[0].Port != binding1.Port ||
		ep.portMapping[1].Proto != binding2.Proto || ep.portMapping[1].Port != binding2.Port {
		t.Fatal("bridgeEndpoint has incorrect port mapping values")
	}
	if ep.portMapping[0].HostIP == nil || ep.portMapping[0].HostPort == 0 ||
		ep.portMapping[1].HostIP == nil || ep.portMapping[1].HostPort == 0 {
		t.Fatal("operational port mapping data not found on bridgeEndpoint")
	}

	// release host mapped ports
	err = d.Leave("dummy", "ep1")
	if err != nil {
		t.Fatal(err)
	}

	err = d.RevokeExternalConnectivity("dummy", "ep1")
	if err != nil {
		t.Fatal(err)
	}
}
