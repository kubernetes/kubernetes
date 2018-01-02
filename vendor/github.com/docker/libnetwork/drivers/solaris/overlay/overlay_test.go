// +build solaris

package overlay

import (
	"os/exec"
	"testing"
	"time"

	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/libkv/store/consul"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	_ "github.com/docker/libnetwork/testutils"
)

func init() {
	consul.Register()
}

type driverTester struct {
	t *testing.T
	d *driver
}

const testNetworkType = "overlay"

func setupDriver(t *testing.T) *driverTester {
	dt := &driverTester{t: t}
	config := make(map[string]interface{})
	config[netlabel.GlobalKVClient] = discoverapi.DatastoreConfigData{
		Scope:    datastore.GlobalScope,
		Provider: "consul",
		Address:  "127.0.0.01:8500",
	}

	if err := Init(dt, config); err != nil {
		t.Fatal(err)
	}

	// Use net0 as default interface
	ifcfgCmd := "/usr/sbin/ifconfig net0 | grep inet | awk '{print $2}'"
	out, err := exec.Command("/usr/bin/bash", "-c", ifcfgCmd).Output()
	if err != nil {
		t.Fatal(err)
	}

	data := discoverapi.NodeDiscoveryData{
		Address: string(out),
		Self:    true,
	}
	dt.d.DiscoverNew(discoverapi.NodeDiscovery, data)
	return dt
}

func cleanupDriver(t *testing.T, dt *driverTester) {
	ch := make(chan struct{})
	go func() {
		Fini(dt.d)
		close(ch)
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		t.Fatal("test timed out because Fini() did not return on time")
	}
}

func (dt *driverTester) GetPluginGetter() plugingetter.PluginGetter {
	return nil
}

func (dt *driverTester) RegisterDriver(name string, drv driverapi.Driver,
	cap driverapi.Capability) error {
	if name != testNetworkType {
		dt.t.Fatalf("Expected driver register name to be %q. Instead got %q",
			testNetworkType, name)
	}

	if _, ok := drv.(*driver); !ok {
		dt.t.Fatalf("Expected driver type to be %T. Instead got %T",
			&driver{}, drv)
	}

	dt.d = drv.(*driver)
	return nil
}

func TestOverlayInit(t *testing.T) {
	if err := Init(&driverTester{t: t}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestOverlayFiniWithoutConfig(t *testing.T) {
	dt := &driverTester{t: t}
	if err := Init(dt, nil); err != nil {
		t.Fatal(err)
	}

	cleanupDriver(t, dt)
}

func TestOverlayConfig(t *testing.T) {
	dt := setupDriver(t)

	time.Sleep(1 * time.Second)

	d := dt.d
	if d.notifyCh == nil {
		t.Fatal("Driver notify channel wasn't initialzed after Config method")
	}

	if d.exitCh == nil {
		t.Fatal("Driver serfloop exit channel wasn't initialzed after Config method")
	}

	if d.serfInstance == nil {
		t.Fatal("Driver serfinstance  hasn't been initialized after Config method")
	}

	cleanupDriver(t, dt)
}

func TestOverlayType(t *testing.T) {
	dt := &driverTester{t: t}
	if err := Init(dt, nil); err != nil {
		t.Fatal(err)
	}

	if dt.d.Type() != testNetworkType {
		t.Fatalf("Expected Type() to return %q. Instead got %q", testNetworkType,
			dt.d.Type())
	}
}
