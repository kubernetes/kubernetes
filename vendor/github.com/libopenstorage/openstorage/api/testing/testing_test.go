package testing

import (
	"os"
	"testing"
	"time"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	"github.com/libopenstorage/openstorage/api/server"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers"
	"github.com/libopenstorage/openstorage/volume/drivers/nfs"
	"github.com/libopenstorage/openstorage/volume/drivers/test"
)

var (
	testPath = string("/tmp/openstorage_client_test")
)

func init() {
	dlog.SetLevel(dlog.LevelDebug)
}

func makeRequest(t *testing.T) {
	versions, err := volumeclient.GetSupportedDriverVersions(nfs.Name, "")
	if err != nil {
		t.Fatalf("Failed to obtain supported versions. Err: %v", err)
	}
	if len(versions) == 0 {
		t.Fatalf("Versions array is empty")
	}
	c, err := volumeclient.NewDriverClient("", nfs.Name, versions[0], "")
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	d := volumeclient.VolumeDriver(c)
	_, err = d.Inspect([]string{"foo"})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
}

func TestAll(t *testing.T) {
	// Test to check if a new status has been added appropriately
	if api.StatusKindMapLength() != int(api.Status_STATUS_MAX)+1 {
		t.Fatalf("Number of defined openstorage statuses do not add up" +
			"with those defined in StatusKind map. Did you add a" +
			" new Status without adding it in StatusKind map?")
	}

	err := os.MkdirAll(testPath, 0744)
	if err != nil {
		t.Fatalf("Failed to create test path: %v", err)
	}

	err = volumedrivers.Register(nfs.Name, map[string]string{"path": testPath})
	if err != nil {
		t.Fatalf("Failed to initialize Driver: %v", err)
	}

	server.StartPluginAPI(
		nfs.Name,
		volume.DriverAPIBase,
		volume.PluginAPIBase,
		0,
		0,
	)
	time.Sleep(time.Second * 2)
	versions, err := volumeclient.GetSupportedDriverVersions(nfs.Name, "")
	if err != nil {
		t.Fatalf("Failed to obtain supported versions. Err: %v", err)
	}
	if len(versions) == 0 {
		t.Fatalf("Versions array is empty")
	}
	c, err := volumeclient.NewDriverClient("", nfs.Name, versions[0], "")
	if err != nil {
		t.Fatalf("Failed to initialize Driver: %v", err)
	}
	d := volumeclient.VolumeDriver(c)
	ctx := test.NewContext(d)
	ctx.Filesystem = api.FSType_FS_TYPE_BTRFS
	test.Run(t, ctx)
}

func TestConnections(t *testing.T) {
	for i := 0; i < 2000; i++ {
		makeRequest(t)
	}
}
