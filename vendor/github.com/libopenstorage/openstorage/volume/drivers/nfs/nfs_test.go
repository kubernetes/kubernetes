package nfs

import (
	"os"
	"testing"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume/drivers/test"
)

var (
	testPath = string("/tmp/openstorage_driver_test")
)

func TestAll(t *testing.T) {
	err := os.MkdirAll(testPath, 0744)
	if err != nil {
		t.Fatalf("Failed to create test path: %v", err)
	}

	d, err := Init(map[string]string{"path": testPath})
	if err != nil {
		t.Fatalf("Failed to initialize Volume Driver: %v", err)
	}
	ctx := test.NewContext(d)
	ctx.Filesystem = api.FSType_FS_TYPE_NFS

	test.RunShort(t, ctx)
}
