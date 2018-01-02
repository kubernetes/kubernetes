package containerd

import (
	"runtime"
	"testing"

	"github.com/containerd/containerd/errdefs"
)

func TestImageIsUnpacked(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip()
	}

	const imageName = "docker.io/library/busybox:latest"
	ctx, cancel := testContext()
	defer cancel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	// Cleanup
	err = client.ImageService().Delete(ctx, imageName)
	if err != nil && !errdefs.IsNotFound(err) {
		t.Fatal(err)
	}

	// By default pull does not unpack an image
	image, err := client.Pull(ctx, imageName)
	if err != nil {
		t.Fatal(err)
	}

	// Check that image is not unpacked
	unpacked, err := image.IsUnpacked(ctx, DefaultSnapshotter)
	if err != nil {
		t.Fatal(err)
	}
	if unpacked {
		t.Fatalf("image should not be unpacked")
	}

	// Check that image is unpacked
	err = image.Unpack(ctx, DefaultSnapshotter)
	if err != nil {
		t.Fatal(err)
	}
	unpacked, err = image.IsUnpacked(ctx, DefaultSnapshotter)
	if err != nil {
		t.Fatal(err)
	}
	if !unpacked {
		t.Fatalf("image should be unpacked")
	}
}
