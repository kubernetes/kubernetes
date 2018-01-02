package containerd

import (
	"runtime"
	"testing"
)

// TestExportAndImport exports testImage as a tar stream,
// and import the tar stream as a new image.
func TestExportAndImport(t *testing.T) {
	// TODO: support windows
	if testing.Short() || runtime.GOOS == "windows" {
		t.Skip()
	}
	ctx, cancel := testContext()
	defer cancel()

	client, err := New(address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	pulled, err := client.Pull(ctx, testImage)
	if err != nil {
		t.Fatal(err)
	}

	exported, err := client.Export(ctx, pulled.Target())
	if err != nil {
		t.Fatal(err)
	}

	importRef := "test/export-and-import:tmp"
	_, err = client.Import(ctx, importRef, exported, WithRefObject("@"+pulled.Target().Digest.String()))
	if err != nil {
		t.Fatal(err)
	}

	err = client.ImageService().Delete(ctx, importRef)
	if err != nil {
		t.Fatal(err)
	}
}
