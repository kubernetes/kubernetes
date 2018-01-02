package remotecontext

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/docker/docker/builder"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/reexec"
	"github.com/pkg/errors"
)

const (
	filename = "test"
	contents = "contents test"
)

func init() {
	reexec.Init()
}

func TestCloseRootDirectory(t *testing.T) {
	contextDir, err := ioutil.TempDir("", "builder-tarsum-test")
	defer os.RemoveAll(contextDir)
	if err != nil {
		t.Fatalf("Error with creating temporary directory: %s", err)
	}

	src := makeTestArchiveContext(t, contextDir)
	err = src.Close()

	if err != nil {
		t.Fatalf("Error while executing Close: %s", err)
	}

	_, err = os.Stat(src.Root())

	if !os.IsNotExist(err) {
		t.Fatal("Directory should not exist at this point")
	}
}

func TestHashFile(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-tarsum-test")
	defer cleanup()

	createTestTempFile(t, contextDir, filename, contents, 0755)

	tarSum := makeTestArchiveContext(t, contextDir)

	sum, err := tarSum.Hash(filename)

	if err != nil {
		t.Fatalf("Error when executing Stat: %s", err)
	}

	if len(sum) == 0 {
		t.Fatalf("Hash returned empty sum")
	}

	expected := "1149ab94af7be6cc1da1335e398f24ee1cf4926b720044d229969dfc248ae7ec"

	if actual := sum; expected != actual {
		t.Fatalf("invalid checksum. expected %s, got %s", expected, actual)
	}
}

func TestHashSubdir(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-tarsum-test")
	defer cleanup()

	contextSubdir := filepath.Join(contextDir, "builder-tarsum-test-subdir")
	err := os.Mkdir(contextSubdir, 0755)
	if err != nil {
		t.Fatalf("Failed to make directory: %s", contextSubdir)
	}

	testFilename := createTestTempFile(t, contextSubdir, filename, contents, 0755)

	tarSum := makeTestArchiveContext(t, contextDir)

	relativePath, err := filepath.Rel(contextDir, testFilename)

	if err != nil {
		t.Fatalf("Error when getting relative path: %s", err)
	}

	sum, err := tarSum.Hash(relativePath)

	if err != nil {
		t.Fatalf("Error when executing Stat: %s", err)
	}

	if len(sum) == 0 {
		t.Fatalf("Hash returned empty sum")
	}

	expected := "d7f8d6353dee4816f9134f4156bf6a9d470fdadfb5d89213721f7e86744a4e69"

	if actual := sum; expected != actual {
		t.Fatalf("invalid checksum. expected %s, got %s", expected, actual)
	}
}

func TestStatNotExisting(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-tarsum-test")
	defer cleanup()

	src := makeTestArchiveContext(t, contextDir)
	_, err := src.Hash("not-existing")
	if !os.IsNotExist(errors.Cause(err)) {
		t.Fatalf("This file should not exist: %s", err)
	}
}

func TestRemoveDirectory(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-tarsum-test")
	defer cleanup()

	contextSubdir := createTestTempSubdir(t, contextDir, "builder-tarsum-test-subdir")

	relativePath, err := filepath.Rel(contextDir, contextSubdir)

	if err != nil {
		t.Fatalf("Error when getting relative path: %s", err)
	}

	src := makeTestArchiveContext(t, contextDir)

	tarSum := src.(modifiableContext)

	err = tarSum.Remove(relativePath)
	if err != nil {
		t.Fatalf("Error when executing Remove: %s", err)
	}

	_, err = src.Hash(contextSubdir)

	if !os.IsNotExist(errors.Cause(err)) {
		t.Fatal("Directory should not exist at this point")
	}
}

func makeTestArchiveContext(t *testing.T, dir string) builder.Source {
	tarStream, err := archive.Tar(dir, archive.Uncompressed)
	if err != nil {
		t.Fatalf("error: %s", err)
	}
	defer tarStream.Close()
	tarSum, err := FromArchive(tarStream)
	if err != nil {
		t.Fatalf("Error when executing FromArchive: %s", err)
	}
	return tarSum
}
