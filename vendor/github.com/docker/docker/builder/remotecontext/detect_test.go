package remotecontext

import (
	"errors"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/docker/docker/builder"
)

const (
	dockerfileContents   = "FROM busybox"
	dockerignoreFilename = ".dockerignore"
	testfileContents     = "test"
)

const shouldStayFilename = "should_stay"

func extractFilenames(files []os.FileInfo) []string {
	filenames := make([]string, len(files), len(files))

	for i, file := range files {
		filenames[i] = file.Name()
	}

	return filenames
}

func checkDirectory(t *testing.T, dir string, expectedFiles []string) {
	files, err := ioutil.ReadDir(dir)

	if err != nil {
		t.Fatalf("Could not read directory: %s", err)
	}

	if len(files) != len(expectedFiles) {
		log.Fatalf("Directory should contain exactly %d file(s), got %d", len(expectedFiles), len(files))
	}

	filenames := extractFilenames(files)
	sort.Strings(filenames)
	sort.Strings(expectedFiles)

	for i, filename := range filenames {
		if filename != expectedFiles[i] {
			t.Fatalf("File %s should be in the directory, got: %s", expectedFiles[i], filename)
		}
	}
}

func executeProcess(t *testing.T, contextDir string) {
	modifiableCtx := &stubRemote{root: contextDir}

	err := removeDockerfile(modifiableCtx, builder.DefaultDockerfileName)

	if err != nil {
		t.Fatalf("Error when executing Process: %s", err)
	}
}

func TestProcessShouldRemoveDockerfileDockerignore(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-dockerignore-process-test")
	defer cleanup()

	createTestTempFile(t, contextDir, shouldStayFilename, testfileContents, 0777)
	createTestTempFile(t, contextDir, dockerignoreFilename, "Dockerfile\n.dockerignore", 0777)
	createTestTempFile(t, contextDir, builder.DefaultDockerfileName, dockerfileContents, 0777)

	executeProcess(t, contextDir)

	checkDirectory(t, contextDir, []string{shouldStayFilename})

}

func TestProcessNoDockerignore(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-dockerignore-process-test")
	defer cleanup()

	createTestTempFile(t, contextDir, shouldStayFilename, testfileContents, 0777)
	createTestTempFile(t, contextDir, builder.DefaultDockerfileName, dockerfileContents, 0777)

	executeProcess(t, contextDir)

	checkDirectory(t, contextDir, []string{shouldStayFilename, builder.DefaultDockerfileName})

}

func TestProcessShouldLeaveAllFiles(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-dockerignore-process-test")
	defer cleanup()

	createTestTempFile(t, contextDir, shouldStayFilename, testfileContents, 0777)
	createTestTempFile(t, contextDir, builder.DefaultDockerfileName, dockerfileContents, 0777)
	createTestTempFile(t, contextDir, dockerignoreFilename, "input1\ninput2", 0777)

	executeProcess(t, contextDir)

	checkDirectory(t, contextDir, []string{shouldStayFilename, builder.DefaultDockerfileName, dockerignoreFilename})

}

// TODO: remove after moving to a separate pkg
type stubRemote struct {
	root string
}

func (r *stubRemote) Hash(path string) (string, error) {
	return "", errors.New("not implemented")
}

func (r *stubRemote) Root() string {
	return r.root
}
func (r *stubRemote) Close() error {
	return errors.New("not implemented")
}
func (r *stubRemote) Remove(p string) error {
	return os.Remove(filepath.Join(r.root, p))
}
