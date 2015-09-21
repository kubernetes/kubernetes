/*
Utility for testing cgroup operations.

Creates a mock of the cgroup filesystem for the duration of the test.
*/
package fs

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/docker/libcontainer/configs"
)

type cgroupTestUtil struct {
	// data to use in tests.
	CgroupData *data

	// Path to the mock cgroup directory.
	CgroupPath string

	// Temporary directory to store mock cgroup filesystem.
	tempDir string
	t       *testing.T
}

// Creates a new test util for the specified subsystem
func NewCgroupTestUtil(subsystem string, t *testing.T) *cgroupTestUtil {
	d := &data{
		c: &configs.Cgroup{},
	}
	tempDir, err := ioutil.TempDir("", "cgroup_test")
	if err != nil {
		t.Fatal(err)
	}
	d.root = tempDir
	testCgroupPath := filepath.Join(d.root, subsystem)
	if err != nil {
		t.Fatal(err)
	}

	// Ensure the full mock cgroup path exists.
	err = os.MkdirAll(testCgroupPath, 0755)
	if err != nil {
		t.Fatal(err)
	}
	return &cgroupTestUtil{CgroupData: d, CgroupPath: testCgroupPath, tempDir: tempDir, t: t}
}

func (c *cgroupTestUtil) cleanup() {
	os.RemoveAll(c.tempDir)
}

// Write the specified contents on the mock of the specified cgroup files.
func (c *cgroupTestUtil) writeFileContents(fileContents map[string]string) {
	for file, contents := range fileContents {
		err := writeFile(c.CgroupPath, file, contents)
		if err != nil {
			c.t.Fatal(err)
		}
	}
}
