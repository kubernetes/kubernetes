// +build awsinclude

package apis

import (
	"os/exec"
	"strings"
	"testing"
)

func TestCollidingFolders(t *testing.T) {
	m := map[string]struct{}{}
	folders, err := getFolderNames()
	if err != nil {
		t.Error(err)
	}

	for _, folder := range folders {
		lcName := strings.ToLower(folder)
		if _, ok := m[lcName]; ok {
			t.Errorf("folder %q collision detected", folder)
		}
		m[lcName] = struct{}{}
	}
}

func getFolderNames() ([]string, error) {
	cmd := exec.Command("git", "ls-tree", "-d", "--name-only", "HEAD")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	return strings.Split(string(output), "\n"), nil
}
