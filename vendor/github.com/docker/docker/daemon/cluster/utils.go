package cluster

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/ioutils"
)

func loadPersistentState(root string) (*nodeStartConfig, error) {
	dt, err := ioutil.ReadFile(filepath.Join(root, stateFile))
	if err != nil {
		return nil, err
	}
	// missing certificate means no actual state to restore from
	if _, err := os.Stat(filepath.Join(root, "certificates/swarm-node.crt")); err != nil {
		if os.IsNotExist(err) {
			clearPersistentState(root)
		}
		return nil, err
	}
	var st nodeStartConfig
	if err := json.Unmarshal(dt, &st); err != nil {
		return nil, err
	}
	return &st, nil
}

func savePersistentState(root string, config nodeStartConfig) error {
	dt, err := json.Marshal(config)
	if err != nil {
		return err
	}
	return ioutils.AtomicWriteFile(filepath.Join(root, stateFile), dt, 0600)
}

func clearPersistentState(root string) error {
	// todo: backup this data instead of removing?
	// rather than delete the entire swarm directory, delete the contents in order to preserve the inode
	// (for example, allowing it to be bind-mounted)
	files, err := ioutil.ReadDir(root)
	if err != nil {
		return err
	}

	for _, f := range files {
		if err := os.RemoveAll(filepath.Join(root, f.Name())); err != nil {
			return err
		}
	}

	return nil
}

func removingManagerCausesLossOfQuorum(reachable, unreachable int) bool {
	return reachable-2 <= unreachable
}

func isLastManager(reachable, unreachable int) bool {
	return reachable == 1 && unreachable == 0
}
