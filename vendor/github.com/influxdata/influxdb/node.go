package influxdb

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
)

const (
	nodeFile      = "node.json"
	oldNodeFile   = "id"
	peersFilename = "peers.json"
)

type Node struct {
	path string
	ID   uint64
}

// LoadNode will load the node information from disk if present
func LoadNode(path string) (*Node, error) {
	// Always check to see if we are upgrading first
	if err := upgradeNodeFile(path); err != nil {
		return nil, err
	}

	n := &Node{
		path: path,
	}

	f, err := os.Open(filepath.Join(path, nodeFile))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(n); err != nil {
		return nil, err
	}

	return n, nil
}

// NewNode will return a new node
func NewNode(path string) *Node {
	return &Node{
		path: path,
	}
}

// Save will save the node file to disk and replace the existing one if present
func (n *Node) Save() error {
	file := filepath.Join(n.path, nodeFile)
	tmpFile := file + "tmp"

	f, err := os.Create(tmpFile)
	if err != nil {
		return err
	}

	if err = json.NewEncoder(f).Encode(n); err != nil {
		f.Close()
		return err
	}

	if err = f.Close(); nil != err {
		return err
	}

	return os.Rename(tmpFile, file)
}

func upgradeNodeFile(path string) error {
	oldFile := filepath.Join(path, oldNodeFile)
	b, err := ioutil.ReadFile(oldFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	// We shouldn't have an empty ID file, but if we do, ignore it
	if len(b) == 0 {
		return nil
	}

	peers := []string{}
	pb, err := ioutil.ReadFile(filepath.Join(path, peersFilename))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	err = json.Unmarshal(pb, &peers)
	if len(peers) > 1 {
		return fmt.Errorf("to upgrade a cluster, please contact support at influxdata")
	}

	n := &Node{
		path: path,
	}
	if n.ID, err = strconv.ParseUint(string(b), 10, 64); err != nil {
		return err
	}
	if err := n.Save(); err != nil {
		return err
	}
	if err := os.Remove(oldFile); err != nil {
		return err
	}
	return nil
}
