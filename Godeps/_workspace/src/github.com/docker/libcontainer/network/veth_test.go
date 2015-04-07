// +build linux

package network

import (
	"testing"

	"github.com/docker/libcontainer/netlink"
)

func TestGenerateVethNames(t *testing.T) {
	if testing.Short() {
		return
	}

	prefix := "veth"

	name1, name2, err := createVethPair(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}

	if name1 == "" {
		t.Fatal("name1 should not be empty")
	}

	if name2 == "" {
		t.Fatal("name2 should not be empty")
	}
}

func TestCreateDuplicateVethPair(t *testing.T) {
	if testing.Short() {
		return
	}

	prefix := "veth"

	name1, name2, err := createVethPair(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}

	// retry to create the name interfaces and make sure that we get the correct error
	err = CreateVethPair(name1, name2, 0)
	if err == nil {
		t.Fatal("expected error to not be nil with duplicate interface")
	}

	if err != netlink.ErrInterfaceExists {
		t.Fatalf("expected error to be ErrInterfaceExists but received %q", err)
	}
}
