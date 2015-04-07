package etcd

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os"
	"testing"
)

// To pass this test, we need to create a cluster of 3 machines
// The server should be listening on localhost:4001, 4002, 4003
func TestSync(t *testing.T) {
	fmt.Println("Make sure there are three nodes at 0.0.0.0:4001-4003")

	// Explicit trailing slash to ensure this doesn't reproduce:
	// https://github.com/coreos/go-etcd/issues/82
	c := NewClient([]string{"http://127.0.0.1:4001/"})

	success := c.SyncCluster()
	if !success {
		t.Fatal("cannot sync machines")
	}

	for _, m := range c.GetCluster() {
		u, err := url.Parse(m)
		if err != nil {
			t.Fatal(err)
		}
		if u.Scheme != "http" {
			t.Fatal("scheme must be http")
		}

		host, _, err := net.SplitHostPort(u.Host)
		if err != nil {
			t.Fatal(err)
		}
		if host != "localhost" {
			t.Fatal("Host must be localhost")
		}
	}

	badMachines := []string{"abc", "edef"}

	success = c.SetCluster(badMachines)

	if success {
		t.Fatal("should not sync on bad machines")
	}

	goodMachines := []string{"127.0.0.1:4002"}

	success = c.SetCluster(goodMachines)

	if !success {
		t.Fatal("cannot sync machines")
	} else {
		fmt.Println(c.cluster.Machines)
	}

}

func TestPersistence(t *testing.T) {
	c := NewClient(nil)
	c.SyncCluster()

	fo, err := os.Create("config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := fo.Close(); err != nil {
			panic(err)
		}
	}()

	c.SetPersistence(fo)
	err = c.saveConfig()
	if err != nil {
		t.Fatal(err)
	}

	c2, err := NewClientFromFile("config.json")
	if err != nil {
		t.Fatal(err)
	}

	// Verify that the two clients have the same config
	b1, _ := json.Marshal(c)
	b2, _ := json.Marshal(c2)

	if string(b1) != string(b2) {
		t.Fatalf("The two configs should be equal!")
	}
}
