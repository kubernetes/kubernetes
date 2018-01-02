package libnetwork

import (
	"os"
	"testing"

	"github.com/docker/libkv/store"
	"github.com/docker/libnetwork/datastore"
)

func TestBoltdbBackend(t *testing.T) {
	defer os.Remove(datastore.DefaultScopes("")[datastore.LocalScope].Client.Address)
	testLocalBackend(t, "", "", nil)
	defer os.Remove("/tmp/boltdb.db")
	config := &store.Config{Bucket: "testBackend"}
	testLocalBackend(t, "boltdb", "/tmp/boltdb.db", config)

}

func TestNoPersist(t *testing.T) {
	cfgOptions, err := OptionBoltdbWithRandomDBFile()
	if err != nil {
		t.Fatalf("Error creating random boltdb file : %v", err)
	}
	ctrl, err := New(cfgOptions...)
	if err != nil {
		t.Fatalf("Error new controller: %v", err)
	}
	nw, err := ctrl.NewNetwork("host", "host", "", NetworkOptionPersist(false))
	if err != nil {
		t.Fatalf("Error creating default \"host\" network: %v", err)
	}
	ep, err := nw.CreateEndpoint("newendpoint", []EndpointOption{}...)
	if err != nil {
		t.Fatalf("Error creating endpoint: %v", err)
	}
	store := ctrl.(*controller).getStore(datastore.LocalScope).KVStore()
	if exists, _ := store.Exists(datastore.Key(datastore.NetworkKeyPrefix, string(nw.ID()))); exists {
		t.Fatalf("Network with persist=false should not be stored in KV Store")
	}
	if exists, _ := store.Exists(datastore.Key([]string{datastore.EndpointKeyPrefix, string(nw.ID()), string(ep.ID())}...)); exists {
		t.Fatalf("Endpoint in Network with persist=false should not be stored in KV Store")
	}
	store.Close()
}
