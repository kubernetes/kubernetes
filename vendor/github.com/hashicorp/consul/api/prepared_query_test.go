package api

import (
	"reflect"
	"testing"

	"github.com/hashicorp/consul/testutil"
)

func TestPreparedQuery(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	// Set up a node and a service.
	reg := &CatalogRegistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
		Service: &AgentService{
			ID:      "redis1",
			Service: "redis",
			Tags:    []string{"master", "v1"},
			Port:    8000,
		},
	}

	catalog := c.Catalog()
	testutil.WaitForResult(func() (bool, error) {
		if _, err := catalog.Register(reg, nil); err != nil {
			return false, err
		}

		if _, _, err := catalog.Node("foobar", nil); err != nil {
			return false, err
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})

	// Create a simple prepared query.
	def := &PreparedQueryDefinition{
		Service: ServiceQuery{
			Service: "redis",
		},
	}

	query := c.PreparedQuery()
	var err error
	def.ID, _, err = query.Create(def, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Read it back.
	defs, _, err := query.Get(def.ID, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(defs) != 1 || !reflect.DeepEqual(defs[0], def) {
		t.Fatalf("bad: %v", defs)
	}

	// List them all.
	defs, _, err = query.List(nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(defs) != 1 || !reflect.DeepEqual(defs[0], def) {
		t.Fatalf("bad: %v", defs)
	}

	// Make an update.
	def.Name = "my-query"
	_, err = query.Update(def, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Read it back again to verify the update worked.
	defs, _, err = query.Get(def.ID, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(defs) != 1 || !reflect.DeepEqual(defs[0], def) {
		t.Fatalf("bad: %v", defs)
	}

	// Execute by ID.
	results, _, err := query.Execute(def.ID, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(results.Nodes) != 1 || results.Nodes[0].Node.Node != "foobar" {
		t.Fatalf("bad: %v", results)
	}

	// Execute by name.
	results, _, err = query.Execute("my-query", nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(results.Nodes) != 1 || results.Nodes[0].Node.Node != "foobar" {
		t.Fatalf("bad: %v", results)
	}

	// Delete it.
	_, err = query.Delete(def.ID, nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure there are no longer any queries.
	defs, _, err = query.List(nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(defs) != 0 {
		t.Fatalf("bad: %v", defs)
	}
}
