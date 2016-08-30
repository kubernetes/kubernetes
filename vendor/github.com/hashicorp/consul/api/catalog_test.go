package api

import (
	"fmt"
	"testing"

	"github.com/hashicorp/consul/testutil"
)

func TestCatalog_Datacenters(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	testutil.WaitForResult(func() (bool, error) {
		datacenters, err := catalog.Datacenters()
		if err != nil {
			return false, err
		}

		if len(datacenters) == 0 {
			return false, fmt.Errorf("Bad: %v", datacenters)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_Nodes(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	testutil.WaitForResult(func() (bool, error) {
		nodes, meta, err := catalog.Nodes(nil)
		if err != nil {
			return false, err
		}

		if meta.LastIndex == 0 {
			return false, fmt.Errorf("Bad: %v", meta)
		}

		if len(nodes) == 0 {
			return false, fmt.Errorf("Bad: %v", nodes)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_Services(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	testutil.WaitForResult(func() (bool, error) {
		services, meta, err := catalog.Services(nil)
		if err != nil {
			return false, err
		}

		if meta.LastIndex == 0 {
			return false, fmt.Errorf("Bad: %v", meta)
		}

		if len(services) == 0 {
			return false, fmt.Errorf("Bad: %v", services)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_Service(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	testutil.WaitForResult(func() (bool, error) {
		services, meta, err := catalog.Service("consul", "", nil)
		if err != nil {
			return false, err
		}

		if meta.LastIndex == 0 {
			return false, fmt.Errorf("Bad: %v", meta)
		}

		if len(services) == 0 {
			return false, fmt.Errorf("Bad: %v", services)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_Node(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()
	name, _ := c.Agent().NodeName()

	testutil.WaitForResult(func() (bool, error) {
		info, meta, err := catalog.Node(name, nil)
		if err != nil {
			return false, err
		}

		if meta.LastIndex == 0 {
			return false, fmt.Errorf("Bad: %v", meta)
		}
		if len(info.Services) == 0 {
			return false, fmt.Errorf("Bad: %v", info)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_Registration(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	service := &AgentService{
		ID:      "redis1",
		Service: "redis",
		Tags:    []string{"master", "v1"},
		Port:    8000,
	}

	check := &AgentCheck{
		Node:      "foobar",
		CheckID:   "service:redis1",
		Name:      "Redis health check",
		Notes:     "Script based health check",
		Status:    "passing",
		ServiceID: "redis1",
	}

	reg := &CatalogRegistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
		Service:    service,
		Check:      check,
	}

	testutil.WaitForResult(func() (bool, error) {
		if _, err := catalog.Register(reg, nil); err != nil {
			return false, err
		}

		node, _, err := catalog.Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if _, ok := node.Services["redis1"]; !ok {
			return false, fmt.Errorf("missing service: redis1")
		}

		health, _, err := c.Health().Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if health[0].CheckID != "service:redis1" {
			return false, fmt.Errorf("missing checkid service:redis1")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})

	// Test catalog deregistration of the previously registered service
	dereg := &CatalogDeregistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
		ServiceID:  "redis1",
	}

	if _, err := catalog.Deregister(dereg, nil); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		node, _, err := catalog.Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if _, ok := node.Services["redis1"]; ok {
			return false, fmt.Errorf("ServiceID:redis1 is not deregistered")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})

	// Test deregistration of the previously registered check
	dereg = &CatalogDeregistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
		CheckID:    "service:redis1",
	}

	if _, err := catalog.Deregister(dereg, nil); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		health, _, err := c.Health().Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if len(health) != 0 {
			return false, fmt.Errorf("CheckID:service:redis1 is not deregistered")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})

	// Test node deregistration of the previously registered node
	dereg = &CatalogDeregistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
	}

	if _, err := catalog.Deregister(dereg, nil); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		node, _, err := catalog.Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if node != nil {
			return false, fmt.Errorf("node is not deregistered: %v", node)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCatalog_EnableTagOverride(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	catalog := c.Catalog()

	service := &AgentService{
		ID:      "redis1",
		Service: "redis",
		Tags:    []string{"master", "v1"},
		Port:    8000,
	}

	reg := &CatalogRegistration{
		Datacenter: "dc1",
		Node:       "foobar",
		Address:    "192.168.10.10",
		Service:    service,
	}

	testutil.WaitForResult(func() (bool, error) {
		if _, err := catalog.Register(reg, nil); err != nil {
			return false, err
		}

		node, _, err := catalog.Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if _, ok := node.Services["redis1"]; !ok {
			return false, fmt.Errorf("missing service: redis1")
		}
		if node.Services["redis1"].EnableTagOverride != false {
			return false, fmt.Errorf("tag override set")
		}

		services, _, err := catalog.Service("redis", "", nil)
		if err != nil {
			return false, err
		}

		if len(services) < 1 || services[0].ServiceName != "redis" {
			return false, fmt.Errorf("missing service: redis")
		}
		if services[0].ServiceEnableTagOverride != false {
			return false, fmt.Errorf("tag override set")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})

	service.EnableTagOverride = true
	testutil.WaitForResult(func() (bool, error) {
		if _, err := catalog.Register(reg, nil); err != nil {
			return false, err
		}

		node, _, err := catalog.Node("foobar", nil)
		if err != nil {
			return false, err
		}

		if _, ok := node.Services["redis1"]; !ok {
			return false, fmt.Errorf("missing service: redis1")
		}
		if node.Services["redis1"].EnableTagOverride != true {
			return false, fmt.Errorf("tag override not set")
		}

		services, _, err := catalog.Service("redis", "", nil)
		if err != nil {
			return false, err
		}

		if len(services) < 1 || services[0].ServiceName != "redis" {
			return false, fmt.Errorf("missing service: redis")
		}
		if services[0].ServiceEnableTagOverride != true {
			return false, fmt.Errorf("tag override not set")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}
