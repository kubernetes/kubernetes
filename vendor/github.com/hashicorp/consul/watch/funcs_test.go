package watch

import (
	"os"
	"testing"
	"time"

	consulapi "github.com/hashicorp/consul/api"
)

var consulAddr string

func init() {
	consulAddr = os.Getenv("CONSUL_ADDR")
}

func TestKeyWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"key", "key":"foo/bar/baz"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.(*consulapi.KVPair)
			if !ok || v == nil || string(v.Value) != "test" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		defer plan.Stop()
		time.Sleep(20 * time.Millisecond)

		kv := plan.client.KV()
		pair := &consulapi.KVPair{
			Key:   "foo/bar/baz",
			Value: []byte("test"),
		}
		_, err := kv.Put(pair, nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Wait for the query to run
		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		// Delete the key
		_, err = kv.Delete("foo/bar/baz", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestKeyPrefixWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"keyprefix", "prefix":"foo/"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.(consulapi.KVPairs)
			if ok && v == nil {
				return
			}
			if !ok || v == nil || string(v[0].Key) != "foo/bar" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		defer plan.Stop()
		time.Sleep(20 * time.Millisecond)

		kv := plan.client.KV()
		pair := &consulapi.KVPair{
			Key: "foo/bar",
		}
		_, err := kv.Put(pair, nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Wait for the query to run
		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		// Delete the key
		_, err = kv.Delete("foo/bar", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestServicesWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"services"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.(map[string][]string)
			if !ok || v["consul"] == nil {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		agent := plan.client.Agent()
		reg := &consulapi.AgentServiceRegistration{
			ID:   "foo",
			Name: "foo",
		}
		agent.ServiceRegister(reg)
		time.Sleep(20 * time.Millisecond)
		agent.ServiceDeregister("foo")
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestNodesWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"nodes"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.([]*consulapi.Node)
			if !ok || len(v) == 0 {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		catalog := plan.client.Catalog()
		reg := &consulapi.CatalogRegistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
		}
		catalog.Register(reg, nil)
		time.Sleep(20 * time.Millisecond)
		dereg := &consulapi.CatalogDeregistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
		}
		catalog.Deregister(dereg, nil)
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestServiceWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"service", "service":"foo", "tag":"bar", "passingonly":true}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.([]*consulapi.ServiceEntry)
			if ok && len(v) == 0 {
				return
			}
			if !ok || v[0].Service.ID != "foo" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		time.Sleep(20 * time.Millisecond)

		agent := plan.client.Agent()
		reg := &consulapi.AgentServiceRegistration{
			ID:   "foo",
			Name: "foo",
			Tags: []string{"bar"},
		}
		agent.ServiceRegister(reg)

		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		agent.ServiceDeregister("foo")
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestChecksWatch_State(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"checks", "state":"warning"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.([]*consulapi.HealthCheck)
			if len(v) == 0 {
				return
			}
			if !ok || v[0].CheckID != "foobar" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		time.Sleep(20 * time.Millisecond)

		catalog := plan.client.Catalog()
		reg := &consulapi.CatalogRegistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
			Check: &consulapi.AgentCheck{
				Node:    "foobar",
				CheckID: "foobar",
				Name:    "foobar",
				Status:  "warning",
			},
		}
		catalog.Register(reg, nil)

		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		dereg := &consulapi.CatalogDeregistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
		}
		catalog.Deregister(dereg, nil)
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestChecksWatch_Service(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"checks", "service":"foobar"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.([]*consulapi.HealthCheck)
			if len(v) == 0 {
				return
			}
			if !ok || v[0].CheckID != "foobar" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		time.Sleep(20 * time.Millisecond)

		catalog := plan.client.Catalog()
		reg := &consulapi.CatalogRegistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
			Service: &consulapi.AgentService{
				ID:      "foobar",
				Service: "foobar",
			},
			Check: &consulapi.AgentCheck{
				Node:      "foobar",
				CheckID:   "foobar",
				Name:      "foobar",
				Status:    "passing",
				ServiceID: "foobar",
			},
		}
		_, err := catalog.Register(reg, nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		time.Sleep(20 * time.Millisecond)
		plan.Stop()

		dereg := &consulapi.CatalogDeregistration{
			Node:       "foobar",
			Address:    "1.1.1.1",
			Datacenter: "dc1",
		}
		catalog.Deregister(dereg, nil)
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}

func TestEventWatch(t *testing.T) {
	if consulAddr == "" {
		t.Skip()
	}
	plan := mustParse(t, `{"type":"event", "name": "foo"}`)
	invoke := 0
	plan.Handler = func(idx uint64, raw interface{}) {
		if invoke == 0 {
			if raw == nil {
				return
			}
			v, ok := raw.([]*consulapi.UserEvent)
			if !ok || len(v) == 0 || string(v[len(v)-1].Name) != "foo" {
				t.Fatalf("Bad: %#v", raw)
			}
			invoke++
		}
	}

	go func() {
		defer plan.Stop()
		time.Sleep(20 * time.Millisecond)

		event := plan.client.Event()
		params := &consulapi.UserEvent{Name: "foo"}
		if _, _, err := event.Fire(params, nil); err != nil {
			t.Fatalf("err: %v", err)
		}
	}()

	err := plan.Run(consulAddr)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if invoke == 0 {
		t.Fatalf("bad: %v", invoke)
	}
}
