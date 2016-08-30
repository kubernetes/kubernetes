package api

import (
	"fmt"
	"testing"

	"github.com/hashicorp/consul/testutil"
)

func TestHealth_Node(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	agent := c.Agent()
	health := c.Health()

	info, err := agent.Self()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	name := info["Config"]["NodeName"].(string)

	testutil.WaitForResult(func() (bool, error) {
		checks, meta, err := health.Node(name, nil)
		if err != nil {
			return false, err
		}
		if meta.LastIndex == 0 {
			return false, fmt.Errorf("bad: %v", meta)
		}
		if len(checks) == 0 {
			return false, fmt.Errorf("bad: %v", checks)
		}
		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestHealth_Checks(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	agent := c.Agent()
	health := c.Health()

	// Make a service with a check
	reg := &AgentServiceRegistration{
		Name: "foo",
		Check: &AgentServiceCheck{
			TTL: "15s",
		},
	}
	if err := agent.ServiceRegister(reg); err != nil {
		t.Fatalf("err: %v", err)
	}
	defer agent.ServiceDeregister("foo")

	testutil.WaitForResult(func() (bool, error) {
		checks, meta, err := health.Checks("foo", nil)
		if err != nil {
			return false, err
		}
		if meta.LastIndex == 0 {
			return false, fmt.Errorf("bad: %v", meta)
		}
		if len(checks) == 0 {
			return false, fmt.Errorf("Bad: %v", checks)
		}
		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestHealth_Service(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	health := c.Health()

	testutil.WaitForResult(func() (bool, error) {
		// consul service should always exist...
		checks, meta, err := health.Service("consul", "", true, nil)
		if err != nil {
			return false, err
		}
		if meta.LastIndex == 0 {
			return false, fmt.Errorf("bad: %v", meta)
		}
		if len(checks) == 0 {
			return false, fmt.Errorf("Bad: %v", checks)
		}
		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestHealth_State(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	health := c.Health()

	testutil.WaitForResult(func() (bool, error) {
		checks, meta, err := health.State("any", nil)
		if err != nil {
			return false, err
		}
		if meta.LastIndex == 0 {
			return false, fmt.Errorf("bad: %v", meta)
		}
		if len(checks) == 0 {
			return false, fmt.Errorf("Bad: %v", checks)
		}
		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}
