package api

import (
	"testing"
)

func TestACL_CreateDestroy(t *testing.T) {
	t.Parallel()
	c, s := makeACLClient(t)
	defer s.Stop()

	acl := c.ACL()

	ae := ACLEntry{
		Name:  "API test",
		Type:  ACLClientType,
		Rules: `key "" { policy = "deny" }`,
	}

	id, wm, err := acl.Create(&ae, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if wm.RequestTime == 0 {
		t.Fatalf("bad: %v", wm)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	ae2, _, err := acl.Info(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if ae2.Name != ae.Name || ae2.Type != ae.Type || ae2.Rules != ae.Rules {
		t.Fatalf("Bad: %#v", ae2)
	}

	wm, err = acl.Destroy(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if wm.RequestTime == 0 {
		t.Fatalf("bad: %v", wm)
	}
}

func TestACL_CloneDestroy(t *testing.T) {
	t.Parallel()
	c, s := makeACLClient(t)
	defer s.Stop()

	acl := c.ACL()

	id, wm, err := acl.Clone(c.config.Token, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if wm.RequestTime == 0 {
		t.Fatalf("bad: %v", wm)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	wm, err = acl.Destroy(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if wm.RequestTime == 0 {
		t.Fatalf("bad: %v", wm)
	}
}

func TestACL_Info(t *testing.T) {
	t.Parallel()
	c, s := makeACLClient(t)
	defer s.Stop()

	acl := c.ACL()

	ae, qm, err := acl.Info(c.config.Token, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if qm.LastIndex == 0 {
		t.Fatalf("bad: %v", qm)
	}
	if !qm.KnownLeader {
		t.Fatalf("bad: %v", qm)
	}

	if ae == nil || ae.ID != c.config.Token || ae.Type != ACLManagementType {
		t.Fatalf("bad: %#v", ae)
	}
}

func TestACL_List(t *testing.T) {
	t.Parallel()
	c, s := makeACLClient(t)
	defer s.Stop()

	acl := c.ACL()

	acls, qm, err := acl.List(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(acls) < 2 {
		t.Fatalf("bad: %v", acls)
	}

	if qm.LastIndex == 0 {
		t.Fatalf("bad: %v", qm)
	}
	if !qm.KnownLeader {
		t.Fatalf("bad: %v", qm)
	}
}
