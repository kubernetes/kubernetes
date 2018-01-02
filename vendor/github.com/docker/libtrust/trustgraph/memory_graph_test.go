package trustgraph

import (
	"fmt"
	"testing"

	"github.com/docker/libtrust"
)

func createTestKeysAndGrants(count int) ([]*Grant, []libtrust.PrivateKey) {
	grants := make([]*Grant, count)
	keys := make([]libtrust.PrivateKey, count)
	for i := 0; i < count; i++ {
		pk, err := libtrust.GenerateECP256PrivateKey()
		if err != nil {
			panic(err)
		}
		grant := &Grant{
			Subject:    fmt.Sprintf("/user-%d", i+1),
			Permission: 0x0f,
			Grantee:    pk.KeyID(),
		}
		keys[i] = pk
		grants[i] = grant
	}
	return grants, keys
}

func testVerified(t *testing.T, g TrustGraph, k libtrust.PublicKey, keyName, target string, permission uint16) {
	if ok, err := g.Verify(k, target, permission); err != nil {
		t.Fatalf("Unexpected error during verification: %s", err)
	} else if !ok {
		t.Errorf("key failed verification\n\tKey: %s(%s)\n\tNamespace: %s", keyName, k.KeyID(), target)
	}
}

func testNotVerified(t *testing.T, g TrustGraph, k libtrust.PublicKey, keyName, target string, permission uint16) {
	if ok, err := g.Verify(k, target, permission); err != nil {
		t.Fatalf("Unexpected error during verification: %s", err)
	} else if ok {
		t.Errorf("key should have failed verification\n\tKey: %s(%s)\n\tNamespace: %s", keyName, k.KeyID(), target)
	}
}

func TestVerify(t *testing.T) {
	grants, keys := createTestKeysAndGrants(4)
	extraGrants := make([]*Grant, 3)
	extraGrants[0] = &Grant{
		Subject:    "/user-3",
		Permission: 0x0f,
		Grantee:    "/user-2",
	}
	extraGrants[1] = &Grant{
		Subject:    "/user-3/sub-project",
		Permission: 0x0f,
		Grantee:    "/user-4",
	}
	extraGrants[2] = &Grant{
		Subject:    "/user-4",
		Permission: 0x07,
		Grantee:    "/user-1",
	}
	grants = append(grants, extraGrants...)

	g := NewMemoryGraph(grants)

	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-1", 0x0f)
	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-1/some-project/sub-value", 0x0f)
	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-4", 0x07)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-2/", 0x0f)
	testVerified(t, g, keys[2].PublicKey(), "user-key-3", "/user-3/sub-value", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3/sub-value", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3/", 0x0f)
	testVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-3/sub-project", 0x0f)
	testVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-3/sub-project/app", 0x0f)
	testVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-4", 0x0f)

	testNotVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-2", 0x0f)
	testNotVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-3/sub-value", 0x0f)
	testNotVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-4", 0x0f)
	testNotVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-1/", 0x0f)
	testNotVerified(t, g, keys[2].PublicKey(), "user-key-3", "/user-2", 0x0f)
	testNotVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-4", 0x0f)
	testNotVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-3", 0x0f)
}

func TestCircularWalk(t *testing.T) {
	grants, keys := createTestKeysAndGrants(3)
	user1Grant := &Grant{
		Subject:    "/user-2",
		Permission: 0x0f,
		Grantee:    "/user-1",
	}
	user2Grant := &Grant{
		Subject:    "/user-1",
		Permission: 0x0f,
		Grantee:    "/user-2",
	}
	grants = append(grants, user1Grant, user2Grant)

	g := NewMemoryGraph(grants)

	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-1", 0x0f)
	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-2", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-2", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-1", 0x0f)
	testVerified(t, g, keys[2].PublicKey(), "user-key-3", "/user-3", 0x0f)

	testNotVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-3", 0x0f)
	testNotVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3", 0x0f)
}

func assertGrantSame(t *testing.T, actual, expected *Grant) {
	if actual != expected {
		t.Fatalf("Unexpected grant retrieved\n\tExpected: %v\n\tActual: %v", expected, actual)
	}
}

func TestGetGrants(t *testing.T) {
	grants, keys := createTestKeysAndGrants(5)
	extraGrants := make([]*Grant, 4)
	extraGrants[0] = &Grant{
		Subject:    "/user-3/friend-project",
		Permission: 0x0f,
		Grantee:    "/user-2/friends",
	}
	extraGrants[1] = &Grant{
		Subject:    "/user-3/sub-project",
		Permission: 0x0f,
		Grantee:    "/user-4",
	}
	extraGrants[2] = &Grant{
		Subject:    "/user-2/friends",
		Permission: 0x0f,
		Grantee:    "/user-5/fun-project",
	}
	extraGrants[3] = &Grant{
		Subject:    "/user-5/fun-project",
		Permission: 0x0f,
		Grantee:    "/user-1",
	}
	grants = append(grants, extraGrants...)

	g := NewMemoryGraph(grants)

	grantChains, err := g.GetGrants(keys[3], "/user-3/sub-project/specific-app", 0x0f)
	if err != nil {
		t.Fatalf("Error getting grants: %s", err)
	}
	if len(grantChains) != 1 {
		t.Fatalf("Expected number of grant chains returned, expected %d, received %d", 1, len(grantChains))
	}
	if len(grantChains[0]) != 2 {
		t.Fatalf("Unexpected number of grants retrieved\n\tExpected: %d\n\tActual: %d", 2, len(grantChains[0]))
	}
	assertGrantSame(t, grantChains[0][0], grants[3])
	assertGrantSame(t, grantChains[0][1], extraGrants[1])

	grantChains, err = g.GetGrants(keys[0], "/user-3/friend-project/fun-app", 0x0f)
	if err != nil {
		t.Fatalf("Error getting grants: %s", err)
	}
	if len(grantChains) != 1 {
		t.Fatalf("Expected number of grant chains returned, expected %d, received %d", 1, len(grantChains))
	}
	if len(grantChains[0]) != 4 {
		t.Fatalf("Unexpected number of grants retrieved\n\tExpected: %d\n\tActual: %d", 2, len(grantChains[0]))
	}
	assertGrantSame(t, grantChains[0][0], grants[0])
	assertGrantSame(t, grantChains[0][1], extraGrants[3])
	assertGrantSame(t, grantChains[0][2], extraGrants[2])
	assertGrantSame(t, grantChains[0][3], extraGrants[0])
}
