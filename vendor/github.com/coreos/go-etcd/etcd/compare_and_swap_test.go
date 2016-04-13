package etcd

import (
	"testing"
)

func TestCompareAndSwap(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
	}()

	c.Set("foo", "bar", 5)

	// This should succeed
	resp, err := c.CompareAndSwap("foo", "bar2", 5, "bar", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Value == "bar2" && resp.Node.Key == "/foo" && resp.Node.TTL == 5) {
		t.Fatalf("CompareAndSwap 1 failed: %#v", resp)
	}

	if !(resp.PrevNode.Value == "bar" && resp.PrevNode.Key == "/foo" && resp.PrevNode.TTL == 5) {
		t.Fatalf("CompareAndSwap 1 prevNode failed: %#v", resp)
	}

	// This should fail because it gives an incorrect prevValue
	resp, err = c.CompareAndSwap("foo", "bar3", 5, "xxx", 0)
	if err == nil {
		t.Fatalf("CompareAndSwap 2 should have failed.  The response is: %#v", resp)
	}

	resp, err = c.Set("foo", "bar", 5)
	if err != nil {
		t.Fatal(err)
	}

	// This should succeed
	resp, err = c.CompareAndSwap("foo", "bar2", 5, "", resp.Node.ModifiedIndex)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Value == "bar2" && resp.Node.Key == "/foo" && resp.Node.TTL == 5) {
		t.Fatalf("CompareAndSwap 3 failed: %#v", resp)
	}

	if !(resp.PrevNode.Value == "bar" && resp.PrevNode.Key == "/foo" && resp.PrevNode.TTL == 5) {
		t.Fatalf("CompareAndSwap 3 prevNode failed: %#v", resp)
	}

	// This should fail because it gives an incorrect prevIndex
	resp, err = c.CompareAndSwap("foo", "bar3", 5, "", 29817514)
	if err == nil {
		t.Fatalf("CompareAndSwap 4 should have failed.  The response is: %#v", resp)
	}
}
