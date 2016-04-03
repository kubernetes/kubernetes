package etcd

import (
	"testing"
)

func TestCompareAndDelete(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
	}()

	c.Set("foo", "bar", 5)

	// This should succeed an correct prevValue
	resp, err := c.CompareAndDelete("foo", "bar", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.PrevNode.Value == "bar" && resp.PrevNode.Key == "/foo" && resp.PrevNode.TTL == 5) {
		t.Fatalf("CompareAndDelete 1 prevNode failed: %#v", resp)
	}

	resp, _ = c.Set("foo", "bar", 5)
	// This should fail because it gives an incorrect prevValue
	_, err = c.CompareAndDelete("foo", "xxx", 0)
	if err == nil {
		t.Fatalf("CompareAndDelete 2 should have failed.  The response is: %#v", resp)
	}

	// This should succeed because it gives an correct prevIndex
	resp, err = c.CompareAndDelete("foo", "", resp.Node.ModifiedIndex)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.PrevNode.Value == "bar" && resp.PrevNode.Key == "/foo" && resp.PrevNode.TTL == 5) {
		t.Fatalf("CompareAndSwap 3 prevNode failed: %#v", resp)
	}

	c.Set("foo", "bar", 5)
	// This should fail because it gives an incorrect prevIndex
	resp, err = c.CompareAndDelete("foo", "", 29817514)
	if err == nil {
		t.Fatalf("CompareAndDelete 4 should have failed.  The response is: %#v", resp)
	}
}
