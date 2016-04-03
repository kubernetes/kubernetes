package etcd

import (
	"testing"
)

func TestDelete(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
	}()

	c.Set("foo", "bar", 5)
	resp, err := c.Delete("foo", false)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Node.Value == "") {
		t.Fatalf("Delete failed with %s", resp.Node.Value)
	}

	if !(resp.PrevNode.Value == "bar") {
		t.Fatalf("Delete PrevNode failed with %s", resp.Node.Value)
	}

	resp, err = c.Delete("foo", false)
	if err == nil {
		t.Fatalf("Delete should have failed because the key foo did not exist.  "+
			"The response was: %v", resp)
	}
}

func TestDeleteAll(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
		c.Delete("fooDir", true)
	}()

	c.SetDir("foo", 5)
	// test delete an empty dir
	resp, err := c.DeleteDir("foo")
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Node.Value == "") {
		t.Fatalf("DeleteAll 1 failed: %#v", resp)
	}

	if !(resp.PrevNode.Dir == true && resp.PrevNode.Value == "") {
		t.Fatalf("DeleteAll 1 PrevNode failed: %#v", resp)
	}

	c.CreateDir("fooDir", 5)
	c.Set("fooDir/foo", "bar", 5)
	_, err = c.DeleteDir("fooDir")
	if err == nil {
		t.Fatal("should not able to delete a non-empty dir with deletedir")
	}

	resp, err = c.Delete("fooDir", true)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Node.Value == "") {
		t.Fatalf("DeleteAll 2 failed: %#v", resp)
	}

	if !(resp.PrevNode.Dir == true && resp.PrevNode.Value == "") {
		t.Fatalf("DeleteAll 2 PrevNode failed: %#v", resp)
	}

	resp, err = c.Delete("foo", true)
	if err == nil {
		t.Fatalf("DeleteAll should have failed because the key foo did not exist.  "+
			"The response was: %v", resp)
	}
}
