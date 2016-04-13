package etcd

import (
	"testing"
)

func TestSet(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
	}()

	resp, err := c.Set("foo", "bar", 5)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Node.Key != "/foo" || resp.Node.Value != "bar" || resp.Node.TTL != 5 {
		t.Fatalf("Set 1 failed: %#v", resp)
	}
	if resp.PrevNode != nil {
		t.Fatalf("Set 1 PrevNode failed: %#v", resp)
	}

	resp, err = c.Set("foo", "bar2", 5)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/foo" && resp.Node.Value == "bar2" && resp.Node.TTL == 5) {
		t.Fatalf("Set 2 failed: %#v", resp)
	}
	if resp.PrevNode.Key != "/foo" || resp.PrevNode.Value != "bar" || resp.Node.TTL != 5 {
		t.Fatalf("Set 2 PrevNode failed: %#v", resp)
	}
}

func TestUpdate(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
		c.Delete("nonexistent", true)
	}()

	resp, err := c.Set("foo", "bar", 5)

	if err != nil {
		t.Fatal(err)
	}

	// This should succeed.
	resp, err = c.Update("foo", "wakawaka", 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "update" && resp.Node.Key == "/foo" && resp.Node.TTL == 5) {
		t.Fatalf("Update 1 failed: %#v", resp)
	}
	if !(resp.PrevNode.Key == "/foo" && resp.PrevNode.Value == "bar" && resp.Node.TTL == 5) {
		t.Fatalf("Update 1 prevValue failed: %#v", resp)
	}

	// This should fail because the key does not exist.
	resp, err = c.Update("nonexistent", "whatever", 5)
	if err == nil {
		t.Fatalf("The key %v did not exist, so the update should have failed."+
			"The response was: %#v", resp.Node.Key, resp)
	}
}

func TestCreate(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("newKey", true)
	}()

	newKey := "/newKey"
	newValue := "/newValue"

	// This should succeed
	resp, err := c.Create(newKey, newValue, 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "create" && resp.Node.Key == newKey &&
		resp.Node.Value == newValue && resp.Node.TTL == 5) {
		t.Fatalf("Create 1 failed: %#v", resp)
	}
	if resp.PrevNode != nil {
		t.Fatalf("Create 1 PrevNode failed: %#v", resp)
	}

	// This should fail, because the key is already there
	resp, err = c.Create(newKey, newValue, 5)
	if err == nil {
		t.Fatalf("The key %v did exist, so the creation should have failed."+
			"The response was: %#v", resp.Node.Key, resp)
	}
}

func TestCreateInOrder(t *testing.T) {
	c := NewClient(nil)
	dir := "/queue"
	defer func() {
		c.DeleteDir(dir)
	}()

	var firstKey, secondKey string

	resp, err := c.CreateInOrder(dir, "1", 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "create" && resp.Node.Value == "1" && resp.Node.TTL == 5) {
		t.Fatalf("Create 1 failed: %#v", resp)
	}

	firstKey = resp.Node.Key

	resp, err = c.CreateInOrder(dir, "2", 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "create" && resp.Node.Value == "2" && resp.Node.TTL == 5) {
		t.Fatalf("Create 2 failed: %#v", resp)
	}

	secondKey = resp.Node.Key

	if firstKey >= secondKey {
		t.Fatalf("Expected first key to be greater than second key, but %s is not greater than %s",
			firstKey, secondKey)
	}
}

func TestSetDir(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
		c.Delete("fooDir", true)
	}()

	resp, err := c.CreateDir("fooDir", 5)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/fooDir" && resp.Node.Value == "" && resp.Node.TTL == 5) {
		t.Fatalf("SetDir 1 failed: %#v", resp)
	}
	if resp.PrevNode != nil {
		t.Fatalf("SetDir 1 PrevNode failed: %#v", resp)
	}

	// This should fail because /fooDir already points to a directory
	resp, err = c.CreateDir("/fooDir", 5)
	if err == nil {
		t.Fatalf("fooDir already points to a directory, so SetDir should have failed."+
			"The response was: %#v", resp)
	}

	_, err = c.Set("foo", "bar", 5)
	if err != nil {
		t.Fatal(err)
	}

	// This should succeed
	// It should replace the key
	resp, err = c.SetDir("foo", 5)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/foo" && resp.Node.Value == "" && resp.Node.TTL == 5) {
		t.Fatalf("SetDir 2 failed: %#v", resp)
	}
	if !(resp.PrevNode.Key == "/foo" && resp.PrevNode.Value == "bar" && resp.PrevNode.TTL == 5) {
		t.Fatalf("SetDir 2 failed: %#v", resp)
	}
}

func TestUpdateDir(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("fooDir", true)
	}()

	resp, err := c.CreateDir("fooDir", 5)
	if err != nil {
		t.Fatal(err)
	}

	// This should succeed.
	resp, err = c.UpdateDir("fooDir", 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "update" && resp.Node.Key == "/fooDir" &&
		resp.Node.Value == "" && resp.Node.TTL == 5) {
		t.Fatalf("UpdateDir 1 failed: %#v", resp)
	}
	if !(resp.PrevNode.Key == "/fooDir" && resp.PrevNode.Dir == true && resp.PrevNode.TTL == 5) {
		t.Fatalf("UpdateDir 1 PrevNode failed: %#v", resp)
	}

	// This should fail because the key does not exist.
	resp, err = c.UpdateDir("nonexistentDir", 5)
	if err == nil {
		t.Fatalf("The key %v did not exist, so the update should have failed."+
			"The response was: %#v", resp.Node.Key, resp)
	}
}

func TestCreateDir(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("fooDir", true)
	}()

	// This should succeed
	resp, err := c.CreateDir("fooDir", 5)
	if err != nil {
		t.Fatal(err)
	}

	if !(resp.Action == "create" && resp.Node.Key == "/fooDir" &&
		resp.Node.Value == "" && resp.Node.TTL == 5) {
		t.Fatalf("CreateDir 1 failed: %#v", resp)
	}
	if resp.PrevNode != nil {
		t.Fatalf("CreateDir 1 PrevNode failed: %#v", resp)
	}

	// This should fail, because the key is already there
	resp, err = c.CreateDir("fooDir", 5)
	if err == nil {
		t.Fatalf("The key %v did exist, so the creation should have failed."+
			"The response was: %#v", resp.Node.Key, resp)
	}
}
