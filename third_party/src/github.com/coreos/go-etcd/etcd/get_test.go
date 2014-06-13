package etcd

import (
	"reflect"
	"testing"
)

// cleanNode scrubs Expiration, ModifiedIndex and CreatedIndex of a node.
func cleanNode(n *Node) {
	n.Expiration = nil
	n.ModifiedIndex = 0
	n.CreatedIndex = 0
}

// cleanResult scrubs a result object two levels deep of Expiration,
// ModifiedIndex and CreatedIndex.
func cleanResult(result *Response) {
	//  TODO(philips): make this recursive.
	cleanNode(result.Node)
	for i, _ := range result.Node.Nodes {
		cleanNode(result.Node.Nodes[i])
		for j, _ := range result.Node.Nodes[i].Nodes {
			cleanNode(result.Node.Nodes[i].Nodes[j])
		}
	}
}

func TestGet(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("foo", true)
	}()

	c.Set("foo", "bar", 5)

	result, err := c.Get("foo", false, false)

	if err != nil {
		t.Fatal(err)
	}

	if result.Node.Key != "/foo" || result.Node.Value != "bar" {
		t.Fatalf("Get failed with %s %s %v", result.Node.Key, result.Node.Value, result.Node.TTL)
	}

	result, err = c.Get("goo", false, false)
	if err == nil {
		t.Fatalf("should not be able to get non-exist key")
	}
}

func TestGetAll(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("fooDir", true)
	}()

	c.CreateDir("fooDir", 5)
	c.Set("fooDir/k0", "v0", 5)
	c.Set("fooDir/k1", "v1", 5)

	// Return kv-pairs in sorted order
	result, err := c.Get("fooDir", true, false)

	if err != nil {
		t.Fatal(err)
	}

	expected := Nodes{
		&Node{
			Key:   "/fooDir/k0",
			Value: "v0",
			TTL:   5,
		},
		&Node{
			Key:   "/fooDir/k1",
			Value: "v1",
			TTL:   5,
		},
	}

	cleanResult(result)

	if !reflect.DeepEqual(result.Node.Nodes, expected) {
		t.Fatalf("(actual) %v != (expected) %v", result.Node.Nodes, expected)
	}

	// Test the `recursive` option
	c.CreateDir("fooDir/childDir", 5)
	c.Set("fooDir/childDir/k2", "v2", 5)

	// Return kv-pairs in sorted order
	result, err = c.Get("fooDir", true, true)

	cleanResult(result)

	if err != nil {
		t.Fatal(err)
	}

	expected = Nodes{
		&Node{
			Key: "/fooDir/childDir",
			Dir: true,
			Nodes: Nodes{
				&Node{
					Key:   "/fooDir/childDir/k2",
					Value: "v2",
					TTL:   5,
				},
			},
			TTL: 5,
		},
		&Node{
			Key:   "/fooDir/k0",
			Value: "v0",
			TTL:   5,
		},
		&Node{
			Key:   "/fooDir/k1",
			Value: "v1",
			TTL:   5,
		},
	}

	cleanResult(result)

	if !reflect.DeepEqual(result.Node.Nodes, expected) {
		t.Fatalf("(actual) %v != (expected) %v", result.Node.Nodes, expected)
	}
}
