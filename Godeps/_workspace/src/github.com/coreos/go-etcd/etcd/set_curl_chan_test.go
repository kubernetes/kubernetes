package etcd

import (
	"fmt"
	"testing"
)

func TestSetCurlChan(t *testing.T) {
	c := NewClient(nil)
	c.OpenCURL()

	defer func() {
		c.Delete("foo", true)
	}()

	_, err := c.Set("foo", "bar", 5)
	if err != nil {
		t.Fatal(err)
	}

	expected := fmt.Sprintf("curl -X PUT %s/v2/keys/foo -d value=bar -d ttl=5",
		c.cluster.pick())
	actual := c.RecvCURL()
	if expected != actual {
		t.Fatalf(`Command "%s" is not equal to expected value "%s"`,
			actual, expected)
	}

	c.SetConsistency(STRONG_CONSISTENCY)
	_, err = c.Get("foo", false, false)
	if err != nil {
		t.Fatal(err)
	}

	expected = fmt.Sprintf("curl -X GET %s/v2/keys/foo?quorum=true&recursive=false&sorted=false",
		c.cluster.pick())
	actual = c.RecvCURL()
	if expected != actual {
		t.Fatalf(`Command "%s" is not equal to expected value "%s"`,
			actual, expected)
	}
}
