package api

import (
	"fmt"
	"testing"

	"github.com/hashicorp/consul/testutil"
)

func TestCoordinate_Datacenters(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	coordinate := c.Coordinate()

	testutil.WaitForResult(func() (bool, error) {
		datacenters, err := coordinate.Datacenters()
		if err != nil {
			return false, err
		}

		if len(datacenters) == 0 {
			return false, fmt.Errorf("Bad: %v", datacenters)
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestCoordinate_Nodes(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	coordinate := c.Coordinate()

	testutil.WaitForResult(func() (bool, error) {
		_, _, err := coordinate.Nodes(nil)
		if err != nil {
			return false, err
		}

		// There's not a good way to populate coordinates without
		// waiting for them to calculate and update, so the best
		// we can do is call the endpoint and make sure we don't
		// get an error.
		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}
