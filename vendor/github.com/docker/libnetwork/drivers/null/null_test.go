package null

import (
	"testing"

	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func TestDriver(t *testing.T) {
	d := &driver{}

	if d.Type() != networkType {
		t.Fatalf("Unexpected network type returned by driver")
	}

	err := d.CreateNetwork("first", nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if d.network != "first" {
		t.Fatalf("Unexpected network id stored")
	}

	err = d.CreateNetwork("second", nil, nil, nil, nil)
	if err == nil {
		t.Fatalf("Second network creation should fail on this driver")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Second network creation failed with unexpected error type")
	}

	err = d.DeleteNetwork("first")
	if err == nil {
		t.Fatalf("network deletion should fail on this driver")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("network deletion failed with unexpected error type")
	}

	// we don't really check if it is there or not, delete is not allowed for this driver, period.
	err = d.DeleteNetwork("unknown")
	if err == nil {
		t.Fatalf("any network deletion should fail on this driver")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("any network deletion failed with unexpected error type")
	}
}
