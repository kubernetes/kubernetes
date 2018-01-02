package metadata

import (
	"testing"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

func TestLeases(t *testing.T) {
	ctx, db, cancel := testEnv(t)
	defer cancel()

	testCases := []struct {
		ID    string
		Cause error
	}{
		{
			ID: "tx1",
		},
		{
			ID:    "tx1",
			Cause: errdefs.ErrAlreadyExists,
		},
		{
			ID: "tx2",
		},
	}

	var leases []Lease

	for _, tc := range testCases {
		if err := db.Update(func(tx *bolt.Tx) error {
			lease, err := NewLeaseManager(tx).Create(ctx, tc.ID, nil)
			if err != nil {
				if tc.Cause != nil && errors.Cause(err) == tc.Cause {
					return nil
				}
				return err
			}
			leases = append(leases, lease)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	var listed []Lease
	// List leases, check same
	if err := db.View(func(tx *bolt.Tx) error {
		var err error
		listed, err = NewLeaseManager(tx).List(ctx, false)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if len(listed) != len(leases) {
		t.Fatalf("Expected %d lease, got %d", len(leases), len(listed))
	}
	for i := range listed {
		if listed[i].ID != leases[i].ID {
			t.Fatalf("Expected lease ID %s, got %s", leases[i].ID, listed[i].ID)
		}
		if listed[i].CreatedAt != leases[i].CreatedAt {
			t.Fatalf("Expected lease created at time %s, got %s", leases[i].CreatedAt, listed[i].CreatedAt)
		}
	}

	for _, tc := range testCases {
		if err := db.Update(func(tx *bolt.Tx) error {
			return NewLeaseManager(tx).Delete(ctx, tc.ID)
		}); err != nil {
			t.Fatal(err)
		}
	}

	if err := db.View(func(tx *bolt.Tx) error {
		var err error
		listed, err = NewLeaseManager(tx).List(ctx, false)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if len(listed) > 0 {
		t.Fatalf("Expected no leases, found %d: %v", len(listed), listed)
	}
}
