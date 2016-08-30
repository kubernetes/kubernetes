package state

import (
	"testing"

	"github.com/hashicorp/go-memdb"
)

func TestStateStore_Schema(t *testing.T) {
	// First call the schema creation
	schema := stateStoreSchema()

	// Try to initialize a new memdb using the schema
	if _, err := memdb.NewMemDB(schema); err != nil {
		t.Fatalf("err: %s", err)
	}
}
