package idm

import (
	"testing"

	_ "github.com/docker/libnetwork/testutils"
)

func TestNew(t *testing.T) {
	_, err := New(nil, "", 0, 1)
	if err == nil {
		t.Fatal("Expected failure, but succeeded")
	}

	_, err = New(nil, "myset", 1<<10, 0)
	if err == nil {
		t.Fatal("Expected failure, but succeeded")
	}

	i, err := New(nil, "myset", 0, 10)
	if err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}
	if i.handle == nil {
		t.Fatal("set is not initialized")
	}
	if i.start != 0 {
		t.Fatal("unexpected start")
	}
	if i.end != 10 {
		t.Fatal("unexpected end")
	}
}

func TestAllocate(t *testing.T) {
	i, err := New(nil, "myids", 50, 52)
	if err != nil {
		t.Fatal(err)
	}

	if err = i.GetSpecificID(49); err == nil {
		t.Fatal("Expected failure but succeeded")
	}

	if err = i.GetSpecificID(53); err == nil {
		t.Fatal("Expected failure but succeeded")
	}

	o, err := i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 50 {
		t.Fatalf("Unexpected first id returned: %d", o)
	}

	err = i.GetSpecificID(50)
	if err == nil {
		t.Fatal(err)
	}

	o, err = i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 51 {
		t.Fatalf("Unexpected id returned: %d", o)
	}

	o, err = i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 52 {
		t.Fatalf("Unexpected id returned: %d", o)
	}

	o, err = i.GetID()
	if err == nil {
		t.Fatalf("Expected failure but succeeded: %d", o)
	}

	i.Release(50)

	o, err = i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 50 {
		t.Fatal("Unexpected id returned")
	}

	i.Release(52)
	err = i.GetSpecificID(52)
	if err != nil {
		t.Fatal(err)
	}
}

func TestUninitialized(t *testing.T) {
	i := &Idm{}

	if _, err := i.GetID(); err == nil {
		t.Fatal("Expected failure but succeeded")
	}

	if err := i.GetSpecificID(44); err == nil {
		t.Fatal("Expected failure but succeeded")
	}
}

func TestAllocateInRange(t *testing.T) {
	i, err := New(nil, "myset", 5, 10)
	if err != nil {
		t.Fatal(err)
	}

	o, err := i.GetIDInRange(6, 6)
	if err != nil {
		t.Fatal(err)
	}
	if o != 6 {
		t.Fatalf("Unexpected id returned. Expected: 6. Got: %d", o)
	}

	if err = i.GetSpecificID(6); err == nil {
		t.Fatalf("Expected failure but succeeded")
	}

	o, err = i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 5 {
		t.Fatalf("Unexpected id returned. Expected: 5. Got: %d", o)
	}

	i.Release(6)

	o, err = i.GetID()
	if err != nil {
		t.Fatal(err)
	}
	if o != 6 {
		t.Fatalf("Unexpected id returned. Expected: 6. Got: %d", o)
	}

	for n := 7; n <= 10; n++ {
		o, err := i.GetIDInRange(7, 10)
		if err != nil {
			t.Fatal(err)
		}
		if o != uint64(n) {
			t.Fatalf("Unexpected id returned. Expected: %d. Got: %d", n, o)
		}
	}

	if err = i.GetSpecificID(7); err == nil {
		t.Fatalf("Expected failure but succeeded")
	}

	if err = i.GetSpecificID(10); err == nil {
		t.Fatalf("Expected failure but succeeded")
	}

	i.Release(10)

	o, err = i.GetIDInRange(5, 10)
	if err != nil {
		t.Fatal(err)
	}
	if o != 10 {
		t.Fatalf("Unexpected id returned. Expected: 10. Got: %d", o)
	}

	i.Release(5)

	o, err = i.GetIDInRange(5, 10)
	if err != nil {
		t.Fatal(err)
	}
	if o != 5 {
		t.Fatalf("Unexpected id returned. Expected: 5. Got: %d", o)
	}

	for n := 5; n <= 10; n++ {
		i.Release(uint64(n))
	}

	for n := 5; n <= 10; n++ {
		o, err := i.GetIDInRange(5, 10)
		if err != nil {
			t.Fatal(err)
		}
		if o != uint64(n) {
			t.Fatalf("Unexpected id returned. Expected: %d. Got: %d", n, o)
		}
	}

	for n := 5; n <= 10; n++ {
		if err = i.GetSpecificID(uint64(n)); err == nil {
			t.Fatalf("Expected failure but succeeded for id: %d", n)
		}
	}

	// New larger set
	ul := uint64((1 << 24) - 1)
	i, err = New(nil, "newset", 0, ul)
	if err != nil {
		t.Fatal(err)
	}

	o, err = i.GetIDInRange(4096, ul)
	if err != nil {
		t.Fatal(err)
	}
	if o != 4096 {
		t.Fatalf("Unexpected id returned. Expected: 4096. Got: %d", o)
	}

	o, err = i.GetIDInRange(4096, ul)
	if err != nil {
		t.Fatal(err)
	}
	if o != 4097 {
		t.Fatalf("Unexpected id returned. Expected: 4097. Got: %d", o)
	}

	o, err = i.GetIDInRange(4096, ul)
	if err != nil {
		t.Fatal(err)
	}
	if o != 4098 {
		t.Fatalf("Unexpected id returned. Expected: 4098. Got: %d", o)
	}
}
