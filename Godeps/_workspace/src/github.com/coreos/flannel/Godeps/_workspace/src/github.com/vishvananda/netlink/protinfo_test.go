package netlink

import "testing"

func TestProtinfo(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	master := &Bridge{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(master); err != nil {
		t.Fatal(err)
	}
	iface1 := &Dummy{LinkAttrs{Name: "bar1", MasterIndex: master.Index}}
	iface2 := &Dummy{LinkAttrs{Name: "bar2", MasterIndex: master.Index}}
	iface3 := &Dummy{LinkAttrs{Name: "bar3"}}

	if err := LinkAdd(iface1); err != nil {
		t.Fatal(err)
	}
	if err := LinkAdd(iface2); err != nil {
		t.Fatal(err)
	}
	if err := LinkAdd(iface3); err != nil {
		t.Fatal(err)
	}

	oldpi1, err := LinkGetProtinfo(iface1)
	if err != nil {
		t.Fatal(err)
	}
	oldpi2, err := LinkGetProtinfo(iface2)
	if err != nil {
		t.Fatal(err)
	}

	if err := LinkSetHairpin(iface1, true); err != nil {
		t.Fatal(err)
	}

	if err := LinkSetRootBlock(iface1, true); err != nil {
		t.Fatal(err)
	}

	pi1, err := LinkGetProtinfo(iface1)
	if err != nil {
		t.Fatal(err)
	}
	if !pi1.Hairpin {
		t.Fatalf("Hairpin mode is not enabled for %s, but should", iface1.Name)
	}
	if !pi1.RootBlock {
		t.Fatalf("RootBlock is not enabled for %s, but should", iface1.Name)
	}
	if pi1.Guard != oldpi1.Guard {
		t.Fatalf("Guard field was changed for %s but shouldn't", iface1.Name)
	}
	if pi1.FastLeave != oldpi1.FastLeave {
		t.Fatalf("FastLeave field was changed for %s but shouldn't", iface1.Name)
	}
	if pi1.Learning != oldpi1.Learning {
		t.Fatalf("Learning field was changed for %s but shouldn't", iface1.Name)
	}
	if pi1.Flood != oldpi1.Flood {
		t.Fatalf("Flood field was changed for %s but shouldn't", iface1.Name)
	}

	if err := LinkSetGuard(iface2, true); err != nil {
		t.Fatal(err)
	}
	if err := LinkSetLearning(iface2, false); err != nil {
		t.Fatal(err)
	}
	pi2, err := LinkGetProtinfo(iface2)
	if err != nil {
		t.Fatal(err)
	}
	if pi2.Hairpin {
		t.Fatalf("Hairpin mode is enabled for %s, but shouldn't", iface2.Name)
	}
	if !pi2.Guard {
		t.Fatalf("Guard is not enabled for %s, but should", iface2.Name)
	}
	if pi2.Learning {
		t.Fatalf("Learning is enabled for %s, but shouldn't", iface2.Name)
	}
	if pi2.RootBlock != oldpi2.RootBlock {
		t.Fatalf("RootBlock field was changed for %s but shouldn't", iface2.Name)
	}
	if pi2.FastLeave != oldpi2.FastLeave {
		t.Fatalf("FastLeave field was changed for %s but shouldn't", iface2.Name)
	}
	if pi2.Flood != oldpi2.Flood {
		t.Fatalf("Flood field was changed for %s but shouldn't", iface2.Name)
	}

	if err := LinkSetHairpin(iface3, true); err == nil || err.Error() != "operation not supported" {
		t.Fatalf("Set protinfo attrs for link without master is not supported, but err: %s", err)
	}
}
