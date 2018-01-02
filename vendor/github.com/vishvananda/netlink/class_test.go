// +build linux

package netlink

import (
	"testing"
)

func SafeQdiscList(link Link) ([]Qdisc, error) {
	qdiscs, err := QdiscList(link)
	if err != nil {
		return nil, err
	}
	result := []Qdisc{}
	for _, qdisc := range qdiscs {
		// filter out pfifo_fast qdiscs because
		// older kernels don't return them
		_, pfifo := qdisc.(*PfifoFast)
		if !pfifo {
			result = append(result, qdisc)
		}
	}
	return result, nil
}

func TestClassAddDel(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	if err := LinkAdd(&Ifb{LinkAttrs{Name: "foo"}}); err != nil {
		t.Fatal(err)
	}
	if err := LinkAdd(&Ifb{LinkAttrs{Name: "bar"}}); err != nil {
		t.Fatal(err)
	}
	link, err := LinkByName("foo")
	if err != nil {
		t.Fatal(err)
	}
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}
	attrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(0xffff, 0),
		Parent:    HANDLE_ROOT,
	}
	qdisc := NewHtb(attrs)
	if err := QdiscAdd(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err := SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 1 {
		t.Fatal("Failed to add qdisc")
	}
	_, ok := qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	classattrs := ClassAttrs{
		LinkIndex: link.Attrs().Index,
		Parent:    MakeHandle(0xffff, 0),
		Handle:    MakeHandle(0xffff, 2),
	}

	htbclassattrs := HtbClassAttrs{
		Rate:    1234000,
		Cbuffer: 1690,
	}
	class := NewHtbClass(classattrs, htbclassattrs)
	if err := ClassAdd(class); err != nil {
		t.Fatal(err)
	}
	classes, err := ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 1 {
		t.Fatal("Failed to add class")
	}

	htb, ok := classes[0].(*HtbClass)
	if !ok {
		t.Fatal("Class is the wrong type")
	}
	if htb.Rate != class.Rate {
		t.Fatal("Rate doesn't match")
	}
	if htb.Ceil != class.Ceil {
		t.Fatal("Ceil doesn't match")
	}
	if htb.Buffer != class.Buffer {
		t.Fatal("Buffer doesn't match")
	}
	if htb.Cbuffer != class.Cbuffer {
		t.Fatal("Cbuffer doesn't match")
	}

	qattrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(0x2, 0),
		Parent:    MakeHandle(0xffff, 2),
	}
	nattrs := NetemQdiscAttrs{
		Latency:     20000,
		Loss:        23.4,
		Duplicate:   14.3,
		LossCorr:    8.34,
		Jitter:      1000,
		DelayCorr:   12.3,
		ReorderProb: 23.4,
		CorruptProb: 10.0,
		CorruptCorr: 10,
	}
	qdiscnetem := NewNetem(qattrs, nattrs)
	if err := QdiscAdd(qdiscnetem); err != nil {
		t.Fatal(err)
	}

	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 2 {
		t.Fatal("Failed to add qdisc")
	}
	_, ok = qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	netem, ok := qdiscs[1].(*Netem)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	// Compare the record we got from the list with the one we created
	if netem.Loss != qdiscnetem.Loss {
		t.Fatal("Loss does not match")
	}
	if netem.Latency != qdiscnetem.Latency {
		t.Fatal("Latency does not match")
	}
	if netem.CorruptProb != qdiscnetem.CorruptProb {
		t.Fatal("CorruptProb does not match")
	}
	if netem.Jitter != qdiscnetem.Jitter {
		t.Fatal("Jitter does not match")
	}
	if netem.LossCorr != qdiscnetem.LossCorr {
		t.Fatal("Loss does not match")
	}
	if netem.DuplicateCorr != qdiscnetem.DuplicateCorr {
		t.Fatal("DuplicateCorr does not match")
	}

	// Deletion
	if err := ClassDel(class); err != nil {
		t.Fatal(err)
	}
	classes, err = ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 0 {
		t.Fatal("Failed to remove class")
	}
	if err := QdiscDel(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 0 {
		t.Fatal("Failed to remove qdisc")
	}
}

func TestHtbClassAddHtbClassChangeDel(t *testing.T) {
	/**
	This test first set up a interface ans set up a Htb qdisc
	A HTB class is attach to it and a Netem qdisc is attached to that class
	Next, we test changing the HTB class in place and confirming the Netem is
	still attached. We also check that invoting ClassChange with a non-existing
	class will fail.
	Finally, we test ClassReplace. We confirm it correctly behave like
	ClassChange when the parent/handle pair exists and that it will create a
	new class if the handle is modified.
	*/
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	if err := LinkAdd(&Ifb{LinkAttrs{Name: "foo"}}); err != nil {
		t.Fatal(err)
	}
	link, err := LinkByName("foo")
	if err != nil {
		t.Fatal(err)
	}
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}
	attrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(0xffff, 0),
		Parent:    HANDLE_ROOT,
	}
	qdisc := NewHtb(attrs)
	if err := QdiscAdd(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err := SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 1 {
		t.Fatal("Failed to add qdisc")
	}
	_, ok := qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	classattrs := ClassAttrs{
		LinkIndex: link.Attrs().Index,
		Parent:    MakeHandle(0xffff, 0),
		Handle:    MakeHandle(0xffff, 2),
	}

	htbclassattrs := HtbClassAttrs{
		Rate:    1234000,
		Cbuffer: 1690,
	}
	class := NewHtbClass(classattrs, htbclassattrs)
	if err := ClassAdd(class); err != nil {
		t.Fatal(err)
	}
	classes, err := ClassList(link, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 1 {
		t.Fatal("Failed to add class")
	}

	htb, ok := classes[0].(*HtbClass)
	if !ok {
		t.Fatal("Class is the wrong type")
	}

	qattrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(0x2, 0),
		Parent:    MakeHandle(0xffff, 2),
	}
	nattrs := NetemQdiscAttrs{
		Latency:     20000,
		Loss:        23.4,
		Duplicate:   14.3,
		LossCorr:    8.34,
		Jitter:      1000,
		DelayCorr:   12.3,
		ReorderProb: 23.4,
		CorruptProb: 10.0,
		CorruptCorr: 10,
	}
	qdiscnetem := NewNetem(qattrs, nattrs)
	if err := QdiscAdd(qdiscnetem); err != nil {
		t.Fatal(err)
	}

	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 2 {
		t.Fatal("Failed to add qdisc")
	}

	_, ok = qdiscs[1].(*Netem)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	// Change
	// For change to work, the handle and parent cannot be changed.

	// First, test it fails if we change the Handle.
	oldHandle := classattrs.Handle
	classattrs.Handle = MakeHandle(0xffff, 3)
	class = NewHtbClass(classattrs, htbclassattrs)
	if err := ClassChange(class); err == nil {
		t.Fatal("ClassChange should not work when using a different handle.")
	}
	// It should work with the same handle
	classattrs.Handle = oldHandle
	htbclassattrs.Rate = 4321000
	class = NewHtbClass(classattrs, htbclassattrs)
	if err := ClassChange(class); err != nil {
		t.Fatal(err)
	}

	classes, err = ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 1 {
		t.Fatalf(
			"1 class expected, %d found",
			len(classes),
		)
	}

	htb, ok = classes[0].(*HtbClass)
	if !ok {
		t.Fatal("Class is the wrong type")
	}
	// Verify that the rate value has changed.
	if htb.Rate != class.Rate {
		t.Fatal("Rate did not get changed while changing the class.")
	}

	// Check that we still have the netem child qdisc
	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}

	if len(qdiscs) != 2 {
		t.Fatalf("2 qdisc expected, %d found", len(qdiscs))
	}
	_, ok = qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	_, ok = qdiscs[1].(*Netem)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}

	// Replace
	// First replace by keeping the same handle, class will be changed.
	// Then, replace by providing a new handle, n new class will be created.

	// Replace acting as Change
	class = NewHtbClass(classattrs, htbclassattrs)
	if err := ClassReplace(class); err != nil {
		t.Fatal("Failed to replace class that is existing.")
	}

	classes, err = ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 1 {
		t.Fatalf(
			"1 class expected, %d found",
			len(classes),
		)
	}

	htb, ok = classes[0].(*HtbClass)
	if !ok {
		t.Fatal("Class is the wrong type")
	}
	// Verify that the rate value has changed.
	if htb.Rate != class.Rate {
		t.Fatal("Rate did not get changed while changing the class.")
	}

	// It should work with the same handle
	classattrs.Handle = MakeHandle(0xffff, 3)
	class = NewHtbClass(classattrs, htbclassattrs)
	if err := ClassReplace(class); err != nil {
		t.Fatal(err)
	}

	classes, err = ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 2 {
		t.Fatalf(
			"2 classes expected, %d found",
			len(classes),
		)
	}

	htb, ok = classes[1].(*HtbClass)
	if !ok {
		t.Fatal("Class is the wrong type")
	}
	// Verify that the rate value has changed.
	if htb.Rate != class.Rate {
		t.Fatal("Rate did not get changed while changing the class.")
	}

	// Deletion
	for _, class := range classes {
		if err := ClassDel(class); err != nil {
			t.Fatal(err)
		}
	}

	classes, err = ClassList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(classes) != 0 {
		t.Fatal("Failed to remove class")
	}
	if err := QdiscDel(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 0 {
		t.Fatal("Failed to remove qdisc")
	}
}
