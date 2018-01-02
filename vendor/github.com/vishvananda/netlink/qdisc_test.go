// +build linux

package netlink

import (
	"testing"
)

func TestTbfAddDel(t *testing.T) {
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
	qdisc := &Tbf{
		QdiscAttrs: QdiscAttrs{
			LinkIndex: link.Attrs().Index,
			Handle:    MakeHandle(1, 0),
			Parent:    HANDLE_ROOT,
		},
		Rate:   131072,
		Limit:  1220703,
		Buffer: 16793,
	}
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
	tbf, ok := qdiscs[0].(*Tbf)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if tbf.Rate != qdisc.Rate {
		t.Fatal("Rate doesn't match")
	}
	if tbf.Limit != qdisc.Limit {
		t.Fatal("Limit doesn't match")
	}
	if tbf.Buffer != qdisc.Buffer {
		t.Fatal("Buffer doesn't match")
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

func TestHtbAddDel(t *testing.T) {
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
		Handle:    MakeHandle(1, 0),
		Parent:    HANDLE_ROOT,
	}

	qdisc := NewHtb(attrs)
	qdisc.Rate2Quantum = 5
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
	htb, ok := qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if htb.Defcls != qdisc.Defcls {
		t.Fatal("Defcls doesn't match")
	}
	if htb.Rate2Quantum != qdisc.Rate2Quantum {
		t.Fatal("Rate2Quantum doesn't match")
	}
	if htb.Debug != qdisc.Debug {
		t.Fatal("Debug doesn't match")
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

func TestPrioAddDel(t *testing.T) {
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
	qdisc := NewPrio(QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(1, 0),
		Parent:    HANDLE_ROOT,
	})
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
	_, ok := qdiscs[0].(*Prio)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
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

func TestTbfAddHtbReplaceDel(t *testing.T) {
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

	// Add
	attrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(1, 0),
		Parent:    HANDLE_ROOT,
	}
	qdisc := &Tbf{
		QdiscAttrs: attrs,
		Rate:       131072,
		Limit:      1220703,
		Buffer:     16793,
	}
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
	tbf, ok := qdiscs[0].(*Tbf)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if tbf.Rate != qdisc.Rate {
		t.Fatal("Rate doesn't match")
	}
	if tbf.Limit != qdisc.Limit {
		t.Fatal("Limit doesn't match")
	}
	if tbf.Buffer != qdisc.Buffer {
		t.Fatal("Buffer doesn't match")
	}
	// Replace
	// For replace to work, the handle MUST be different that the running one
	attrs.Handle = MakeHandle(2, 0)
	qdisc2 := NewHtb(attrs)
	qdisc2.Rate2Quantum = 5
	if err := QdiscReplace(qdisc2); err != nil {
		t.Fatal(err)
	}

	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 1 {
		t.Fatal("Failed to add qdisc")
	}
	htb, ok := qdiscs[0].(*Htb)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if htb.Defcls != qdisc2.Defcls {
		t.Fatal("Defcls doesn't match")
	}
	if htb.Rate2Quantum != qdisc2.Rate2Quantum {
		t.Fatal("Rate2Quantum doesn't match")
	}
	if htb.Debug != qdisc2.Debug {
		t.Fatal("Debug doesn't match")
	}

	if err := QdiscDel(qdisc2); err != nil {
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

func TestTbfAddTbfChangeDel(t *testing.T) {
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

	// Add
	attrs := QdiscAttrs{
		LinkIndex: link.Attrs().Index,
		Handle:    MakeHandle(1, 0),
		Parent:    HANDLE_ROOT,
	}
	qdisc := &Tbf{
		QdiscAttrs: attrs,
		Rate:       131072,
		Limit:      1220703,
		Buffer:     16793,
	}
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
	tbf, ok := qdiscs[0].(*Tbf)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if tbf.Rate != qdisc.Rate {
		t.Fatal("Rate doesn't match")
	}
	if tbf.Limit != qdisc.Limit {
		t.Fatal("Limit doesn't match")
	}
	if tbf.Buffer != qdisc.Buffer {
		t.Fatal("Buffer doesn't match")
	}
	// Change
	// For change to work, the handle MUST not change
	qdisc.Rate = 23456
	if err := QdiscChange(qdisc); err != nil {
		t.Fatal(err)
	}

	qdiscs, err = SafeQdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 1 {
		t.Fatal("Failed to add qdisc")
	}
	tbf, ok = qdiscs[0].(*Tbf)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	if tbf.Rate != qdisc.Rate {
		t.Fatal("Rate doesn't match")
	}
	if tbf.Limit != qdisc.Limit {
		t.Fatal("Limit doesn't match")
	}
	if tbf.Buffer != qdisc.Buffer {
		t.Fatal("Buffer doesn't match")
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
