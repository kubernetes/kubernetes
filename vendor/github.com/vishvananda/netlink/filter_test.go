package netlink

import (
	"syscall"
	"testing"
)

func TestFilterAddDel(t *testing.T) {
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
	redir, err := LinkByName("bar")
	if err != nil {
		t.Fatal(err)
	}
	if err := LinkSetUp(redir); err != nil {
		t.Fatal(err)
	}
	qdisc := &Ingress{
		QdiscAttrs: QdiscAttrs{
			LinkIndex: link.Attrs().Index,
			Handle:    MakeHandle(0xffff, 0),
			Parent:    HANDLE_INGRESS,
		},
	}
	if err := QdiscAdd(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err := QdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 1 {
		t.Fatal("Failed to add qdisc")
	}
	_, ok := qdiscs[0].(*Ingress)
	if !ok {
		t.Fatal("Qdisc is the wrong type")
	}
	filter := &U32{
		FilterAttrs: FilterAttrs{
			LinkIndex: link.Attrs().Index,
			Parent:    MakeHandle(0xffff, 0),
			Priority:  1,
			Protocol:  syscall.ETH_P_IP,
		},
		RedirIndex: redir.Attrs().Index,
	}
	if err := FilterAdd(filter); err != nil {
		t.Fatal(err)
	}
	filters, err := FilterList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(filters) != 1 {
		t.Fatal("Failed to add filter")
	}
	if err := FilterDel(filter); err != nil {
		t.Fatal(err)
	}
	filters, err = FilterList(link, MakeHandle(0xffff, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(filters) != 0 {
		t.Fatal("Failed to remove filter")
	}
	if err := QdiscDel(qdisc); err != nil {
		t.Fatal(err)
	}
	qdiscs, err = QdiscList(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(qdiscs) != 0 {
		t.Fatal("Failed to remove qdisc")
	}
}
