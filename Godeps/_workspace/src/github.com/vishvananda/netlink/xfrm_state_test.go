package netlink

import (
	"net"
	"testing"
)

func TestXfrmStateAddDel(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	state := XfrmState{
		Src:   net.ParseIP("127.0.0.1"),
		Dst:   net.ParseIP("127.0.0.2"),
		Proto: XFRM_PROTO_ESP,
		Mode:  XFRM_MODE_TUNNEL,
		Spi:   1,
		Auth: &XfrmStateAlgo{
			Name: "hmac(sha256)",
			Key:  []byte("abcdefghijklmnopqrstuvwzyzABCDEF"),
		},
		Crypt: &XfrmStateAlgo{
			Name: "cbc(aes)",
			Key:  []byte("abcdefghijklmnopqrstuvwzyzABCDEF"),
		},
	}
	if err := XfrmStateAdd(&state); err != nil {
		t.Fatal(err)
	}
	policies, err := XfrmStateList(FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}

	if len(policies) != 1 {
		t.Fatal("State not added properly")
	}

	if err = XfrmStateDel(&state); err != nil {
		t.Fatal(err)
	}

	policies, err = XfrmStateList(FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}
	if len(policies) != 0 {
		t.Fatal("State not removed properly")
	}
}
