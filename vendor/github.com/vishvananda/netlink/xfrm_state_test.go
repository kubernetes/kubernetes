// +build linux

package netlink

import (
	"bytes"
	"encoding/hex"
	"net"
	"testing"
	"time"
)

func TestXfrmStateAddGetDel(t *testing.T) {
	for _, s := range []*XfrmState{getBaseState(), getAeadState()} {
		testXfrmStateAddGetDel(t, s)
	}
}

func testXfrmStateAddGetDel(t *testing.T, state *XfrmState) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	if err := XfrmStateAdd(state); err != nil {
		t.Fatal(err)
	}
	states, err := XfrmStateList(FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}

	if len(states) != 1 {
		t.Fatal("State not added properly")
	}

	if !compareStates(state, &states[0]) {
		t.Fatalf("unexpected states returned")
	}

	// Get specific state
	sa, err := XfrmStateGet(state)
	if err != nil {
		t.Fatal(err)
	}

	if !compareStates(state, sa) {
		t.Fatalf("unexpected state returned")
	}

	if err = XfrmStateDel(state); err != nil {
		t.Fatal(err)
	}

	states, err = XfrmStateList(FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 0 {
		t.Fatal("State not removed properly")
	}

	if _, err := XfrmStateGet(state); err == nil {
		t.Fatalf("Unexpected success")
	}
}

func TestXfrmStateAllocSpi(t *testing.T) {
	defer setUpNetlinkTest(t)()

	state := getBaseState()
	state.Spi = 0
	state.Auth = nil
	state.Crypt = nil
	rstate, err := XfrmStateAllocSpi(state)
	if err != nil {
		t.Fatal(err)
	}
	if rstate.Spi == 0 {
		t.Fatalf("SPI is not allocated")
	}
	rstate.Spi = 0
	if !compareStates(state, rstate) {
		t.Fatalf("State not properly allocated")
	}
}

func TestXfrmStateFlush(t *testing.T) {
	defer setUpNetlinkTest(t)()

	state1 := getBaseState()
	state2 := getBaseState()
	state2.Src = net.ParseIP("127.1.0.1")
	state2.Dst = net.ParseIP("127.1.0.2")
	state2.Proto = XFRM_PROTO_AH
	state2.Mode = XFRM_MODE_TUNNEL
	state2.Spi = 20
	state2.Mark = nil
	state2.Crypt = nil

	if err := XfrmStateAdd(state1); err != nil {
		t.Fatal(err)
	}
	if err := XfrmStateAdd(state2); err != nil {
		t.Fatal(err)
	}

	// flushing proto for which no state is present should return silently
	if err := XfrmStateFlush(XFRM_PROTO_COMP); err != nil {
		t.Fatal(err)
	}

	if err := XfrmStateFlush(XFRM_PROTO_AH); err != nil {
		t.Fatal(err)
	}

	if _, err := XfrmStateGet(state2); err == nil {
		t.Fatalf("Unexpected success")
	}

	if err := XfrmStateAdd(state2); err != nil {
		t.Fatal(err)
	}

	if err := XfrmStateFlush(0); err != nil {
		t.Fatal(err)
	}

	states, err := XfrmStateList(FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 0 {
		t.Fatal("State not flushed properly")
	}

}

func TestXfrmStateUpdateLimits(t *testing.T) {
	defer setUpNetlinkTest(t)()

	// Program state with limits
	state := getBaseState()
	state.Limits.TimeHard = 3600
	state.Limits.TimeSoft = 60
	state.Limits.PacketHard = 1000
	state.Limits.PacketSoft = 50
	state.Limits.ByteHard = 1000000
	state.Limits.ByteSoft = 50000
	state.Limits.TimeUseHard = 3000
	state.Limits.TimeUseSoft = 1500
	if err := XfrmStateAdd(state); err != nil {
		t.Fatal(err)
	}
	// Verify limits
	s, err := XfrmStateGet(state)
	if err != nil {
		t.Fatal(err)
	}
	if !compareLimits(state, s) {
		t.Fatalf("Incorrect time hard/soft retrieved: %s", s.Print(true))
	}

	// Update limits
	state.Limits.TimeHard = 1800
	state.Limits.TimeSoft = 30
	state.Limits.PacketHard = 500
	state.Limits.PacketSoft = 25
	state.Limits.ByteHard = 500000
	state.Limits.ByteSoft = 25000
	state.Limits.TimeUseHard = 2000
	state.Limits.TimeUseSoft = 1000
	if err := XfrmStateUpdate(state); err != nil {
		t.Fatal(err)
	}

	// Verify new limits
	s, err = XfrmStateGet(state)
	if err != nil {
		t.Fatal(err)
	}
	if s.Limits.TimeHard != 1800 || s.Limits.TimeSoft != 30 {
		t.Fatalf("Incorrect time hard retrieved: (%d, %d)", s.Limits.TimeHard, s.Limits.TimeSoft)
	}
}

func TestXfrmStateStats(t *testing.T) {
	defer setUpNetlinkTest(t)()

	// Program state and record time
	state := getBaseState()
	now := time.Now()
	if err := XfrmStateAdd(state); err != nil {
		t.Fatal(err)
	}
	// Retrieve state
	s, err := XfrmStateGet(state)
	if err != nil {
		t.Fatal(err)
	}
	// Verify stats: We expect zero counters, same second add time and unset use time
	if s.Statistics.Bytes != 0 || s.Statistics.Packets != 0 || s.Statistics.AddTime != uint64(now.Unix()) || s.Statistics.UseTime != 0 {
		t.Fatalf("Unexpected statistics (addTime: %s) for state:\n%s", now.Format(time.UnixDate), s.Print(true))
	}
}

func getBaseState() *XfrmState {
	return &XfrmState{
		// Force 4 byte notation for the IPv4 addresses
		Src:   net.ParseIP("127.0.0.1").To4(),
		Dst:   net.ParseIP("127.0.0.2").To4(),
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
		Mark: &XfrmMark{
			Value: 0x12340000,
			Mask:  0xffff0000,
		},
	}
}

func getAeadState() *XfrmState {
	// 128 key bits + 32 salt bits
	k, _ := hex.DecodeString("d0562776bf0e75830ba3f7f8eb6c09b555aa1177")
	return &XfrmState{
		// Leave IPv4 addresses in Ipv4 in IPv6 notation
		Src:   net.ParseIP("192.168.1.1"),
		Dst:   net.ParseIP("192.168.2.2"),
		Proto: XFRM_PROTO_ESP,
		Mode:  XFRM_MODE_TUNNEL,
		Spi:   2,
		Aead: &XfrmStateAlgo{
			Name:   "rfc4106(gcm(aes))",
			Key:    k,
			ICVLen: 64,
		},
	}
}

func compareStates(a, b *XfrmState) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.Src.Equal(b.Src) && a.Dst.Equal(b.Dst) &&
		a.Mode == b.Mode && a.Spi == b.Spi && a.Proto == b.Proto &&
		compareAlgo(a.Auth, b.Auth) &&
		compareAlgo(a.Crypt, b.Crypt) &&
		compareAlgo(a.Aead, b.Aead) &&
		compareMarks(a.Mark, b.Mark)
}

func compareLimits(a, b *XfrmState) bool {
	return a.Limits.TimeHard == b.Limits.TimeHard &&
		a.Limits.TimeSoft == b.Limits.TimeSoft &&
		a.Limits.PacketHard == b.Limits.PacketHard &&
		a.Limits.PacketSoft == b.Limits.PacketSoft &&
		a.Limits.ByteHard == b.Limits.ByteHard &&
		a.Limits.ByteSoft == b.Limits.ByteSoft &&
		a.Limits.TimeUseHard == b.Limits.TimeUseHard &&
		a.Limits.TimeUseSoft == b.Limits.TimeUseSoft
}

func compareAlgo(a, b *XfrmStateAlgo) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.Name == b.Name && bytes.Equal(a.Key, b.Key) &&
		(a.TruncateLen == 0 || a.TruncateLen == b.TruncateLen) &&
		(a.ICVLen == 0 || a.ICVLen == b.ICVLen)
}

func compareMarks(a, b *XfrmMark) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.Value == b.Value && a.Mask == b.Mask
}
