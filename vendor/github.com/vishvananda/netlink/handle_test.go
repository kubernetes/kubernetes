// +build linux

package netlink

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

func TestHandleCreateDelete(t *testing.T) {
	h, err := NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range nl.SupportedNlFamilies {
		sh, ok := h.sockets[f]
		if !ok {
			t.Fatalf("Handle socket(s) for family %d was not created", f)
		}
		if sh.Socket == nil {
			t.Fatalf("Socket for family %d was not created", f)
		}
	}

	h.Delete()
	if h.sockets != nil {
		t.Fatalf("Handle socket(s) were not destroyed")
	}
}

func TestHandleCreateNetns(t *testing.T) {
	skipUnlessRoot(t)

	id := make([]byte, 4)
	if _, err := io.ReadFull(rand.Reader, id); err != nil {
		t.Fatal(err)
	}
	ifName := "dummy-" + hex.EncodeToString(id)

	// Create an handle on the current netns
	curNs, err := netns.Get()
	if err != nil {
		t.Fatal(err)
	}
	defer curNs.Close()

	ch, err := NewHandleAt(curNs)
	if err != nil {
		t.Fatal(err)
	}
	defer ch.Delete()

	// Create an handle on a custom netns
	newNs, err := netns.New()
	if err != nil {
		t.Fatal(err)
	}
	defer newNs.Close()

	nh, err := NewHandleAt(newNs)
	if err != nil {
		t.Fatal(err)
	}
	defer nh.Delete()

	// Create an interface using the current handle
	err = ch.LinkAdd(&Dummy{LinkAttrs{Name: ifName}})
	if err != nil {
		t.Fatal(err)
	}
	l, err := ch.LinkByName(ifName)
	if err != nil {
		t.Fatal(err)
	}
	if l.Type() != "dummy" {
		t.Fatalf("Unexpected link type: %s", l.Type())
	}

	// Verify the new handle cannot find the interface
	ll, err := nh.LinkByName(ifName)
	if err == nil {
		t.Fatalf("Unexpected link found on netns %s: %v", newNs, ll)
	}

	// Move the interface to the new netns
	err = ch.LinkSetNsFd(l, int(newNs))
	if err != nil {
		t.Fatal(err)
	}

	// Verify new netns handle can find the interface while current cannot
	ll, err = nh.LinkByName(ifName)
	if err != nil {
		t.Fatal(err)
	}
	if ll.Type() != "dummy" {
		t.Fatalf("Unexpected link type: %s", ll.Type())
	}
	ll, err = ch.LinkByName(ifName)
	if err == nil {
		t.Fatalf("Unexpected link found on netns %s: %v", curNs, ll)
	}
}

func TestHandleTimeout(t *testing.T) {
	h, err := NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer h.Delete()

	for _, sh := range h.sockets {
		verifySockTimeVal(t, sh.Socket.GetFd(), unix.Timeval{Sec: 0, Usec: 0})
	}

	h.SetSocketTimeout(2*time.Second + 8*time.Millisecond)

	for _, sh := range h.sockets {
		verifySockTimeVal(t, sh.Socket.GetFd(), unix.Timeval{Sec: 2, Usec: 8000})
	}
}

func TestHandleReceiveBuffer(t *testing.T) {
	h, err := NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer h.Delete()
	if err := h.SetSocketReceiveBufferSize(65536, false); err != nil {
		t.Fatal(err)
	}
	sizes, err := h.GetSocketReceiveBufferSize()
	if err != nil {
		t.Fatal(err)
	}
	if len(sizes) != len(h.sockets) {
		t.Fatalf("Unexpected number of socket buffer sizes: %d (expected %d)",
			len(sizes), len(h.sockets))
	}
	for _, s := range sizes {
		if s < 65536 || s > 2*65536 {
			t.Fatalf("Unexpected socket receive buffer size: %d (expected around %d)",
				s, 65536)
		}
	}
}

func verifySockTimeVal(t *testing.T, fd int, tv unix.Timeval) {
	var (
		tr unix.Timeval
		v  = uint32(0x10)
	)
	_, _, errno := unix.Syscall6(unix.SYS_GETSOCKOPT, uintptr(fd), unix.SOL_SOCKET, unix.SO_SNDTIMEO, uintptr(unsafe.Pointer(&tr)), uintptr(unsafe.Pointer(&v)), 0)
	if errno != 0 {
		t.Fatal(errno)
	}

	if tr.Sec != tv.Sec || tr.Usec != tv.Usec {
		t.Fatalf("Unexpected timeout value read: %v. Expected: %v", tr, tv)
	}

	_, _, errno = unix.Syscall6(unix.SYS_GETSOCKOPT, uintptr(fd), unix.SOL_SOCKET, unix.SO_RCVTIMEO, uintptr(unsafe.Pointer(&tr)), uintptr(unsafe.Pointer(&v)), 0)
	if errno != 0 {
		t.Fatal(errno)
	}

	if tr.Sec != tv.Sec || tr.Usec != tv.Usec {
		t.Fatalf("Unexpected timeout value read: %v. Expected: %v", tr, tv)
	}
}

var (
	iter      = 10
	numThread = uint32(4)
	prefix    = "iface"
	handle1   *Handle
	handle2   *Handle
	ns1       netns.NsHandle
	ns2       netns.NsHandle
	done      uint32
	initError error
	once      sync.Once
)

func getXfrmState(thread int) *XfrmState {
	return &XfrmState{
		Src:   net.IPv4(byte(192), byte(168), 1, byte(1+thread)),
		Dst:   net.IPv4(byte(192), byte(168), 2, byte(1+thread)),
		Proto: XFRM_PROTO_AH,
		Mode:  XFRM_MODE_TUNNEL,
		Spi:   thread,
		Auth: &XfrmStateAlgo{
			Name: "hmac(sha256)",
			Key:  []byte("abcdefghijklmnopqrstuvwzyzABCDEF"),
		},
	}
}

func getXfrmPolicy(thread int) *XfrmPolicy {
	return &XfrmPolicy{
		Src:     &net.IPNet{IP: net.IPv4(byte(10), byte(10), byte(thread), 0), Mask: []byte{255, 255, 255, 0}},
		Dst:     &net.IPNet{IP: net.IPv4(byte(10), byte(10), byte(thread), 0), Mask: []byte{255, 255, 255, 0}},
		Proto:   17,
		DstPort: 1234,
		SrcPort: 5678,
		Dir:     XFRM_DIR_OUT,
		Tmpls: []XfrmPolicyTmpl{
			{
				Src:   net.IPv4(byte(192), byte(168), 1, byte(thread)),
				Dst:   net.IPv4(byte(192), byte(168), 2, byte(thread)),
				Proto: XFRM_PROTO_ESP,
				Mode:  XFRM_MODE_TUNNEL,
			},
		},
	}
}
func initParallel() {
	ns1, initError = netns.New()
	if initError != nil {
		return
	}
	handle1, initError = NewHandleAt(ns1)
	if initError != nil {
		return
	}
	ns2, initError = netns.New()
	if initError != nil {
		return
	}
	handle2, initError = NewHandleAt(ns2)
	if initError != nil {
		return
	}
}

func parallelDone() {
	atomic.AddUint32(&done, 1)
	if done == numThread {
		if ns1.IsOpen() {
			ns1.Close()
		}
		if ns2.IsOpen() {
			ns2.Close()
		}
		if handle1 != nil {
			handle1.Delete()
		}
		if handle2 != nil {
			handle2.Delete()
		}
	}
}

// Do few route and xfrm operation on the two handles in parallel
func runParallelTests(t *testing.T, thread int) {
	skipUnlessRoot(t)
	defer parallelDone()

	t.Parallel()

	once.Do(initParallel)
	if initError != nil {
		t.Fatal(initError)
	}

	state := getXfrmState(thread)
	policy := getXfrmPolicy(thread)
	for i := 0; i < iter; i++ {
		ifName := fmt.Sprintf("%s_%d_%d", prefix, thread, i)
		link := &Dummy{LinkAttrs{Name: ifName}}
		err := handle1.LinkAdd(link)
		if err != nil {
			t.Fatal(err)
		}
		l, err := handle1.LinkByName(ifName)
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.LinkSetUp(l)
		if err != nil {
			t.Fatal(err)
		}
		handle1.LinkSetNsFd(l, int(ns2))
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.XfrmStateAdd(state)
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.XfrmPolicyAdd(policy)
		if err != nil {
			t.Fatal(err)
		}
		err = handle2.LinkSetDown(l)
		if err != nil {
			t.Fatal(err)
		}
		err = handle2.XfrmStateAdd(state)
		if err != nil {
			t.Fatal(err)
		}
		err = handle2.XfrmPolicyAdd(policy)
		if err != nil {
			t.Fatal(err)
		}
		_, err = handle2.LinkByName(ifName)
		if err != nil {
			t.Fatal(err)
		}
		handle2.LinkSetNsFd(l, int(ns1))
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.LinkSetUp(l)
		if err != nil {
			t.Fatal(err)
		}
		l, err = handle1.LinkByName(ifName)
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.XfrmPolicyDel(policy)
		if err != nil {
			t.Fatal(err)
		}
		err = handle2.XfrmPolicyDel(policy)
		if err != nil {
			t.Fatal(err)
		}
		err = handle1.XfrmStateDel(state)
		if err != nil {
			t.Fatal(err)
		}
		err = handle2.XfrmStateDel(state)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestHandleParallel1(t *testing.T) {
	runParallelTests(t, 1)
}

func TestHandleParallel2(t *testing.T) {
	runParallelTests(t, 2)
}

func TestHandleParallel3(t *testing.T) {
	runParallelTests(t, 3)
}

func TestHandleParallel4(t *testing.T) {
	runParallelTests(t, 4)
}
