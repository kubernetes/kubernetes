// +build !solaris

package osl

import (
	"os"
	"runtime"
	"testing"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/testutils"
)

func TestMain(m *testing.M) {
	if reexec.Init() {
		return
	}
	os.Exit(m.Run())
}

func TestSandboxCreate(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()

	if s.Key() != key {
		t.Fatalf("s.Key() returned %s. Expected %s", s.Key(), key)
	}

	tbox, err := newInfo(ns.NlHandle(), t)
	if err != nil {
		t.Fatalf("Failed to generate new sandbox info: %v", err)
	}

	for _, i := range tbox.Info().Interfaces() {
		err = s.AddInterface(i.SrcName(), i.DstName(),
			tbox.InterfaceOptions().Bridge(i.Bridge()),
			tbox.InterfaceOptions().Address(i.Address()),
			tbox.InterfaceOptions().AddressIPv6(i.AddressIPv6()))
		if err != nil {
			t.Fatalf("Failed to add interfaces to sandbox: %v", err)
		}
	}

	err = s.SetGateway(tbox.Info().Gateway())
	if err != nil {
		t.Fatalf("Failed to set gateway to sandbox: %v", err)
	}

	err = s.SetGatewayIPv6(tbox.Info().GatewayIPv6())
	if err != nil {
		t.Fatalf("Failed to set ipv6 gateway to sandbox: %v", err)
	}

	verifySandbox(t, s, []string{"0", "1", "2"})

	err = s.Destroy()
	if err != nil {
		t.Fatal(err)
	}
	verifyCleanup(t, s, true)
}

func TestSandboxCreateTwice(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	_, err = NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()

	// Create another sandbox with the same key to see if we handle it
	// gracefully.
	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()

	err = s.Destroy()
	if err != nil {
		t.Fatal(err)
	}
	GC()
	verifyCleanup(t, s, false)
}

func TestSandboxGC(t *testing.T) {
	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}

	err = s.Destroy()
	if err != nil {
		t.Fatal(err)
	}

	GC()
	verifyCleanup(t, s, false)
}

func TestAddRemoveInterface(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()

	if s.Key() != key {
		t.Fatalf("s.Key() returned %s. Expected %s", s.Key(), key)
	}

	tbox, err := newInfo(ns.NlHandle(), t)
	if err != nil {
		t.Fatalf("Failed to generate new sandbox info: %v", err)
	}

	for _, i := range tbox.Info().Interfaces() {
		err = s.AddInterface(i.SrcName(), i.DstName(),
			tbox.InterfaceOptions().Bridge(i.Bridge()),
			tbox.InterfaceOptions().Address(i.Address()),
			tbox.InterfaceOptions().AddressIPv6(i.AddressIPv6()))
		if err != nil {
			t.Fatalf("Failed to add interfaces to sandbox: %v", err)
		}
	}

	verifySandbox(t, s, []string{"0", "1", "2"})

	interfaces := s.Info().Interfaces()
	if err := interfaces[0].Remove(); err != nil {
		t.Fatalf("Failed to remove interfaces from sandbox: %v", err)
	}

	verifySandbox(t, s, []string{"1", "2"})

	i := tbox.Info().Interfaces()[0]
	if err := s.AddInterface(i.SrcName(), i.DstName(),
		tbox.InterfaceOptions().Bridge(i.Bridge()),
		tbox.InterfaceOptions().Address(i.Address()),
		tbox.InterfaceOptions().AddressIPv6(i.AddressIPv6())); err != nil {
		t.Fatalf("Failed to add interfaces to sandbox: %v", err)
	}

	verifySandbox(t, s, []string{"1", "2", "3"})

	err = s.Destroy()
	if err != nil {
		t.Fatal(err)
	}

	GC()
	verifyCleanup(t, s, false)
}
