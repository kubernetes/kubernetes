package netlink

import (
	"log"
	"os"
	"runtime"
	"testing"

	"github.com/vishvananda/netns"
)

type tearDownNetlinkTest func()

func setUpNetlinkTest(t *testing.T) tearDownNetlinkTest {
	if os.Getuid() != 0 {
		msg := "Skipped test because it requires root privileges."
		log.Printf(msg)
		t.Skip(msg)
	}

	// new temporary namespace so we don't pollute the host
	// lock thread since the namespace is thread local
	runtime.LockOSThread()
	var err error
	ns, err := netns.New()
	if err != nil {
		t.Fatal("Failed to create newns", ns)
	}

	return func() {
		ns.Close()
		runtime.UnlockOSThread()
	}
}
