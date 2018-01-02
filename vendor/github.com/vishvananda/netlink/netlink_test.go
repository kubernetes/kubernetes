// +build linux

package netlink

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

type tearDownNetlinkTest func()

func skipUnlessRoot(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("Test requires root privileges.")
	}
}

func setUpNetlinkTest(t *testing.T) tearDownNetlinkTest {
	skipUnlessRoot(t)

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

func setUpNetlinkTestWithLoopback(t *testing.T) tearDownNetlinkTest {
	skipUnlessRoot(t)

	runtime.LockOSThread()
	ns, err := netns.New()
	if err != nil {
		t.Fatal("Failed to create new netns", ns)
	}

	link, err := LinkByName("lo")
	if err != nil {
		t.Fatalf("Failed to find \"lo\" in new netns: %v", err)
	}
	if err := LinkSetUp(link); err != nil {
		t.Fatalf("Failed to bring up \"lo\" in new netns: %v", err)
	}

	return func() {
		ns.Close()
		runtime.UnlockOSThread()
	}
}

func setUpMPLSNetlinkTest(t *testing.T) tearDownNetlinkTest {
	if _, err := os.Stat("/proc/sys/net/mpls/platform_labels"); err != nil {
		t.Skip("Test requires MPLS support.")
	}
	f := setUpNetlinkTest(t)
	setUpF := func(path, value string) {
		file, err := os.Create(path)
		defer file.Close()
		if err != nil {
			t.Fatalf("Failed to open %s: %s", path, err)
		}
		file.WriteString(value)
	}
	setUpF("/proc/sys/net/mpls/platform_labels", "1024")
	setUpF("/proc/sys/net/mpls/conf/lo/input", "1")
	return f
}

func setUpSEG6NetlinkTest(t *testing.T) tearDownNetlinkTest {
	// check if SEG6 options are enabled in Kernel Config
	cmd := exec.Command("uname", "-r")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		t.Fatal("Failed to run: uname -r")
	}
	s := []string{"/boot/config-", strings.TrimRight(out.String(), "\n")}
	filename := strings.Join(s, "")

	grepKey := func(key, fname string) (string, error) {
		cmd := exec.Command("grep", key, filename)
		var out bytes.Buffer
		cmd.Stdout = &out
		err := cmd.Run() // "err != nil" if no line matched with grep
		return strings.TrimRight(out.String(), "\n"), err
	}
	key := string("CONFIG_IPV6_SEG6_LWTUNNEL=y")
	if _, err := grepKey(key, filename); err != nil {
		msg := "Skipped test because it requires SEG6_LWTUNNEL support."
		log.Printf(msg)
		t.Skip(msg)
	}
	key = string("CONFIG_IPV6_SEG6_INLINE=y")
	if _, err := grepKey(key, filename); err != nil {
		msg := "Skipped test because it requires SEG6_INLINE support."
		log.Printf(msg)
		t.Skip(msg)
	}
	// Add CONFIG_IPV6_SEG6_HMAC to support seg6_hamc
	// key := string("CONFIG_IPV6_SEG6_HMAC=y")

	return setUpNetlinkTest(t)
}

func setUpNetlinkTestWithKModule(t *testing.T, name string) tearDownNetlinkTest {
	file, err := ioutil.ReadFile("/proc/modules")
	if err != nil {
		t.Fatal("Failed to open /proc/modules", err)
	}
	found := false
	for _, line := range strings.Split(string(file), "\n") {
		n := strings.Split(line, " ")[0]
		if n == name {
			found = true
			break
		}

	}
	if !found {
		t.Skipf("Test requires kmodule %q.", name)
	}
	return setUpNetlinkTest(t)
}

func remountSysfs() error {
	if err := unix.Mount("", "/", "none", unix.MS_SLAVE|unix.MS_REC, ""); err != nil {
		return err
	}
	if err := unix.Unmount("/sys", unix.MNT_DETACH); err != nil {
		return err
	}
	return unix.Mount("", "/sys", "sysfs", 0, "")
}

func minKernelRequired(t *testing.T, kernel, major int) {
	k, m, err := KernelVersion()
	if err != nil {
		t.Fatal(err)
	}
	if k < kernel || k == kernel && m < major {
		t.Skipf("Host Kernel (%d.%d) does not meet test's minimum required version: (%d.%d)",
			k, m, kernel, major)
	}
}

func KernelVersion() (kernel, major int, err error) {
	uts := unix.Utsname{}
	if err = unix.Uname(&uts); err != nil {
		return
	}

	ba := make([]byte, 0, len(uts.Release))
	for _, b := range uts.Release {
		if b == 0 {
			break
		}
		ba = append(ba, byte(b))
	}
	var rest string
	if n, _ := fmt.Sscanf(string(ba), "%d.%d%s", &kernel, &major, &rest); n < 2 {
		err = fmt.Errorf("can't parse kernel version in %q", string(ba))
	}
	return
}
