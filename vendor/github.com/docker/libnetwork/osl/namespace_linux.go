package osl

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netns"
)

const defaultPrefix = "/var/run/docker"

func init() {
	reexec.Register("set-ipv6", reexecSetIPv6)
}

var (
	once             sync.Once
	garbagePathMap   = make(map[string]bool)
	gpmLock          sync.Mutex
	gpmWg            sync.WaitGroup
	gpmCleanupPeriod = 60 * time.Second
	gpmChan          = make(chan chan struct{})
	prefix           = defaultPrefix
)

// The networkNamespace type is the linux implementation of the Sandbox
// interface. It represents a linux network namespace, and moves an interface
// into it when called on method AddInterface or sets the gateway etc.
type networkNamespace struct {
	path         string
	iFaces       []*nwIface
	gw           net.IP
	gwv6         net.IP
	staticRoutes []*types.StaticRoute
	neighbors    []*neigh
	nextIfIndex  map[string]int
	isDefault    bool
	nlHandle     *netlink.Handle
	loV6Enabled  bool
	sync.Mutex
}

// SetBasePath sets the base url prefix for the ns path
func SetBasePath(path string) {
	prefix = path
}

func init() {
	reexec.Register("netns-create", reexecCreateNamespace)
}

func basePath() string {
	return filepath.Join(prefix, "netns")
}

func createBasePath() {
	err := os.MkdirAll(basePath(), 0755)
	if err != nil {
		panic("Could not create net namespace path directory")
	}

	// Start the garbage collection go routine
	go removeUnusedPaths()
}

func removeUnusedPaths() {
	gpmLock.Lock()
	period := gpmCleanupPeriod
	gpmLock.Unlock()

	ticker := time.NewTicker(period)
	for {
		var (
			gc   chan struct{}
			gcOk bool
		)

		select {
		case <-ticker.C:
		case gc, gcOk = <-gpmChan:
		}

		gpmLock.Lock()
		pathList := make([]string, 0, len(garbagePathMap))
		for path := range garbagePathMap {
			pathList = append(pathList, path)
		}
		garbagePathMap = make(map[string]bool)
		gpmWg.Add(1)
		gpmLock.Unlock()

		for _, path := range pathList {
			os.Remove(path)
		}

		gpmWg.Done()
		if gcOk {
			close(gc)
		}
	}
}

func addToGarbagePaths(path string) {
	gpmLock.Lock()
	garbagePathMap[path] = true
	gpmLock.Unlock()
}

func removeFromGarbagePaths(path string) {
	gpmLock.Lock()
	delete(garbagePathMap, path)
	gpmLock.Unlock()
}

// GC triggers garbage collection of namespace path right away
// and waits for it.
func GC() {
	gpmLock.Lock()
	if len(garbagePathMap) == 0 {
		// No need for GC if map is empty
		gpmLock.Unlock()
		return
	}
	gpmLock.Unlock()

	// if content exists in the garbage paths
	// we can trigger GC to run, providing a
	// channel to be notified on completion
	waitGC := make(chan struct{})
	gpmChan <- waitGC
	// wait for GC completion
	<-waitGC
}

// GenerateKey generates a sandbox key based on the passed
// container id.
func GenerateKey(containerID string) string {
	maxLen := 12
	// Read sandbox key from host for overlay
	if strings.HasPrefix(containerID, "-") {
		var (
			index    int
			indexStr string
			tmpkey   string
		)
		dir, err := ioutil.ReadDir(basePath())
		if err != nil {
			return ""
		}

		for _, v := range dir {
			id := v.Name()
			if strings.HasSuffix(id, containerID[:maxLen-1]) {
				indexStr = strings.TrimSuffix(id, containerID[:maxLen-1])
				tmpindex, err := strconv.Atoi(indexStr)
				if err != nil {
					return ""
				}
				if tmpindex > index {
					index = tmpindex
					tmpkey = id
				}

			}
		}
		containerID = tmpkey
		if containerID == "" {
			return ""
		}
	}

	if len(containerID) < maxLen {
		maxLen = len(containerID)
	}

	return basePath() + "/" + containerID[:maxLen]
}

// NewSandbox provides a new sandbox instance created in an os specific way
// provided a key which uniquely identifies the sandbox
func NewSandbox(key string, osCreate, isRestore bool) (Sandbox, error) {
	if !isRestore {
		err := createNetworkNamespace(key, osCreate)
		if err != nil {
			return nil, err
		}
	} else {
		once.Do(createBasePath)
	}

	n := &networkNamespace{path: key, isDefault: !osCreate, nextIfIndex: make(map[string]int)}

	sboxNs, err := netns.GetFromPath(n.path)
	if err != nil {
		return nil, fmt.Errorf("failed get network namespace %q: %v", n.path, err)
	}
	defer sboxNs.Close()

	n.nlHandle, err = netlink.NewHandleAt(sboxNs, syscall.NETLINK_ROUTE)
	if err != nil {
		return nil, fmt.Errorf("failed to create a netlink handle: %v", err)
	}

	err = n.nlHandle.SetSocketTimeout(ns.NetlinkSocketsTimeout)
	if err != nil {
		logrus.Warnf("Failed to set the timeout on the sandbox netlink handle sockets: %v", err)
	}

	// As starting point, disable IPv6 on all interfaces
	if !n.isDefault {
		err = setIPv6(n.path, "all", false)
		if err != nil {
			logrus.Warnf("Failed to disable IPv6 on all interfaces on network namespace %q: %v", n.path, err)
		}
	}

	if err = n.loopbackUp(); err != nil {
		n.nlHandle.Delete()
		return nil, err
	}

	return n, nil
}

func (n *networkNamespace) InterfaceOptions() IfaceOptionSetter {
	return n
}

func (n *networkNamespace) NeighborOptions() NeighborOptionSetter {
	return n
}

func mountNetworkNamespace(basePath string, lnPath string) error {
	return syscall.Mount(basePath, lnPath, "bind", syscall.MS_BIND, "")
}

// GetSandboxForExternalKey returns sandbox object for the supplied path
func GetSandboxForExternalKey(basePath string, key string) (Sandbox, error) {
	if err := createNamespaceFile(key); err != nil {
		return nil, err
	}

	if err := mountNetworkNamespace(basePath, key); err != nil {
		return nil, err
	}
	n := &networkNamespace{path: key, nextIfIndex: make(map[string]int)}

	sboxNs, err := netns.GetFromPath(n.path)
	if err != nil {
		return nil, fmt.Errorf("failed get network namespace %q: %v", n.path, err)
	}
	defer sboxNs.Close()

	n.nlHandle, err = netlink.NewHandleAt(sboxNs, syscall.NETLINK_ROUTE)
	if err != nil {
		return nil, fmt.Errorf("failed to create a netlink handle: %v", err)
	}

	err = n.nlHandle.SetSocketTimeout(ns.NetlinkSocketsTimeout)
	if err != nil {
		logrus.Warnf("Failed to set the timeout on the sandbox netlink handle sockets: %v", err)
	}

	// As starting point, disable IPv6 on all interfaces
	err = setIPv6(n.path, "all", false)
	if err != nil {
		logrus.Warnf("Failed to disable IPv6 on all interfaces on network namespace %q: %v", n.path, err)
	}

	if err = n.loopbackUp(); err != nil {
		n.nlHandle.Delete()
		return nil, err
	}

	return n, nil
}

func reexecCreateNamespace() {
	if len(os.Args) < 2 {
		logrus.Fatal("no namespace path provided")
	}
	if err := mountNetworkNamespace("/proc/self/ns/net", os.Args[1]); err != nil {
		logrus.Fatal(err)
	}
}

func createNetworkNamespace(path string, osCreate bool) error {
	if err := createNamespaceFile(path); err != nil {
		return err
	}

	cmd := &exec.Cmd{
		Path:   reexec.Self(),
		Args:   append([]string{"netns-create"}, path),
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if osCreate {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
		cmd.SysProcAttr.Cloneflags = syscall.CLONE_NEWNET
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("namespace creation reexec command failed: %v", err)
	}

	return nil
}

func unmountNamespaceFile(path string) {
	if _, err := os.Stat(path); err == nil {
		syscall.Unmount(path, syscall.MNT_DETACH)
	}
}

func createNamespaceFile(path string) (err error) {
	var f *os.File

	once.Do(createBasePath)
	// Remove it from garbage collection list if present
	removeFromGarbagePaths(path)

	// If the path is there unmount it first
	unmountNamespaceFile(path)

	// wait for garbage collection to complete if it is in progress
	// before trying to create the file.
	gpmWg.Wait()

	if f, err = os.Create(path); err == nil {
		f.Close()
	}

	return err
}

func (n *networkNamespace) loopbackUp() error {
	iface, err := n.nlHandle.LinkByName("lo")
	if err != nil {
		return err
	}
	return n.nlHandle.LinkSetUp(iface)
}

func (n *networkNamespace) AddLoopbackAliasIP(ip *net.IPNet) error {
	iface, err := n.nlHandle.LinkByName("lo")
	if err != nil {
		return err
	}
	return n.nlHandle.AddrAdd(iface, &netlink.Addr{IPNet: ip})
}

func (n *networkNamespace) RemoveLoopbackAliasIP(ip *net.IPNet) error {
	iface, err := n.nlHandle.LinkByName("lo")
	if err != nil {
		return err
	}
	return n.nlHandle.AddrDel(iface, &netlink.Addr{IPNet: ip})
}

func (n *networkNamespace) InvokeFunc(f func()) error {
	return nsInvoke(n.nsPath(), func(nsFD int) error { return nil }, func(callerFD int) error {
		f()
		return nil
	})
}

// InitOSContext initializes OS context while configuring network resources
func InitOSContext() func() {
	runtime.LockOSThread()
	if err := ns.SetNamespace(); err != nil {
		logrus.Error(err)
	}
	return runtime.UnlockOSThread
}

func nsInvoke(path string, prefunc func(nsFD int) error, postfunc func(callerFD int) error) error {
	defer InitOSContext()()

	newNs, err := netns.GetFromPath(path)
	if err != nil {
		return fmt.Errorf("failed get network namespace %q: %v", path, err)
	}
	defer newNs.Close()

	// Invoked before the namespace switch happens but after the namespace file
	// handle is obtained.
	if err := prefunc(int(newNs)); err != nil {
		return fmt.Errorf("failed in prefunc: %v", err)
	}

	if err = netns.Set(newNs); err != nil {
		return err
	}
	defer ns.SetNamespace()

	// Invoked after the namespace switch.
	return postfunc(ns.ParseHandlerInt())
}

func (n *networkNamespace) nsPath() string {
	n.Lock()
	defer n.Unlock()

	return n.path
}

func (n *networkNamespace) Info() Info {
	return n
}

func (n *networkNamespace) Key() string {
	return n.path
}

func (n *networkNamespace) Destroy() error {
	if n.nlHandle != nil {
		n.nlHandle.Delete()
	}
	// Assuming no running process is executing in this network namespace,
	// unmounting is sufficient to destroy it.
	if err := syscall.Unmount(n.path, syscall.MNT_DETACH); err != nil {
		return err
	}

	// Stash it into the garbage collection list
	addToGarbagePaths(n.path)
	return nil
}

// Restore restore the network namespace
func (n *networkNamespace) Restore(ifsopt map[string][]IfaceOption, routes []*types.StaticRoute, gw net.IP, gw6 net.IP) error {
	// restore interfaces
	for name, opts := range ifsopt {
		if !strings.Contains(name, "+") {
			return fmt.Errorf("wrong iface name in restore osl sandbox interface: %s", name)
		}
		seps := strings.Split(name, "+")
		srcName := seps[0]
		dstPrefix := seps[1]
		i := &nwIface{srcName: srcName, dstName: dstPrefix, ns: n}
		i.processInterfaceOptions(opts...)
		if i.master != "" {
			i.dstMaster = n.findDst(i.master, true)
			if i.dstMaster == "" {
				return fmt.Errorf("could not find an appropriate master %q for %q",
					i.master, i.srcName)
			}
		}
		if n.isDefault {
			i.dstName = i.srcName
		} else {
			links, err := n.nlHandle.LinkList()
			if err != nil {
				return fmt.Errorf("failed to retrieve list of links in network namespace %q during restore", n.path)
			}
			// due to the docker network connect/disconnect, so the dstName should
			// restore from the namespace
			for _, link := range links {
				addrs, err := n.nlHandle.AddrList(link, netlink.FAMILY_V4)
				if err != nil {
					return err
				}
				ifaceName := link.Attrs().Name
				if strings.HasPrefix(ifaceName, "vxlan") {
					if i.dstName == "vxlan" {
						i.dstName = ifaceName
						break
					}
				}
				// find the interface name by ip
				if i.address != nil {
					for _, addr := range addrs {
						if addr.IPNet.String() == i.address.String() {
							i.dstName = ifaceName
							break
						}
						continue
					}
					if i.dstName == ifaceName {
						break
					}
				}
				// This is to find the interface name of the pair in overlay sandbox
				if strings.HasPrefix(ifaceName, "veth") {
					if i.master != "" && i.dstName == "veth" {
						i.dstName = ifaceName
					}
				}
			}

			var index int
			indexStr := strings.TrimPrefix(i.dstName, dstPrefix)
			if indexStr != "" {
				index, err = strconv.Atoi(indexStr)
				if err != nil {
					return err
				}
			}
			index++
			n.Lock()
			if index > n.nextIfIndex[dstPrefix] {
				n.nextIfIndex[dstPrefix] = index
			}
			n.iFaces = append(n.iFaces, i)
			n.Unlock()
		}
	}

	// restore routes
	for _, r := range routes {
		n.Lock()
		n.staticRoutes = append(n.staticRoutes, r)
		n.Unlock()
	}

	// restore gateway
	if len(gw) > 0 {
		n.Lock()
		n.gw = gw
		n.Unlock()
	}

	if len(gw6) > 0 {
		n.Lock()
		n.gwv6 = gw6
		n.Unlock()
	}

	return nil
}

// Checks whether IPv6 needs to be enabled/disabled on the loopback interface
func (n *networkNamespace) checkLoV6() {
	var (
		enable = false
		action = "disable"
	)

	n.Lock()
	for _, iface := range n.iFaces {
		if iface.AddressIPv6() != nil {
			enable = true
			action = "enable"
			break
		}
	}
	n.Unlock()

	if n.loV6Enabled == enable {
		return
	}

	if err := setIPv6(n.path, "lo", enable); err != nil {
		logrus.Warnf("Failed to %s IPv6 on loopback interface on network namespace %q: %v", action, n.path, err)
	}

	n.loV6Enabled = enable
}

func reexecSetIPv6() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if len(os.Args) < 3 {
		logrus.Errorf("invalid number of arguments for %s", os.Args[0])
		os.Exit(1)
	}

	ns, err := netns.GetFromPath(os.Args[1])
	if err != nil {
		logrus.Errorf("failed get network namespace %q: %v", os.Args[1], err)
		os.Exit(2)
	}
	defer ns.Close()

	if err = netns.Set(ns); err != nil {
		logrus.Errorf("setting into container netns %q failed: %v", os.Args[1], err)
		os.Exit(3)
	}

	var (
		action = "disable"
		value  = byte('1')
		path   = fmt.Sprintf("/proc/sys/net/ipv6/conf/%s/disable_ipv6", os.Args[2])
	)

	if os.Args[3] == "true" {
		action = "enable"
		value = byte('0')
	}

	if err = ioutil.WriteFile(path, []byte{value, '\n'}, 0644); err != nil {
		logrus.Errorf("failed to %s IPv6 forwarding for container's interface %s: %v", action, os.Args[2], err)
		os.Exit(4)
	}

	os.Exit(0)
}

func setIPv6(path, iface string, enable bool) error {
	cmd := &exec.Cmd{
		Path:   reexec.Self(),
		Args:   append([]string{"set-ipv6"}, path, iface, strconv.FormatBool(enable)),
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("reexec to set IPv6 failed: %v", err)
	}
	return nil
}
