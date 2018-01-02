package ns

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netns"
)

var (
	initNs   netns.NsHandle
	initNl   *netlink.Handle
	initOnce sync.Once
	// NetlinkSocketsTimeout represents the default timeout duration for the sockets
	NetlinkSocketsTimeout = 3 * time.Second
)

// Init initializes a new network namespace
func Init() {
	var err error
	initNs, err = netns.Get()
	if err != nil {
		logrus.Errorf("could not get initial namespace: %v", err)
	}
	initNl, err = netlink.NewHandle(getSupportedNlFamilies()...)
	if err != nil {
		logrus.Errorf("could not create netlink handle on initial namespace: %v", err)
	}
	err = initNl.SetSocketTimeout(NetlinkSocketsTimeout)
	if err != nil {
		logrus.Warnf("Failed to set the timeout on the default netlink handle sockets: %v", err)
	}
}

// SetNamespace sets the initial namespace handler
func SetNamespace() error {
	initOnce.Do(Init)
	if err := netns.Set(initNs); err != nil {
		linkInfo, linkErr := getLink()
		if linkErr != nil {
			linkInfo = linkErr.Error()
		}
		return fmt.Errorf("failed to set to initial namespace, %v, initns fd %d: %v", linkInfo, initNs, err)
	}
	return nil
}

// ParseHandlerInt transforms the namespace handler into an integer
func ParseHandlerInt() int {
	return int(getHandler())
}

// GetHandler returns the namespace handler
func getHandler() netns.NsHandle {
	initOnce.Do(Init)
	return initNs
}

func getLink() (string, error) {
	return os.Readlink(fmt.Sprintf("/proc/%d/task/%d/ns/net", os.Getpid(), syscall.Gettid()))
}

// NlHandle returns the netlink handler
func NlHandle() *netlink.Handle {
	initOnce.Do(Init)
	return initNl
}

func getSupportedNlFamilies() []int {
	fams := []int{syscall.NETLINK_ROUTE}
	// NETLINK_XFRM test
	if err := loadXfrmModules(); err != nil {
		if checkXfrmSocket() != nil {
			logrus.Warnf("Could not load necessary modules for IPSEC rules: %v", err)
		} else {
			fams = append(fams, syscall.NETLINK_XFRM)
		}
	} else {
		fams = append(fams, syscall.NETLINK_XFRM)
	}
	// NETLINK_NETFILTER test
	if err := loadNfConntrackModules(); err != nil {
		if checkNfSocket() != nil {
			logrus.Warnf("Could not load necessary modules for Conntrack: %v", err)
		} else {
			fams = append(fams, syscall.NETLINK_NETFILTER)
		}
	} else {
		fams = append(fams, syscall.NETLINK_NETFILTER)
	}

	return fams
}

func loadXfrmModules() error {
	if out, err := exec.Command("modprobe", "-va", "xfrm_user").CombinedOutput(); err != nil {
		return fmt.Errorf("Running modprobe xfrm_user failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
	if out, err := exec.Command("modprobe", "-va", "xfrm_algo").CombinedOutput(); err != nil {
		return fmt.Errorf("Running modprobe xfrm_algo failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
	return nil
}

// API check on required xfrm modules (xfrm_user, xfrm_algo)
func checkXfrmSocket() error {
	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW, syscall.NETLINK_XFRM)
	if err != nil {
		return err
	}
	syscall.Close(fd)
	return nil
}

func loadNfConntrackModules() error {
	if out, err := exec.Command("modprobe", "-va", "nf_conntrack").CombinedOutput(); err != nil {
		return fmt.Errorf("Running modprobe nf_conntrack failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
	if out, err := exec.Command("modprobe", "-va", "nf_conntrack_netlink").CombinedOutput(); err != nil {
		return fmt.Errorf("Running modprobe nf_conntrack_netlink failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
	return nil
}

// API check on required nf_conntrack* modules (nf_conntrack, nf_conntrack_netlink)
func checkNfSocket() error {
	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW, syscall.NETLINK_NETFILTER)
	if err != nil {
		return err
	}
	syscall.Close(fd)
	return nil
}
