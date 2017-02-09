// +build linux

/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bandwidth

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"fmt"
	"net"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
)

// tcShaper provides an implementation of the BandwidthShaper interface on Linux using the 'tc' tool.
// In general, using this requires that the caller posses the NET_CAP_ADMIN capability, though if you
// do this within an container, it only requires the NS_CAPABLE capability for manipulations to that
// container's network namespace.
// Uses the hierarchical token bucket queuing discipline (htb), this requires Linux 2.4.20 or newer
// or a custom kernel with that queuing discipline backported.
type tcShaper struct {
	e     exec.Interface
	iface string
}

func NewTCShaper(iface string) BandwidthShaper {
	shaper := &tcShaper{
		e:     exec.New(),
		iface: iface,
	}
	return shaper
}

func (t *tcShaper) execAndLog(cmdStr string, args ...string) error {
	glog.V(6).Infof("Running: %s %s", cmdStr, strings.Join(args, " "))
	cmd := t.e.Command(cmdStr, args...)
	out, err := cmd.CombinedOutput()
	glog.V(6).Infof("Output from tc: %s", string(out))
	return err
}

func (t *tcShaper) nextClassID() (int, error) {
	data, err := t.e.Command("tc", "class", "show", "dev", t.iface).CombinedOutput()
	if err != nil {
		return -1, err
	}

	scanner := bufio.NewScanner(bytes.NewBuffer(data))
	classes := sets.String{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// skip empty lines
		if len(line) == 0 {
			continue
		}
		parts := strings.Split(line, " ")
		// expected tc line:
		// class htb 1:1 root prio 0 rate 1000Kbit ceil 1000Kbit burst 1600b cburst 1600b
		if len(parts) != 14 {
			return -1, fmt.Errorf("unexpected output from tc: %s (%v)", scanner.Text(), parts)
		}
		classes.Insert(parts[2])
	}

	// Make sure it doesn't go forever
	for nextClass := 1; nextClass < 10000; nextClass++ {
		if !classes.Has(fmt.Sprintf("1:%d", nextClass)) {
			return nextClass, nil
		}
	}
	// This should really never happen
	return -1, fmt.Errorf("exhausted class space, please try again")
}

// Convert a CIDR from text to a hex representation
// Strips any masked parts of the IP, so 1.2.3.4/16 becomes hex(1.2.0.0)/ffffffff
func hexCIDR(cidr string) (string, error) {
	ip, ipnet, err := net.ParseCIDR(cidr)
	if err != nil {
		return "", err
	}
	ip = ip.Mask(ipnet.Mask)
	hexIP := hex.EncodeToString([]byte(ip.To4()))
	hexMask := ipnet.Mask.String()
	return hexIP + "/" + hexMask, nil
}

// Convert a CIDR from hex representation to text, opposite of the above.
func asciiCIDR(cidr string) (string, error) {
	parts := strings.Split(cidr, "/")
	if len(parts) != 2 {
		return "", fmt.Errorf("unexpected CIDR format: %s", cidr)
	}
	ipData, err := hex.DecodeString(parts[0])
	if err != nil {
		return "", err
	}
	ip := net.IP(ipData)

	maskData, err := hex.DecodeString(parts[1])
	mask := net.IPMask(maskData)
	size, _ := mask.Size()

	return fmt.Sprintf("%s/%d", ip.String(), size), nil
}

func (t *tcShaper) findCIDRClass(cidr string) (class, handle string, found bool, err error) {
	data, err := t.e.Command("tc", "filter", "show", "dev", t.iface).CombinedOutput()
	if err != nil {
		return "", "", false, err
	}

	hex, err := hexCIDR(cidr)
	if err != nil {
		return "", "", false, err
	}
	spec := fmt.Sprintf("match %s", hex)

	scanner := bufio.NewScanner(bytes.NewBuffer(data))
	filter := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) == 0 {
			continue
		}
		if strings.HasPrefix(line, "filter") {
			filter = line
			continue
		}
		if strings.Contains(line, spec) {
			parts := strings.Split(filter, " ")
			// expected tc line:
			// filter parent 1: protocol ip pref 1 u32 fh 800::800 order 2048 key ht 800 bkt 0 flowid 1:1
			if len(parts) != 19 {
				return "", "", false, fmt.Errorf("unexpected output from tc: %s %d (%v)", filter, len(parts), parts)
			}
			return parts[18], parts[9], true, nil
		}
	}
	return "", "", false, nil
}

func makeKBitString(rsrc *resource.Quantity) string {
	return fmt.Sprintf("%dkbit", (rsrc.Value() / 1000))
}

func (t *tcShaper) makeNewClass(rate string) (int, error) {
	class, err := t.nextClassID()
	if err != nil {
		return -1, err
	}
	if err := t.execAndLog("tc", "class", "add",
		"dev", t.iface,
		"parent", "1:",
		"classid", fmt.Sprintf("1:%d", class),
		"htb", "rate", rate); err != nil {
		return -1, err
	}
	return class, nil
}

func (t *tcShaper) Limit(cidr string, upload, download *resource.Quantity) (err error) {
	var downloadClass, uploadClass int
	if download != nil {
		if downloadClass, err = t.makeNewClass(makeKBitString(download)); err != nil {
			return err
		}
		if err := t.execAndLog("tc", "filter", "add",
			"dev", t.iface,
			"protocol", "ip",
			"parent", "1:0",
			"prio", "1", "u32",
			"match", "ip", "dst", cidr,
			"flowid", fmt.Sprintf("1:%d", downloadClass)); err != nil {
			return err
		}
	}
	if upload != nil {
		if uploadClass, err = t.makeNewClass(makeKBitString(upload)); err != nil {
			return err
		}
		if err := t.execAndLog("tc", "filter", "add",
			"dev", t.iface,
			"protocol", "ip",
			"parent", "1:0",
			"prio", "1", "u32",
			"match", "ip", "src", cidr,
			"flowid", fmt.Sprintf("1:%d", uploadClass)); err != nil {
			return err
		}
	}
	return nil
}

// tests to see if an interface exists, if it does, return true and the status line for the interface
// returns false, "", <err> if an error occurs.
func (t *tcShaper) interfaceExists() (bool, string, error) {
	data, err := t.e.Command("tc", "qdisc", "show", "dev", t.iface).CombinedOutput()
	if err != nil {
		return false, "", err
	}
	value := strings.TrimSpace(string(data))
	if len(value) == 0 {
		return false, "", nil
	}
	// Newer versions of tc and/or the kernel return the following instead of nothing:
	// qdisc noqueue 0: root refcnt 2
	fields := strings.Fields(value)
	if len(fields) > 1 && fields[1] == "noqueue" {
		return false, "", nil
	}
	return true, value, nil
}

func (t *tcShaper) ReconcileCIDR(cidr string, upload, download *resource.Quantity) error {
	_, _, found, err := t.findCIDRClass(cidr)
	if err != nil {
		return err
	}
	if !found {
		return t.Limit(cidr, upload, download)
	}
	// TODO: actually check bandwidth limits here
	return nil
}

func (t *tcShaper) ReconcileInterface() error {
	exists, output, err := t.interfaceExists()
	if err != nil {
		return err
	}
	if !exists {
		glog.V(4).Info("Didn't find bandwidth interface, creating")
		return t.initializeInterface()
	}
	fields := strings.Split(output, " ")
	if len(fields) < 12 || fields[1] != "htb" || fields[2] != "1:" {
		if err := t.deleteInterface(fields[2]); err != nil {
			return err
		}
		return t.initializeInterface()
	}
	return nil
}

func (t *tcShaper) initializeInterface() error {
	return t.execAndLog("tc", "qdisc", "add", "dev", t.iface, "root", "handle", "1:", "htb", "default", "30")
}

func (t *tcShaper) Reset(cidr string) error {
	class, handle, found, err := t.findCIDRClass(cidr)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("Failed to find cidr: %s on interface: %s", cidr, t.iface)
	}
	if err := t.execAndLog("tc", "filter", "del",
		"dev", t.iface,
		"parent", "1:",
		"proto", "ip",
		"prio", "1",
		"handle", handle, "u32"); err != nil {
		return err
	}
	return t.execAndLog("tc", "class", "del", "dev", t.iface, "parent", "1:", "classid", class)
}

func (t *tcShaper) deleteInterface(class string) error {
	return t.execAndLog("tc", "qdisc", "delete", "dev", t.iface, "root", "handle", class)
}

func (t *tcShaper) GetCIDRs() ([]string, error) {
	data, err := t.e.Command("tc", "filter", "show", "dev", t.iface).CombinedOutput()
	if err != nil {
		return nil, err
	}

	result := []string{}
	scanner := bufio.NewScanner(bytes.NewBuffer(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) == 0 {
			continue
		}
		if strings.Contains(line, "match") {
			parts := strings.Split(line, " ")
			// expected tc line:
			// match <cidr> at <number>
			if len(parts) != 4 {
				return nil, fmt.Errorf("unexpected output: %v", parts)
			}
			cidr, err := asciiCIDR(parts[1])
			if err != nil {
				return nil, err
			}
			result = append(result, cidr)
		}
	}
	return result, nil
}
