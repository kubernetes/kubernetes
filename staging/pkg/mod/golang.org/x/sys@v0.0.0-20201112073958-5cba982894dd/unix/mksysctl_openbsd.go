// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Parse the header files for OpenBSD and generate a Go usable sysctl MIB.
//
// Build a MIB with each entry being an array containing the level, type and
// a hash that will contain additional entries if the current entry is a node.
// We then walk this MIB and create a flattened sysctl name to OID hash.

package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var (
	goos, goarch string
)

// cmdLine returns this programs's commandline arguments.
func cmdLine() string {
	return "go run mksysctl_openbsd.go " + strings.Join(os.Args[1:], " ")
}

// buildTags returns build tags.
func buildTags() string {
	return fmt.Sprintf("%s,%s", goarch, goos)
}

// reMatch performs regular expression match and stores the substring slice to value pointed by m.
func reMatch(re *regexp.Regexp, str string, m *[]string) bool {
	*m = re.FindStringSubmatch(str)
	if *m != nil {
		return true
	}
	return false
}

type nodeElement struct {
	n  int
	t  string
	pE *map[string]nodeElement
}

var (
	debugEnabled bool
	mib          map[string]nodeElement
	node         *map[string]nodeElement
	nodeMap      map[string]string
	sysCtl       []string
)

var (
	ctlNames1RE = regexp.MustCompile(`^#define\s+(CTL_NAMES)\s+{`)
	ctlNames2RE = regexp.MustCompile(`^#define\s+(CTL_(.*)_NAMES)\s+{`)
	ctlNames3RE = regexp.MustCompile(`^#define\s+((.*)CTL_NAMES)\s+{`)
	netInetRE   = regexp.MustCompile(`^netinet/`)
	netInet6RE  = regexp.MustCompile(`^netinet6/`)
	netRE       = regexp.MustCompile(`^net/`)
	bracesRE    = regexp.MustCompile(`{.*}`)
	ctlTypeRE   = regexp.MustCompile(`{\s+"(\w+)",\s+(CTLTYPE_[A-Z]+)\s+}`)
	fsNetKernRE = regexp.MustCompile(`^(fs|net|kern)_`)
)

func debug(s string) {
	if debugEnabled {
		fmt.Fprintln(os.Stderr, s)
	}
}

// Walk the MIB and build a sysctl name to OID mapping.
func buildSysctl(pNode *map[string]nodeElement, name string, oid []int) {
	lNode := pNode // local copy of pointer to node
	var keys []string
	for k := range *lNode {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		nodename := name
		if name != "" {
			nodename += "."
		}
		nodename += key

		nodeoid := append(oid, (*pNode)[key].n)

		if (*pNode)[key].t == `CTLTYPE_NODE` {
			if _, ok := nodeMap[nodename]; ok {
				lNode = &mib
				ctlName := nodeMap[nodename]
				for _, part := range strings.Split(ctlName, ".") {
					lNode = ((*lNode)[part]).pE
				}
			} else {
				lNode = (*pNode)[key].pE
			}
			buildSysctl(lNode, nodename, nodeoid)
		} else if (*pNode)[key].t != "" {
			oidStr := []string{}
			for j := range nodeoid {
				oidStr = append(oidStr, fmt.Sprintf("%d", nodeoid[j]))
			}
			text := "\t{ \"" + nodename + "\", []_C_int{ " + strings.Join(oidStr, ", ") + " } }, \n"
			sysCtl = append(sysCtl, text)
		}
	}
}

func main() {
	// Get the OS (using GOOS_TARGET if it exist)
	goos = os.Getenv("GOOS_TARGET")
	if goos == "" {
		goos = os.Getenv("GOOS")
	}
	// Get the architecture (using GOARCH_TARGET if it exists)
	goarch = os.Getenv("GOARCH_TARGET")
	if goarch == "" {
		goarch = os.Getenv("GOARCH")
	}
	// Check if GOOS and GOARCH environment variables are defined
	if goarch == "" || goos == "" {
		fmt.Fprintf(os.Stderr, "GOARCH or GOOS not defined in environment\n")
		os.Exit(1)
	}

	mib = make(map[string]nodeElement)
	headers := [...]string{
		`sys/sysctl.h`,
		`sys/socket.h`,
		`sys/tty.h`,
		`sys/malloc.h`,
		`sys/mount.h`,
		`sys/namei.h`,
		`sys/sem.h`,
		`sys/shm.h`,
		`sys/vmmeter.h`,
		`uvm/uvmexp.h`,
		`uvm/uvm_param.h`,
		`uvm/uvm_swap_encrypt.h`,
		`ddb/db_var.h`,
		`net/if.h`,
		`net/if_pfsync.h`,
		`net/pipex.h`,
		`netinet/in.h`,
		`netinet/icmp_var.h`,
		`netinet/igmp_var.h`,
		`netinet/ip_ah.h`,
		`netinet/ip_carp.h`,
		`netinet/ip_divert.h`,
		`netinet/ip_esp.h`,
		`netinet/ip_ether.h`,
		`netinet/ip_gre.h`,
		`netinet/ip_ipcomp.h`,
		`netinet/ip_ipip.h`,
		`netinet/tcp_var.h`,
		`netinet/udp_var.h`,
		`netinet6/in6.h`,
		`netinet6/ip6_divert.h`,
		`netinet/icmp6.h`,
		`netmpls/mpls.h`,
	}

	ctls := [...]string{
		`kern`,
		`vm`,
		`fs`,
		`net`,
		//debug			/* Special handling required */
		`hw`,
		//machdep		/* Arch specific */
		`user`,
		`ddb`,
		//vfs			/* Special handling required */
		`fs.posix`,
		`kern.forkstat`,
		`kern.intrcnt`,
		`kern.malloc`,
		`kern.nchstats`,
		`kern.seminfo`,
		`kern.shminfo`,
		`kern.timecounter`,
		`kern.tty`,
		`kern.watchdog`,
		`net.bpf`,
		`net.ifq`,
		`net.inet`,
		`net.inet.ah`,
		`net.inet.carp`,
		`net.inet.divert`,
		`net.inet.esp`,
		`net.inet.etherip`,
		`net.inet.gre`,
		`net.inet.icmp`,
		`net.inet.igmp`,
		`net.inet.ip`,
		`net.inet.ip.ifq`,
		`net.inet.ipcomp`,
		`net.inet.ipip`,
		`net.inet.mobileip`,
		`net.inet.pfsync`,
		`net.inet.tcp`,
		`net.inet.udp`,
		`net.inet6`,
		`net.inet6.divert`,
		`net.inet6.ip6`,
		`net.inet6.icmp6`,
		`net.inet6.tcp6`,
		`net.inet6.udp6`,
		`net.mpls`,
		`net.mpls.ifq`,
		`net.key`,
		`net.pflow`,
		`net.pfsync`,
		`net.pipex`,
		`net.rt`,
		`vm.swapencrypt`,
		//vfsgenctl		/* Special handling required */
	}

	// Node name "fixups"
	ctlMap := map[string]string{
		"ipproto":             "net.inet",
		"net.inet.ipproto":    "net.inet",
		"net.inet6.ipv6proto": "net.inet6",
		"net.inet6.ipv6":      "net.inet6.ip6",
		"net.inet.icmpv6":     "net.inet6.icmp6",
		"net.inet6.divert6":   "net.inet6.divert",
		"net.inet6.tcp6":      "net.inet.tcp",
		"net.inet6.udp6":      "net.inet.udp",
		"mpls":                "net.mpls",
		"swpenc":              "vm.swapencrypt",
	}

	// Node mappings
	nodeMap = map[string]string{
		"net.inet.ip.ifq": "net.ifq",
		"net.inet.pfsync": "net.pfsync",
		"net.mpls.ifq":    "net.ifq",
	}

	mCtls := make(map[string]bool)
	for _, ctl := range ctls {
		mCtls[ctl] = true
	}

	for _, header := range headers {
		debug("Processing " + header)
		file, err := os.Open(filepath.Join("/usr/include", header))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		s := bufio.NewScanner(file)
		for s.Scan() {
			var sub []string
			if reMatch(ctlNames1RE, s.Text(), &sub) ||
				reMatch(ctlNames2RE, s.Text(), &sub) ||
				reMatch(ctlNames3RE, s.Text(), &sub) {
				if sub[1] == `CTL_NAMES` {
					// Top level.
					node = &mib
				} else {
					// Node.
					nodename := strings.ToLower(sub[2])
					ctlName := ""
					if reMatch(netInetRE, header, &sub) {
						ctlName = "net.inet." + nodename
					} else if reMatch(netInet6RE, header, &sub) {
						ctlName = "net.inet6." + nodename
					} else if reMatch(netRE, header, &sub) {
						ctlName = "net." + nodename
					} else {
						ctlName = nodename
						ctlName = fsNetKernRE.ReplaceAllString(ctlName, `$1.`)
					}

					if val, ok := ctlMap[ctlName]; ok {
						ctlName = val
					}
					if _, ok := mCtls[ctlName]; !ok {
						debug("Ignoring " + ctlName + "...")
						continue
					}

					// Walk down from the top of the MIB.
					node = &mib
					for _, part := range strings.Split(ctlName, ".") {
						if _, ok := (*node)[part]; !ok {
							debug("Missing node " + part)
							(*node)[part] = nodeElement{n: 0, t: "", pE: &map[string]nodeElement{}}
						}
						node = (*node)[part].pE
					}
				}

				// Populate current node with entries.
				i := -1
				for !strings.HasPrefix(s.Text(), "}") {
					s.Scan()
					if reMatch(bracesRE, s.Text(), &sub) {
						i++
					}
					if !reMatch(ctlTypeRE, s.Text(), &sub) {
						continue
					}
					(*node)[sub[1]] = nodeElement{n: i, t: sub[2], pE: &map[string]nodeElement{}}
				}
			}
		}
		err = s.Err()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		file.Close()
	}
	buildSysctl(&mib, "", []int{})

	sort.Strings(sysCtl)
	text := strings.Join(sysCtl, "")

	fmt.Printf(srcTemplate, cmdLine(), buildTags(), text)
}

const srcTemplate = `// %s
// Code generated by the command above; DO NOT EDIT.

// +build %s

package unix

type mibentry struct {
	ctlname string
	ctloid []_C_int
}

var sysctlMib = []mibentry {
%s
}
`
