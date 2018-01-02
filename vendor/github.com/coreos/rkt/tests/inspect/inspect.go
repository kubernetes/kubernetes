// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"bytes"
	"crypto/sha1"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/appc/spec/pkg/device"
	"github.com/coreos/rkt/common/cgroup"
	"github.com/coreos/rkt/common/cgroup/v1"
	"github.com/coreos/rkt/common/cgroup/v2"
	"github.com/coreos/rkt/tests/testutils"
	"github.com/syndtr/gocapability/capability"
)

var (
	globalFlagset = flag.NewFlagSet("inspect", flag.ExitOnError)
	globalFlags   = struct {
		ReadStdin          bool
		CheckTty           bool
		CheckPath          bool
		PrintExec          bool
		PrintMsg           string
		SuffixMsg          string
		PrintEnv           string
		PrintCapsPid       int
		PrintUser          bool
		PrintGroups        bool
		PrintCwd           bool
		ExitCode           int
		ReadFile           bool
		WriteFile          bool
		StatFile           bool
		HashFile           bool
		FileSymlinkTarget  bool
		Sleep              int
		PreSleep           int
		PrintMemoryLimit   bool
		PrintCPUQuota      bool
		PrintCPUShares     bool
		FileName           string
		Content            string
		CheckCgroupMounts  bool
		PrintNetNS         bool
		PrintIPv4          string
		PrintIPv6          string
		PrintDefaultGWv4   bool
		PrintDefaultGWv6   bool
		PrintGWv4          string
		PrintGWv6          string
		PrintHostname      bool
		GetHTTP            string
		ServeHTTP          string
		ServeHTTPTimeout   int
		PrintIfaceCount    bool
		PrintAppAnnotation string
		SilentSigterm      bool
		CheckMountNS       bool
		PrintNoNewPrivs    bool
		CheckMknod         string
	}{}
)

func init() {
	globalFlagset.BoolVar(&globalFlags.ReadStdin, "read-stdin", false, "Read a line from stdin")
	globalFlagset.BoolVar(&globalFlags.CheckTty, "check-tty", false, "Check if stdin is a terminal")
	globalFlagset.BoolVar(&globalFlags.CheckPath, "check-path", false, "Check if environment variable PATH does not contain \\n")
	globalFlagset.BoolVar(&globalFlags.PrintExec, "print-exec", false, "Print the command we were execed as (i.e. argv[0])")
	globalFlagset.StringVar(&globalFlags.PrintMsg, "print-msg", "", "Print the message given as parameter")
	globalFlagset.StringVar(&globalFlags.SuffixMsg, "suffix-msg", "", "Print this suffix after some commands")
	globalFlagset.BoolVar(&globalFlags.PrintCwd, "print-cwd", false, "Print the current working directory")
	globalFlagset.StringVar(&globalFlags.PrintEnv, "print-env", "", "Print the specified environment variable")
	globalFlagset.IntVar(&globalFlags.PrintCapsPid, "print-caps-pid", -1, "Print capabilities of the specified pid (or current process if pid=0)")
	globalFlagset.BoolVar(&globalFlags.PrintUser, "print-user", false, "Print uid and gid")
	globalFlagset.BoolVar(&globalFlags.PrintGroups, "print-groups", false, "Print all gids")
	globalFlagset.IntVar(&globalFlags.ExitCode, "exit-code", 0, "Return this exit code")
	globalFlagset.BoolVar(&globalFlags.ReadFile, "read-file", false, "Print the content of the file $FILE")
	globalFlagset.BoolVar(&globalFlags.WriteFile, "write-file", false, "Write $CONTENT in the file $FILE")
	globalFlagset.BoolVar(&globalFlags.StatFile, "stat-file", false, "Print the ownership and mode of the file $FILE")
	globalFlagset.BoolVar(&globalFlags.HashFile, "hash-file", false, "Print the SHA1SUM of the file $FILE")
	globalFlagset.BoolVar(&globalFlags.FileSymlinkTarget, "file-symlink-target", false, "Print the symlink target of $FILE")
	globalFlagset.IntVar(&globalFlags.Sleep, "sleep", -1, "Sleep before exiting (in seconds)")
	globalFlagset.IntVar(&globalFlags.PreSleep, "pre-sleep", -1, "Sleep before executing (in seconds)")
	globalFlagset.BoolVar(&globalFlags.PrintMemoryLimit, "print-memorylimit", false, "Print cgroup memory limit")
	globalFlagset.BoolVar(&globalFlags.PrintCPUQuota, "print-cpuquota", false, "Print cgroup cpu quota in milli-cores")
	globalFlagset.BoolVar(&globalFlags.PrintCPUShares, "print-cpushares", false, "Print cgroup cpu shares")
	globalFlagset.StringVar(&globalFlags.FileName, "file-name", "", "The file to read/write, $FILE will be ignored if this is specified")
	globalFlagset.StringVar(&globalFlags.Content, "content", "", "The content to write, $CONTENT will be ignored if this is specified")
	globalFlagset.BoolVar(&globalFlags.CheckCgroupMounts, "check-cgroups", false, "Try to write to the cgroup filesystem. Everything should be RO except some well-known files")
	globalFlagset.BoolVar(&globalFlags.PrintNetNS, "print-netns", false, "Print the network namespace")
	globalFlagset.StringVar(&globalFlags.PrintIPv4, "print-ipv4", "", "Takes an interface name and prints its IPv4")
	globalFlagset.StringVar(&globalFlags.PrintIPv6, "print-ipv6", "", "Takes an interface name and prints its IPv6")
	globalFlagset.BoolVar(&globalFlags.PrintDefaultGWv4, "print-defaultgwv4", false, "Print the default IPv4 gateway")
	globalFlagset.BoolVar(&globalFlags.PrintDefaultGWv6, "print-defaultgwv6", false, "Print the default IPv6 gateway")
	globalFlagset.StringVar(&globalFlags.PrintGWv4, "print-gwv4", "", "Takes an interface name and prints its gateway's IPv4")
	globalFlagset.StringVar(&globalFlags.PrintGWv6, "print-gwv6", "", "Takes an interface name and prints its gateway's IPv6")
	globalFlagset.BoolVar(&globalFlags.PrintHostname, "print-hostname", false, "Prints the pod hostname")
	globalFlagset.StringVar(&globalFlags.GetHTTP, "get-http", "", "HTTP-Get from the given address")
	globalFlagset.StringVar(&globalFlags.ServeHTTP, "serve-http", "", "Serve the hostname via HTTP on the given address:port")
	globalFlagset.IntVar(&globalFlags.ServeHTTPTimeout, "serve-http-timeout", 30, "HTTP Timeout to wait for a client connection")
	globalFlagset.BoolVar(&globalFlags.PrintIfaceCount, "print-iface-count", false, "Print the interface count")
	globalFlagset.StringVar(&globalFlags.PrintAppAnnotation, "print-app-annotation", "", "Take an annotation name of the app, and prints its value")
	globalFlagset.BoolVar(&globalFlags.SilentSigterm, "silent-sigterm", false, "Exit with a success exit status if we receive SIGTERM")
	globalFlagset.BoolVar(&globalFlags.CheckMountNS, "check-mountns", false, "Check if app's mount ns is different than stage1's. Requires CAP_SYS_PTRACE")
	globalFlagset.BoolVar(&globalFlags.PrintNoNewPrivs, "print-no-new-privs", false, "print the prctl PR_GET_NO_NEW_PRIVS value")
	globalFlagset.StringVar(&globalFlags.CheckMknod, "check-mknod", "", "check whether mknod on restricted devices is allowed")
}

func in(list []int, el int) bool {
	for _, x := range list {
		if el == x {
			return true
		}
	}
	return false
}

func main() {
	globalFlagset.Parse(os.Args[1:])
	args := globalFlagset.Args()
	if len(args) > 0 {
		fmt.Fprintln(os.Stderr, "Wrong parameters")
		os.Exit(254)
	}

	if globalFlags.SilentSigterm {
		terminateCh := make(chan os.Signal, 1)
		signal.Notify(terminateCh, syscall.SIGTERM)
		go func() {
			<-terminateCh
			os.Exit(0)
		}()
	}

	if globalFlags.PreSleep >= 0 {
		time.Sleep(time.Duration(globalFlags.PreSleep) * time.Second)
	}

	if globalFlags.ReadStdin {
		reader := bufio.NewReader(os.Stdin)
		fmt.Printf("Enter text:\n")
		text, _ := reader.ReadString('\n')
		fmt.Printf("Received text: %s\n", text)
	}

	if globalFlags.PrintNoNewPrivs {
		r1, _, err := syscall.Syscall(
			syscall.SYS_PRCTL,
			uintptr(unix.PR_GET_NO_NEW_PRIVS),
			uintptr(0), uintptr(0),
		)

		fmt.Printf("no_new_privs: %v err: %v\n", r1, err)
	}

	if globalFlags.CheckMknod != "" {
		/* format: c:5:2:name */
		dev := strings.SplitN(globalFlags.CheckMknod, ":", 4)
		if len(dev) < 4 {
			fmt.Fprintln(os.Stderr, "Not enough parameters for mknod")
			os.Exit(254)
		}
		typ := dev[0]
		major, err := strconv.Atoi(dev[1])
		if err != nil {
			fmt.Fprintln(os.Stderr, "Wrong major")
			os.Exit(254)
		}
		minor, err := strconv.Atoi(dev[2])
		if err != nil {
			fmt.Fprintln(os.Stderr, "Wrong minor")
			os.Exit(254)
		}
		nodeName := dev[3]

		majorMinor := device.Makedev(uint(major), uint(minor))
		mode := uint32(0777)
		switch typ {
		case "c":
			mode |= syscall.S_IFCHR
		case "b":
			mode |= syscall.S_IFBLK
		default:
			fmt.Fprintln(os.Stderr, "Wrong device node type")
			os.Exit(254)
		}

		if err := syscall.Mknod(nodeName, mode, int(majorMinor)); err != nil {
			fmt.Fprintf(os.Stderr, "mknod %s: fail: %v\n", nodeName, err)
			os.Exit(254)
		} else {
			fmt.Printf("mknod %s: succeed\n", nodeName)
			os.Exit(0)
		}
	}

	if globalFlags.CheckTty {
		fd := int(os.Stdin.Fd())
		var termios syscall.Termios
		_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, uintptr(fd), syscall.TCGETS, uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
		if err == 0 {
			fmt.Printf("stdin is a terminal\n")
		} else {
			fmt.Printf("stdin is not a terminal\n")
		}
	}

	if globalFlags.CheckPath {
		envBytes, err := ioutil.ReadFile("/proc/self/environ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading environment from \"/proc/self/environ\": %v\n", err)
			os.Exit(254)
		}
		for _, v := range bytes.Split(envBytes, []byte{0}) {
			if len(v) == 0 {
				continue
			}
			if strings.HasPrefix(string(v), "PATH=") {
				if strings.Contains(string(v), "\n") {
					fmt.Fprintf(os.Stderr, "Malformed PATH: found new line")
					os.Exit(254)
				} else {
					fmt.Printf("PATH is good\n")
					os.Exit(0)
				}
			} else {
				continue
			}
		}
		fmt.Fprintf(os.Stderr, "PATH not found")
		os.Exit(254)
	}

	if globalFlags.PrintExec {
		fmt.Fprintf(os.Stdout, "inspect execed as: %s\n", os.Args[0])
	}

	if globalFlags.PrintMsg != "" {
		fmt.Fprintf(os.Stdout, "%s\n", globalFlags.PrintMsg)
		messageLoopStr := os.Getenv("MESSAGE_LOOP")
		messageLoop, err := strconv.Atoi(messageLoopStr)
		if err == nil {
			for i := 0; i < messageLoop; i++ {
				time.Sleep(time.Second)
				fmt.Fprintf(os.Stdout, "%s\n", globalFlags.PrintMsg)
			}
		}
	}

	if globalFlags.PrintEnv != "" {
		fmt.Fprintf(os.Stdout, "%s=%s\n", globalFlags.PrintEnv, os.Getenv(globalFlags.PrintEnv))
	}

	if globalFlags.PrintCapsPid >= 0 {
		caps, err := capability.NewPid(globalFlags.PrintCapsPid)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot get caps: %v\n", err)
			os.Exit(254)
		}
		fmt.Printf("Capability set: effective: %s (%s)\n", caps.StringCap(capability.EFFECTIVE), globalFlags.SuffixMsg)
		fmt.Printf("Capability set: permitted: %s (%s)\n", caps.StringCap(capability.PERMITTED), globalFlags.SuffixMsg)
		fmt.Printf("Capability set: inheritable: %s (%s)\n", caps.StringCap(capability.INHERITABLE), globalFlags.SuffixMsg)
		fmt.Printf("Capability set: bounding: %s (%s)\n", caps.StringCap(capability.BOUNDING), globalFlags.SuffixMsg)

		if capStr := os.Getenv("CAPABILITY"); capStr != "" {
			capInt, err := strconv.Atoi(capStr)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Environment variable $CAPABILITY is not a valid capability number: %v\n", err)
				os.Exit(254)
			}
			c := capability.Cap(capInt)
			if caps.Get(capability.BOUNDING, c) {
				fmt.Printf("%v=enabled (%s)\n", c.String(), globalFlags.SuffixMsg)
			} else {
				fmt.Printf("%v=disabled (%s)\n", c.String(), globalFlags.SuffixMsg)
			}
		}
	}

	if globalFlags.PrintUser {
		fmt.Printf("User: uid=%d euid=%d gid=%d egid=%d\n", os.Getuid(), os.Geteuid(), os.Getgid(), os.Getegid())
	}

	if globalFlags.PrintGroups {
		gids, err := os.Getgroups()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting groups: %v\n", err)
			os.Exit(254)
		}
		// getgroups(2): It is unspecified whether the effective group ID of
		// the calling process is included in the returned list. (Thus, an
		// application should also call getegid(2) and add or remove the
		// resulting value.)
		egid := os.Getegid()
		if !in(gids, egid) {
			gids = append(gids, egid)
			sort.Ints(gids)
		}
		var b bytes.Buffer
		for _, gid := range gids {
			b.WriteString(fmt.Sprintf("%d ", gid))
		}
		fmt.Printf("Groups: %s\n", b.String())
	}

	if globalFlags.WriteFile {
		fileName := os.Getenv("FILE")
		if globalFlags.FileName != "" {
			fileName = globalFlags.FileName
		}
		content := os.Getenv("CONTENT")
		if globalFlags.Content != "" {
			content = globalFlags.Content
		}

		err := ioutil.WriteFile(fileName, []byte(content), 0600)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot write to file %q: %v\n", fileName, err)
			os.Exit(254)
		}
	}

	if globalFlags.ReadFile {
		fileName := os.Getenv("FILE")
		if globalFlags.FileName != "" {
			fileName = globalFlags.FileName
		}

		dat, err := ioutil.ReadFile(fileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot read file %q: %v\n", fileName, err)
			os.Exit(254)
		}
		fmt.Print("<<<")
		fmt.Print(string(dat))
		fmt.Print(">>>\n")
	}

	if globalFlags.StatFile {
		fileName := os.Getenv("FILE")
		if globalFlags.FileName != "" {
			fileName = globalFlags.FileName
		}

		fi, err := os.Stat(fileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot stat file %q: %v\n", fileName, err)
			os.Exit(254)
		}
		fmt.Printf("%s: mode: %s\n", fileName, fi.Mode().String())
		fmt.Printf("%s: user: %v\n", fileName, fi.Sys().(*syscall.Stat_t).Uid)
		fmt.Printf("%s: group: %v\n", fileName, fi.Sys().(*syscall.Stat_t).Gid)
	}

	if globalFlags.HashFile {
		fileName := os.Getenv("FILE")
		if globalFlags.FileName != "" {
			fileName = globalFlags.FileName
		}

		dat, err := ioutil.ReadFile(fileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot read file %q: %v\n", fileName, err)
			os.Exit(254)
		}

		fmt.Printf("sha1sum: %x\n", sha1.Sum(dat))
	}

	if globalFlags.FileSymlinkTarget {
		fileName := os.Getenv("FILE")
		if globalFlags.FileName != "" {
			fileName = globalFlags.FileName
		}

		dst, err := os.Readlink(fileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot read file target %q: %v\n", fileName, err)
			os.Exit(1)
		}

		fmt.Printf("symlink: %q -> %q\n", fileName, dst)
	}

	if globalFlags.PrintCwd {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot get working directory: %v\n", err)
			os.Exit(254)
		}
		fmt.Printf("cwd: %s\n", wd)
	}

	if globalFlags.PrintMemoryLimit {
		// we use /proc/1/root to escape the chroot we're in and read the file
		isUnified, err := cgroup.IsCgroupUnified("/proc/1/root/")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting cgroup type: %v\n", err)
			os.Exit(254)
		}

		var limitPath string
		if isUnified {
			cgroupPath, err := v2.GetOwnCgroupPath()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error getting own memory cgroup path: %v\n", err)
				os.Exit(254)
			}
			limitPath = filepath.Join("/proc/1/root/sys/fs/cgroup/", cgroupPath, "memory.max")
			fmt.Fprintln(os.Stderr, "limitPath:", limitPath)
		} else {
			memCgroupPath, err := v1.GetOwnCgroupPath("memory")
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error getting own memory cgroup path: %v\n", err)
				os.Exit(254)
			}
			limitPath = filepath.Join("/proc/1/root/sys/fs/cgroup/memory", memCgroupPath, "memory.limit_in_bytes")
			fmt.Fprintln(os.Stderr, limitPath)
		}
		limit, err := ioutil.ReadFile(limitPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Can't read %s\n", limitPath)
			os.Exit(254)
		}

		fmt.Printf("Memory Limit: %s\n", string(limit))
	}

	if globalFlags.PrintCPUQuota {
		cpuCgroupPath, err := v1.GetOwnCgroupPath("cpu")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting own cpu cgroup path: %v\n", err)
			os.Exit(254)
		}
		// we use /proc/1/root to escape the chroot we're in and read our
		// cpu quota
		periodPath := filepath.Join("/proc/1/root/sys/fs/cgroup/cpu", cpuCgroupPath, "cpu.cfs_period_us")
		periodBytes, err := ioutil.ReadFile(periodPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Can't read cpu.cpu_period_us\n")
			os.Exit(254)
		}
		quotaPath := filepath.Join("/proc/1/root/sys/fs/cgroup/cpu", cpuCgroupPath, "cpu.cfs_quota_us")
		quotaBytes, err := ioutil.ReadFile(quotaPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Can't read cpu.cpu_quota_us\n")
			os.Exit(254)
		}

		period, err := strconv.Atoi(strings.Trim(string(periodBytes), "\n"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		quota, err := strconv.Atoi(strings.Trim(string(quotaBytes), "\n"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}

		quotaMilliCores := quota * 1000 / period
		fmt.Printf("CPU Quota: %s\n", strconv.Itoa(quotaMilliCores))
	}

	if globalFlags.PrintCPUShares {
		cpuCgroupPath, err := v1.GetOwnCgroupPath("cpu")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting own cpu cgroup path: %v\n", err)
			os.Exit(1)
		}
		// we use /proc/1/root to escape the chroot we're in and read our
		// cpu quota
		sharesPath := filepath.Join("/proc/1/root/sys/fs/cgroup/cpu", cpuCgroupPath, "cpu.shares")
		sharesBytes, err := ioutil.ReadFile(sharesPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Can't read cpu.shares\n")
			os.Exit(1)
		}

		fmt.Printf("CPU Shares: %s", string(sharesBytes))
	}

	if globalFlags.CheckCgroupMounts {
		rootCgroupPath := "/proc/1/root/sys/fs/cgroup"
		testPaths := []string{rootCgroupPath}

		// test a couple of controllers if they're available
		if _, err := os.Stat(filepath.Join(rootCgroupPath, "memory")); err == nil {
			testPaths = append(testPaths, filepath.Join(rootCgroupPath, "memory"))
		}
		if _, err := os.Stat(filepath.Join(rootCgroupPath, "cpu")); err == nil {
			testPaths = append(testPaths, filepath.Join(rootCgroupPath, "cpu"))
		}

		for _, p := range testPaths {
			if err := syscall.Mkdir(filepath.Join(p, "test"), 0600); err == nil || err != syscall.EROFS {
				fmt.Fprintf(os.Stderr, "check-cgroups: FAIL (%v)", err)
				os.Exit(254)
			}
		}

		fmt.Println("check-cgroups: SUCCESS")
	}

	if globalFlags.PrintNetNS {
		ns, err := os.Readlink("/proc/self/ns/net")
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		fmt.Printf("NetNS: %s\n", ns)
	}

	if globalFlags.PrintIPv4 != "" {
		iface := globalFlags.PrintIPv4
		ips, err := testutils.GetIPsv4(iface)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		if len(ips) == 0 {
			fmt.Fprintf(os.Stderr, "No IPv4 found for interface %+v:\n", iface)
			os.Exit(254)
		}
		fmt.Printf("%v IPv4: %s\n", iface, ips[0])
	}

	if globalFlags.PrintDefaultGWv4 {
		gw, err := testutils.GetDefaultGWv4()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("DefaultGWv4: %s\n", gw)
	}

	if globalFlags.PrintDefaultGWv6 {
		gw, err := testutils.GetDefaultGWv6()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("DefaultGWv6: %s\n", gw)
	}

	if globalFlags.PrintGWv4 != "" {
		// TODO: GetGW not implemented yet
		iface := globalFlags.PrintGWv4
		gw, err := testutils.GetGWv4(iface)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("%v GWv4: %s\n", iface, gw)
	}

	if globalFlags.PrintIPv6 != "" {
		// TODO
	}

	if globalFlags.PrintGWv6 != "" {
		// TODO
	}

	if globalFlags.PrintHostname {
		hostname, err := os.Hostname()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("Hostname: %s\n", hostname)
	}

	if globalFlags.ServeHTTP != "" {
		err := testutils.HTTPServe(globalFlags.ServeHTTP, globalFlags.ServeHTTPTimeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
	}

	if globalFlags.GetHTTP != "" {
		body, err := testutils.HTTPGet(globalFlags.GetHTTP)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("HTTP-Get received: %s\n", body)
	}

	if globalFlags.PrintIfaceCount {
		ifaceCount, err := testutils.GetIfaceCount()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("Interface count: %d\n", ifaceCount)
	}

	if globalFlags.PrintAppAnnotation != "" {
		mdsUrl, appName := os.Getenv("AC_METADATA_URL"), os.Getenv("AC_APP_NAME")
		body, err := testutils.HTTPGet(fmt.Sprintf("%s/acMetadata/v1/apps/%s/annotations/%s", mdsUrl, appName, globalFlags.PrintAppAnnotation))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		fmt.Printf("Annotation %s=%s\n", globalFlags.PrintAppAnnotation, body)
	}

	if globalFlags.CheckMountNS {
		appMountNS, err := os.Readlink("/proc/self/ns/mnt")
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		s1MountNS, err := os.Readlink("/proc/1/ns/mnt")
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(254)
		}
		if appMountNS != s1MountNS {
			fmt.Println("check-mountns: DIFFERENT")
		} else {
			fmt.Println("check-mountns: IDENTICAL")
			os.Exit(254)
		}
	}

	if globalFlags.Sleep >= 0 {
		time.Sleep(time.Duration(globalFlags.Sleep) * time.Second)
	}

	os.Exit(globalFlags.ExitCode)
}
