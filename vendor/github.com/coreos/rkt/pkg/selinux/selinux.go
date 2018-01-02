// Copyright 2016 The rkt Authors
// Copyright 2014,2015 Red Hat, Inc
// Copyright 2014,2015 Docker, Inc
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

// +build linux

package selinux

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"

	"github.com/coreos/rkt/pkg/fileutil"
)

const (
	Enforcing        = 1
	Permissive       = 0
	Disabled         = -1
	selinuxDir       = "/etc/selinux/"
	selinuxConfig    = selinuxDir + "config"
	selinuxTypeTag   = "SELINUXTYPE"
	selinuxTag       = "SELINUX"
	selinuxPath      = "/sys/fs/selinux"
	xattrNameSelinux = "security.selinux"
	stRdOnly         = 0x01
)

var (
	assignRegex           = regexp.MustCompile(`^([^=]+)=(.*)$`)
	spaceRegex            = regexp.MustCompile(`^([^=]+) (.*)$`)
	selinuxfs             = "unknown"
	selinuxEnabled        = false // Stores whether selinux is currently enabled
	selinuxEnabledChecked = false // Stores whether selinux enablement has been checked or established yet
	mcsdir                = ""    // Directory to use for MCS storage

	errMCSAlreadyExists = fmt.Errorf("MCS label already exists")
)

type SELinuxContext map[string]string

// SetDisabled disables selinux support for the package
func SetDisabled() {
	selinuxEnabled, selinuxEnabledChecked = false, true
}

// getSelinuxMountPoint returns the path to the mountpoint of an selinuxfs
// filesystem or an empty string if no mountpoint is found.  Selinuxfs is
// a proc-like pseudo-filesystem that exposes the selinux policy API to
// processes.  The existence of an selinuxfs mount is used to determine
// whether selinux is currently enabled or not.
func getSelinuxMountPoint() string {
	if _, err := os.Stat(selinuxPath); os.IsNotExist(err) {
		return ""
	}
	return selinuxPath
}

// SelinuxEnabled returns whether selinux is currently enabled.
func SelinuxEnabled() bool {
	if selinuxEnabledChecked {
		return selinuxEnabled
	}
	selinuxEnabledChecked = true
	if fs := getSelinuxMountPoint(); fs != "" {
		if con, _ := Getcon(); con != "kernel" {
			selinuxEnabled = true
		}
	}
	return selinuxEnabled
}

func readConfig(target string) (value string) {
	var (
		val, key string
		bufin    *bufio.Reader
	)

	in, err := os.Open(selinuxConfig)
	if err != nil {
		return ""
	}
	defer in.Close()

	bufin = bufio.NewReader(in)

	for done := false; !done; {
		var line string
		if line, err = bufin.ReadString('\n'); err != nil {
			if err != io.EOF {
				return ""
			}
			done = true
		}
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			// Skip blank lines
			continue
		}
		if line[0] == ';' || line[0] == '#' {
			// Skip comments
			continue
		}
		if groups := assignRegex.FindStringSubmatch(line); groups != nil {
			key, val = strings.TrimSpace(groups[1]), strings.TrimSpace(groups[2])
			if key == target {
				return strings.Trim(val, "\"")
			}
		}
	}
	return ""
}

func getSELinuxPolicyRoot() string {
	return selinuxDir + readConfig(selinuxTypeTag)
}

func readCon(name string) (string, error) {
	var val string

	in, err := os.Open(name)
	if err != nil {
		return "", err
	}
	defer in.Close()

	_, err = fmt.Fscanf(in, "%s", &val)
	return val, err
}

type SelinuxError struct {
	Errno int
	Prob  string
}

const (
	InvalidContext = iota
)

func (e *SelinuxError) Error() string {
	return fmt.Sprintf("SELinux: %s", e.Prob)
}

// Setfilecon sets the SELinux label for this path or returns an error.
func Setfilecon(path string, scon string) error {
	err := fileutil.Lsetxattr(path, xattrNameSelinux, []byte(scon), 0)
	if err != syscall.EINVAL {
		return err
	}
	return &SelinuxError{InvalidContext, "Invalid Context"}
}

// Getfilecon returns the SELinux label for this path or returns an error.
func Getfilecon(path string) (string, error) {
	con, err := fileutil.Lgetxattr(path, xattrNameSelinux)

	// Trim the NUL byte at the end of the byte buffer, if present.
	if con[len(con)-1] == '\x00' {
		con = con[:len(con)-1]
	}
	return string(con), err
}

func Setfscreatecon(scon string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/fscreate", syscall.Gettid()), scon)
}

func Getfscreatecon() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/fscreate", syscall.Gettid()))
}

// Getcon returns the SELinux label of the current process thread, or an error.
func Getcon() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/current", syscall.Gettid()))
}

// Getpidcon returns the SELinux label of the given pid, or an error.
func Getpidcon(pid int) (string, error) {
	return readCon(fmt.Sprintf("/proc/%d/attr/current", pid))
}

func Getexeccon() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/exec", syscall.Gettid()))
}

func writeCon(name string, val string) error {
	out, err := os.OpenFile(name, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer out.Close()

	if val != "" {
		_, err = out.Write([]byte(val))
	} else {
		_, err = out.Write(nil)
	}
	return err
}

func Setexeccon(scon string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/exec", syscall.Gettid()), scon)
}

func (c SELinuxContext) Get() string {
	return fmt.Sprintf("%s:%s:%s:%s", c["user"], c["role"], c["type"], c["level"])
}

func NewContext(scon string) SELinuxContext {
	c := make(SELinuxContext)

	if len(scon) != 0 {
		con := strings.SplitN(scon, ":", 4)
		c["user"] = con[0]
		c["role"] = con[1]
		c["type"] = con[2]
		c["level"] = con[3]
	}
	return c
}

func ReserveLabel(scon string) {
	if len(scon) != 0 {
		con := strings.SplitN(scon, ":", 4)
		mcsAdd(con[3])
	}
}

func SelinuxGetEnforce() int {
	var enforce int

	enforceS, err := readCon(fmt.Sprintf("%s/enforce", selinuxPath))
	if err != nil {
		return -1
	}

	enforce, err = strconv.Atoi(string(enforceS))
	if err != nil {
		return -1
	}
	return enforce
}

func SelinuxGetEnforceMode() int {
	switch readConfig(selinuxTag) {
	case "enforcing":
		return Enforcing
	case "permissive":
		return Permissive
	}
	return Disabled
}

func mcsPath(mcs string) string {
	filename := fmt.Sprintf("%s/%s", mcsdir, mcs)
	return filename
}

func mcsAdd(mcs string) error {
	filename := mcsPath(mcs)
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_EXCL|os.O_RDONLY, 0644)
	if err != nil {
		if os.IsExist(err) {
			return errMCSAlreadyExists
		}
		return fmt.Errorf("unable to create MCS: %v", err)
	}
	file.Close()
	return nil
}

func mcsDelete(mcs string) {
	filename := mcsPath(mcs)
	os.Remove(filename)
}

func mcsExists(mcs string) bool {
	filename := mcsPath(mcs)
	_, err := os.Stat(filename)
	if err == nil {
		return true
	}
	return false
}

func IntToMcs(id int, catRange uint32) string {
	var (
		SETSIZE = int(catRange)
		TIER    = SETSIZE
		ORD     = id
	)

	if id < 1 || id > 523776 {
		return ""
	}

	for ORD > TIER {
		ORD = ORD - TIER
		TIER -= 1
	}
	TIER = SETSIZE - TIER
	ORD = ORD + TIER
	return fmt.Sprintf("s0:c%d,c%d", TIER, ORD)
}

func makeUniqueMcs(catRange uint32) (string, error) {
	var (
		n      uint32
		c1, c2 uint32
		mcs    string
		tries  = 1000000
		err    error
	)

	for i := 0; i < tries; i++ {
		binary.Read(rand.Reader, binary.LittleEndian, &n)
		c1 = n % catRange
		binary.Read(rand.Reader, binary.LittleEndian, &n)
		c2 = n % catRange
		if c1 == c2 {
			continue
		}
		if c1 > c2 {
			c1, c2 = c2, c1
		}
		mcs = fmt.Sprintf("s0:c%d,c%d", c1, c2)
		err = mcsAdd(mcs)
		if err == nil {
			return mcs, nil
		}
	}
	return "", fmt.Errorf("couldn't generate unique MCS after %d tries! (last err=%v)", tries, err)
}

func FreeLxcContexts(scon string) {
	if len(scon) != 0 {
		con := strings.SplitN(scon, ":", 4)
		mcsDelete(con[3])
	}
}

func GetLxcContexts() (processLabel string, fileLabel string, err error) {
	var (
		val, key string
		bufin    *bufio.Reader
	)

	if !SelinuxEnabled() {
		return "", "", nil
	}
	lxcPath := fmt.Sprintf("%s/contexts/lxc_contexts", getSELinuxPolicyRoot())
	in, err := os.Open(lxcPath)
	if err != nil {
		return "", "", nil
	}
	defer in.Close()

	bufin = bufio.NewReader(in)

	for done := false; !done; {
		var line string
		if line, err = bufin.ReadString('\n'); err != nil {
			if err == io.EOF {
				done = true
			} else {
				goto exit
			}
		}
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			// Skip blank lines
			continue
		}
		if line[0] == ';' || line[0] == '#' {
			// Skip comments
			continue
		}
		if groups := assignRegex.FindStringSubmatch(line); groups != nil {
			key, val = strings.TrimSpace(groups[1]), strings.TrimSpace(groups[2])
			if key == "process" {
				processLabel = strings.Trim(val, "\"")
			}
			if key == "file" {
				fileLabel = strings.Trim(val, "\"")
			}
		}
	}

	if processLabel == "" || fileLabel == "" {
		return "", "", nil
	}

exit:
	mcs, err := makeUniqueMcs(1024)
	if err != nil {
		return "", "", err
	}
	scon := NewContext(processLabel)
	scon["level"] = mcs
	processLabel = scon.Get()
	scon = NewContext(fileLabel)
	scon["level"] = mcs
	fileLabel = scon.Get()
	return processLabel, fileLabel, nil
}

func SecurityCheckContext(val string) error {
	return writeCon(fmt.Sprintf("%s.context", selinuxPath), val)
}

func CopyLevel(src, dest string) (string, error) {
	if src == "" {
		return "", nil
	}
	if err := SecurityCheckContext(src); err != nil {
		return "", err
	}
	if err := SecurityCheckContext(dest); err != nil {
		return "", err
	}
	scon := NewContext(src)
	tcon := NewContext(dest)
	mcsDelete(tcon["level"])
	mcsAdd(scon["level"])
	tcon["level"] = scon["level"]
	return tcon.Get(), nil
}

// Prevent users from relabeling system files
func badPrefix(fpath string) error {
	var badprefixes = []string{"/usr"}

	for _, prefix := range badprefixes {
		if fpath == prefix || strings.HasPrefix(fpath, fmt.Sprintf("%s/", prefix)) {
			return fmt.Errorf("Relabeling content in %s is not allowed.", prefix)
		}
	}
	return nil
}

// Change the fpath file object to the SELinux label scon.
// If the fpath is a directory and recurse is true Chcon will walk the
// directory tree setting the label
func Chcon(fpath string, scon string, recurse bool) error {
	if scon == "" {
		return nil
	}
	if err := badPrefix(fpath); err != nil {
		return err
	}
	callback := func(p string, info os.FileInfo, err error) error {
		return Setfilecon(p, scon)
	}

	if recurse {
		return filepath.Walk(fpath, callback)
	}

	return Setfilecon(fpath, scon)
}

// DupSecOpt takes an SELinux process label and returns security options that
// can will set the SELinux Type and Level for future container processes
func DupSecOpt(src string) []string {
	if src == "" {
		return nil
	}
	con := NewContext(src)
	if con["user"] == "" ||
		con["role"] == "" ||
		con["type"] == "" ||
		con["level"] == "" {
		return nil
	}
	return []string{"label:user:" + con["user"],
		"label:role:" + con["role"],
		"label:type:" + con["type"],
		"label:level:" + con["level"]}
}

// DisableSecOpt returns a security opt that can be used to disabling SELinux
// labeling support for future container processes
func DisableSecOpt() []string {
	return []string{"label:disable"}
}

// Set the directory used for storage of used MCS contexts
func SetMCSDir(arg string) error {
	mcsdir = arg
	return os.MkdirAll(mcsdir, 0755)
}
