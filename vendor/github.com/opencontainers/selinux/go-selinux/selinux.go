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
	"sync"
	"syscall"
)

const (
	// Enforcing constant indicate SELinux is in enforcing mode
	Enforcing = 1
	// Permissive constant to indicate SELinux is in permissive mode
	Permissive = 0
	// Disabled constant to indicate SELinux is disabled
	Disabled         = -1
	selinuxDir       = "/etc/selinux/"
	selinuxConfig    = selinuxDir + "config"
	selinuxTypeTag   = "SELINUXTYPE"
	selinuxTag       = "SELINUX"
	selinuxPath      = "/sys/fs/selinux"
	xattrNameSelinux = "security.selinux"
	stRdOnly         = 0x01
)

type selinuxState struct {
	enabledSet   bool
	enabled      bool
	selinuxfsSet bool
	selinuxfs    string
	mcsList      map[string]bool
	sync.Mutex
}

var (
	assignRegex = regexp.MustCompile(`^([^=]+)=(.*)$`)
	state       = selinuxState{
		mcsList: make(map[string]bool),
	}
)

// Context is a representation of the SELinux label broken into 4 parts
type Context map[string]string

func (s *selinuxState) setEnable(enabled bool) bool {
	s.Lock()
	defer s.Unlock()
	s.enabledSet = true
	s.enabled = enabled
	return s.enabled
}

func (s *selinuxState) getEnabled() bool {
	s.Lock()
	enabled := s.enabled
	enabledSet := s.enabledSet
	s.Unlock()
	if enabledSet {
		return enabled
	}

	enabled = false
	if fs := getSelinuxMountPoint(); fs != "" {
		if con, _ := CurrentLabel(); con != "kernel" {
			enabled = true
		}
	}
	return s.setEnable(enabled)
}

// SetDisabled disables selinux support for the package
func SetDisabled() {
	state.setEnable(false)
}

func (s *selinuxState) setSELinuxfs(selinuxfs string) string {
	s.Lock()
	defer s.Unlock()
	s.selinuxfsSet = true
	s.selinuxfs = selinuxfs
	return s.selinuxfs
}

func (s *selinuxState) getSELinuxfs() string {
	s.Lock()
	selinuxfs := s.selinuxfs
	selinuxfsSet := s.selinuxfsSet
	s.Unlock()
	if selinuxfsSet {
		return selinuxfs
	}

	selinuxfs = ""
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return selinuxfs
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		txt := scanner.Text()
		// Safe as mountinfo encodes mountpoints with spaces as \040.
		sepIdx := strings.Index(txt, " - ")
		if sepIdx == -1 {
			continue
		}
		if !strings.Contains(txt[sepIdx:], "selinuxfs") {
			continue
		}
		fields := strings.Split(txt, " ")
		if len(fields) < 5 {
			continue
		}
		selinuxfs = fields[4]
		break
	}

	if selinuxfs != "" {
		var buf syscall.Statfs_t
		syscall.Statfs(selinuxfs, &buf)
		if (buf.Flags & stRdOnly) == 1 {
			selinuxfs = ""
		}
	}
	return s.setSELinuxfs(selinuxfs)
}

// getSelinuxMountPoint returns the path to the mountpoint of an selinuxfs
// filesystem or an empty string if no mountpoint is found.  Selinuxfs is
// a proc-like pseudo-filesystem that exposes the selinux policy API to
// processes.  The existence of an selinuxfs mount is used to determine
// whether selinux is currently enabled or not.
func getSelinuxMountPoint() string {
	return state.getSELinuxfs()
}

// GetEnabled returns whether selinux is currently enabled.
func GetEnabled() bool {
	return state.getEnabled()
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

// SetFileLabel sets the SELinux label for this path or returns an error.
func SetFileLabel(path string, label string) error {
	return lsetxattr(path, xattrNameSelinux, []byte(label), 0)
}

// FileLabel returns the SELinux label for this path or returns an error.
func FileLabel(path string) (string, error) {
	label, err := lgetxattr(path, xattrNameSelinux)
	if err != nil {
		return "", err
	}
	// Trim the NUL byte at the end of the byte buffer, if present.
	if len(label) > 0 && label[len(label)-1] == '\x00' {
		label = label[:len(label)-1]
	}
	return string(label), nil
}

/*
SetFSCreateLabel tells kernel the label to create all file system objects
created by this task. Setting label="" to return to default.
*/
func SetFSCreateLabel(label string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/fscreate", syscall.Gettid()), label)
}

/*
FSCreateLabel returns the default label the kernel which the kernel is using
for file system objects created by this task. "" indicates default.
*/
func FSCreateLabel() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/fscreate", syscall.Gettid()))
}

// CurrentLabel returns the SELinux label of the current process thread, or an error.
func CurrentLabel() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/current", syscall.Gettid()))
}

// PidLabel returns the SELinux label of the given pid, or an error.
func PidLabel(pid int) (string, error) {
	return readCon(fmt.Sprintf("/proc/%d/attr/current", pid))
}

/*
ExecLabel returns the SELinux label that the kernel will use for any programs
that are executed by the current process thread, or an error.
*/
func ExecLabel() (string, error) {
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

/*
SetExecLabel sets the SELinux label that the kernel will use for any programs
that are executed by the current process thread, or an error.
*/
func SetExecLabel(label string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/exec", syscall.Gettid()), label)
}

// Get returns the Context as a string
func (c Context) Get() string {
	return fmt.Sprintf("%s:%s:%s:%s", c["user"], c["role"], c["type"], c["level"])
}

// NewContext creates a new Context struct from the specified label
func NewContext(label string) Context {
	c := make(Context)

	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		c["user"] = con[0]
		c["role"] = con[1]
		c["type"] = con[2]
		c["level"] = con[3]
	}
	return c
}

// ReserveLabel reserves the MLS/MCS level component of the specified label
func ReserveLabel(label string) {
	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		mcsAdd(con[3])
	}
}

func selinuxEnforcePath() string {
	return fmt.Sprintf("%s/enforce", selinuxPath)
}

// EnforceMode returns the current SELinux mode Enforcing, Permissive, Disabled
func EnforceMode() int {
	var enforce int

	enforceS, err := readCon(selinuxEnforcePath())
	if err != nil {
		return -1
	}

	enforce, err = strconv.Atoi(string(enforceS))
	if err != nil {
		return -1
	}
	return enforce
}

/*
SetEnforceMode sets the current SELinux mode Enforcing, Permissive.
Disabled is not valid, since this needs to be set at boot time.
*/
func SetEnforceMode(mode int) error {
	return writeCon(selinuxEnforcePath(), fmt.Sprintf("%d", mode))
}

/*
DefaultEnforceMode returns the systems default SELinux mode Enforcing,
Permissive or Disabled. Note this is is just the default at boot time.
EnforceMode tells you the systems current mode.
*/
func DefaultEnforceMode() int {
	switch readConfig(selinuxTag) {
	case "enforcing":
		return Enforcing
	case "permissive":
		return Permissive
	}
	return Disabled
}

func mcsAdd(mcs string) error {
	state.Lock()
	defer state.Unlock()
	if state.mcsList[mcs] {
		return fmt.Errorf("MCS Label already exists")
	}
	state.mcsList[mcs] = true
	return nil
}

func mcsDelete(mcs string) {
	state.Lock()
	defer state.Unlock()
	state.mcsList[mcs] = false
}

func intToMcs(id int, catRange uint32) string {
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
		TIER--
	}
	TIER = SETSIZE - TIER
	ORD = ORD + TIER
	return fmt.Sprintf("s0:c%d,c%d", TIER, ORD)
}

func uniqMcs(catRange uint32) string {
	var (
		n      uint32
		c1, c2 uint32
		mcs    string
	)

	for {
		binary.Read(rand.Reader, binary.LittleEndian, &n)
		c1 = n % catRange
		binary.Read(rand.Reader, binary.LittleEndian, &n)
		c2 = n % catRange
		if c1 == c2 {
			continue
		} else {
			if c1 > c2 {
				c1, c2 = c2, c1
			}
		}
		mcs = fmt.Sprintf("s0:c%d,c%d", c1, c2)
		if err := mcsAdd(mcs); err != nil {
			continue
		}
		break
	}
	return mcs
}

/*
ReleaseLabel will unreserve the MLS/MCS Level field of the specified label.
Allowing it to be used by another process.
*/
func ReleaseLabel(label string) {
	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		mcsDelete(con[3])
	}
}

var roFileLabel string

// ROFileLabel returns the specified SELinux readonly file label
func ROFileLabel() (fileLabel string) {
	return roFileLabel
}

/*
ContainerLabels returns an allocated processLabel and fileLabel to be used for
container labeling by the calling process.
*/
func ContainerLabels() (processLabel string, fileLabel string) {
	var (
		val, key string
		bufin    *bufio.Reader
	)

	if !GetEnabled() {
		return "", ""
	}
	lxcPath := fmt.Sprintf("%s/contexts/lxc_contexts", getSELinuxPolicyRoot())
	in, err := os.Open(lxcPath)
	if err != nil {
		return "", ""
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
			if key == "ro_file" {
				roFileLabel = strings.Trim(val, "\"")
			}
		}
	}

	if processLabel == "" || fileLabel == "" {
		return "", ""
	}

	if roFileLabel == "" {
		roFileLabel = fileLabel
	}
exit:
	mcs := uniqMcs(1024)
	scon := NewContext(processLabel)
	scon["level"] = mcs
	processLabel = scon.Get()
	scon = NewContext(fileLabel)
	scon["level"] = mcs
	fileLabel = scon.Get()
	return processLabel, fileLabel
}

// SecurityCheckContext validates that the SELinux label is understood by the kernel
func SecurityCheckContext(val string) error {
	return writeCon(fmt.Sprintf("%s.context", selinuxPath), val)
}

/*
CopyLevel returns a label with the MLS/MCS level from src label replaces on
the dest label.
*/
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

// Prevent users from relabing system files
func badPrefix(fpath string) error {
	var badprefixes = []string{"/usr"}

	for _, prefix := range badprefixes {
		if fpath == prefix || strings.HasPrefix(fpath, fmt.Sprintf("%s/", prefix)) {
			return fmt.Errorf("relabeling content in %s is not allowed", prefix)
		}
	}
	return nil
}

// Chcon changes the fpath file object to the SELinux label label.
// If the fpath is a directory and recurse is true Chcon will walk the
// directory tree setting the label
func Chcon(fpath string, label string, recurse bool) error {
	if label == "" {
		return nil
	}
	if err := badPrefix(fpath); err != nil {
		return err
	}
	callback := func(p string, info os.FileInfo, err error) error {
		return SetFileLabel(p, label)
	}

	if recurse {
		return filepath.Walk(fpath, callback)
	}

	return SetFileLabel(fpath, label)
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
	return []string{"user:" + con["user"],
		"role:" + con["role"],
		"type:" + con["type"],
		"level:" + con["level"]}
}

// DisableSecOpt returns a security opt that can be used to disabling SELinux
// labeling support for future container processes
func DisableSecOpt() []string {
	return []string{"disable"}
}
