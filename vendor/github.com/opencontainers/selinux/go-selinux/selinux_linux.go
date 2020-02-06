// +build selinux,linux

package selinux

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"golang.org/x/sys/unix"
)

const (
	// Enforcing constant indicate SELinux is in enforcing mode
	Enforcing = 1
	// Permissive constant to indicate SELinux is in permissive mode
	Permissive = 0
	// Disabled constant to indicate SELinux is disabled
	Disabled = -1

	selinuxDir       = "/etc/selinux/"
	selinuxConfig    = selinuxDir + "config"
	selinuxfsMount   = "/sys/fs/selinux"
	selinuxTypeTag   = "SELINUXTYPE"
	selinuxTag       = "SELINUX"
	xattrNameSelinux = "security.selinux"
	stRdOnly         = 0x01
	selinuxfsMagic   = 0xf97cff8c
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
	// ErrMCSAlreadyExists is returned when trying to allocate a duplicate MCS.
	ErrMCSAlreadyExists = errors.New("MCS label already exists")
	// ErrEmptyPath is returned when an empty path has been specified.
	ErrEmptyPath = errors.New("empty path")
	// InvalidLabel is returned when an invalid label is specified.
	InvalidLabel = errors.New("Invalid Label")

	assignRegex = regexp.MustCompile(`^([^=]+)=(.*)$`)
	roFileLabel string
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

func verifySELinuxfsMount(mnt string) bool {
	var buf syscall.Statfs_t
	for {
		err := syscall.Statfs(mnt, &buf)
		if err == nil {
			break
		}
		if err == syscall.EAGAIN {
			continue
		}
		return false
	}
	if uint32(buf.Type) != uint32(selinuxfsMagic) {
		return false
	}
	if (buf.Flags & stRdOnly) != 0 {
		return false
	}

	return true
}

func findSELinuxfs() string {
	// fast path: check the default mount first
	if verifySELinuxfsMount(selinuxfsMount) {
		return selinuxfsMount
	}

	// check if selinuxfs is available before going the slow path
	fs, err := ioutil.ReadFile("/proc/filesystems")
	if err != nil {
		return ""
	}
	if !bytes.Contains(fs, []byte("\tselinuxfs\n")) {
		return ""
	}

	// slow path: try to find among the mounts
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return ""
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for {
		mnt := findSELinuxfsMount(scanner)
		if mnt == "" { // error or not found
			return ""
		}
		if verifySELinuxfsMount(mnt) {
			return mnt
		}
	}
}

// findSELinuxfsMount returns a next selinuxfs mount point found,
// if there is one, or an empty string in case of EOF or error.
func findSELinuxfsMount(s *bufio.Scanner) string {
	for s.Scan() {
		txt := s.Text()
		// The first field after - is fs type.
		// Safe as spaces in mountpoints are encoded as \040
		if !strings.Contains(txt, " - selinuxfs ") {
			continue
		}
		const mPos = 5 // mount point is 5th field
		fields := strings.SplitN(txt, " ", mPos+1)
		if len(fields) < mPos+1 {
			continue
		}
		return fields[mPos-1]
	}

	return ""
}

func (s *selinuxState) getSELinuxfs() string {
	s.Lock()
	selinuxfs := s.selinuxfs
	selinuxfsSet := s.selinuxfsSet
	s.Unlock()
	if selinuxfsSet {
		return selinuxfs
	}

	return s.setSELinuxfs(findSELinuxfs())
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

func readConfig(target string) string {
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
	return filepath.Join(selinuxDir, readConfig(selinuxTypeTag))
}

func isProcHandle(fh *os.File) (bool, error) {
	var buf unix.Statfs_t
	err := unix.Fstatfs(int(fh.Fd()), &buf)
	return buf.Type == unix.PROC_SUPER_MAGIC, err
}

func readCon(fpath string) (string, error) {
	if fpath == "" {
		return "", ErrEmptyPath
	}

	in, err := os.Open(fpath)
	if err != nil {
		return "", err
	}
	defer in.Close()

	if ok, err := isProcHandle(in); err != nil {
		return "", err
	} else if !ok {
		return "", fmt.Errorf("%s not on procfs", fpath)
	}

	var retval string
	if _, err := fmt.Fscanf(in, "%s", &retval); err != nil {
		return "", err
	}
	return strings.Trim(retval, "\x00"), nil
}

// SetFileLabel sets the SELinux label for this path or returns an error.
func SetFileLabel(fpath string, label string) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	return lsetxattr(fpath, xattrNameSelinux, []byte(label), 0)
}

// FileLabel returns the SELinux label for this path or returns an error.
func FileLabel(fpath string) (string, error) {
	if fpath == "" {
		return "", ErrEmptyPath
	}

	label, err := lgetxattr(fpath, xattrNameSelinux)
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

func writeCon(fpath string, val string) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	if val == "" {
		if !GetEnabled() {
			return nil
		}
	}

	out, err := os.OpenFile(fpath, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer out.Close()

	if ok, err := isProcHandle(out); err != nil {
		return err
	} else if !ok {
		return fmt.Errorf("%s not on procfs", fpath)
	}

	if val != "" {
		_, err = out.Write([]byte(val))
	} else {
		_, err = out.Write(nil)
	}
	return err
}

/*
CanonicalizeContext takes a context string and writes it to the kernel
the function then returns the context that the kernel will use.  This function
can be used to see if two contexts are equivalent
*/
func CanonicalizeContext(val string) (string, error) {
	return readWriteCon(filepath.Join(getSelinuxMountPoint(), "context"), val)
}

func readWriteCon(fpath string, val string) (string, error) {
	if fpath == "" {
		return "", ErrEmptyPath
	}
	f, err := os.OpenFile(fpath, os.O_RDWR, 0)
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = f.Write([]byte(val))
	if err != nil {
		return "", err
	}

	var retval string
	if _, err := fmt.Fscanf(f, "%s", &retval); err != nil {
		return "", err
	}
	return strings.Trim(retval, "\x00"), nil
}

/*
SetExecLabel sets the SELinux label that the kernel will use for any programs
that are executed by the current process thread, or an error.
*/
func SetExecLabel(label string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/exec", syscall.Gettid()), label)
}

/*
SetTaskLabel sets the SELinux label for the current thread, or an error.
This requires the dyntransition permission.
*/
func SetTaskLabel(label string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/current", syscall.Gettid()), label)
}

// SetSocketLabel takes a process label and tells the kernel to assign the
// label to the next socket that gets created
func SetSocketLabel(label string) error {
	return writeCon(fmt.Sprintf("/proc/self/task/%d/attr/sockcreate", syscall.Gettid()), label)
}

// SocketLabel retrieves the current socket label setting
func SocketLabel() (string, error) {
	return readCon(fmt.Sprintf("/proc/self/task/%d/attr/sockcreate", syscall.Gettid()))
}

// PeerLabel retrieves the label of the client on the other side of a socket
func PeerLabel(fd uintptr) (string, error) {
	return unix.GetsockoptString(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERSEC)
}

// SetKeyLabel takes a process label and tells the kernel to assign the
// label to the next kernel keyring that gets created
func SetKeyLabel(label string) error {
	err := writeCon("/proc/self/attr/keycreate", label)
	if os.IsNotExist(err) {
		return nil
	}
	if label == "" && os.IsPermission(err) && !GetEnabled() {
		return nil
	}
	return err
}

// KeyLabel retrieves the current kernel keyring label setting
func KeyLabel() (string, error) {
	return readCon("/proc/self/attr/keycreate")
}

// Get returns the Context as a string
func (c Context) Get() string {
	if c["level"] != "" {
		return fmt.Sprintf("%s:%s:%s:%s", c["user"], c["role"], c["type"], c["level"])
	}
	return fmt.Sprintf("%s:%s:%s", c["user"], c["role"], c["type"])
}

// NewContext creates a new Context struct from the specified label
func NewContext(label string) (Context, error) {
	c := make(Context)

	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		if len(con) < 3 {
			return c, InvalidLabel
		}
		c["user"] = con[0]
		c["role"] = con[1]
		c["type"] = con[2]
		if len(con) > 3 {
			c["level"] = con[3]
		}
	}
	return c, nil
}

// ClearLabels clears all reserved labels
func ClearLabels() {
	state.Lock()
	state.mcsList = make(map[string]bool)
	state.Unlock()
}

// ReserveLabel reserves the MLS/MCS level component of the specified label
func ReserveLabel(label string) {
	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		if len(con) > 3 {
			mcsAdd(con[3])
		}
	}
}

func selinuxEnforcePath() string {
	return fmt.Sprintf("%s/enforce", getSelinuxMountPoint())
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
	if mcs == "" {
		return nil
	}
	state.Lock()
	defer state.Unlock()
	if state.mcsList[mcs] {
		return ErrMCSAlreadyExists
	}
	state.mcsList[mcs] = true
	return nil
}

func mcsDelete(mcs string) {
	if mcs == "" {
		return
	}
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
		if len(con) > 3 {
			mcsDelete(con[3])
		}
	}
}

// ROFileLabel returns the specified SELinux readonly file label
func ROFileLabel() string {
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
	scon, _ := NewContext(processLabel)
	if scon["level"] != "" {
		mcs := uniqMcs(1024)
		scon["level"] = mcs
		processLabel = scon.Get()
		scon, _ = NewContext(fileLabel)
		scon["level"] = mcs
		fileLabel = scon.Get()
	}
	return processLabel, fileLabel
}

// SecurityCheckContext validates that the SELinux label is understood by the kernel
func SecurityCheckContext(val string) error {
	return writeCon(fmt.Sprintf("%s/context", getSelinuxMountPoint()), val)
}

/*
CopyLevel returns a label with the MLS/MCS level from src label replaced on
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
	scon, err := NewContext(src)
	if err != nil {
		return "", err
	}
	tcon, err := NewContext(dest)
	if err != nil {
		return "", err
	}
	mcsDelete(tcon["level"])
	mcsAdd(scon["level"])
	tcon["level"] = scon["level"]
	return tcon.Get(), nil
}

// Prevent users from relabing system files
func badPrefix(fpath string) error {
	if fpath == "" {
		return ErrEmptyPath
	}

	badPrefixes := []string{"/usr"}
	for _, prefix := range badPrefixes {
		if strings.HasPrefix(fpath, prefix) {
			return fmt.Errorf("relabeling content in %s is not allowed", prefix)
		}
	}
	return nil
}

// Chcon changes the `fpath` file object to the SELinux label `label`.
// If `fpath` is a directory and `recurse`` is true, Chcon will walk the
// directory tree setting the label.
func Chcon(fpath string, label string, recurse bool) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	if label == "" {
		return nil
	}
	if err := badPrefix(fpath); err != nil {
		return err
	}
	callback := func(p string, info os.FileInfo, err error) error {
		e := SetFileLabel(p, label)
		if os.IsNotExist(e) {
			return nil
		}
		return e
	}

	if recurse {
		return filepath.Walk(fpath, callback)
	}

	return SetFileLabel(fpath, label)
}

// DupSecOpt takes an SELinux process label and returns security options that
// can be used to set the SELinux Type and Level for future container processes.
func DupSecOpt(src string) ([]string, error) {
	if src == "" {
		return nil, nil
	}
	con, err := NewContext(src)
	if err != nil {
		return nil, err
	}
	if con["user"] == "" ||
		con["role"] == "" ||
		con["type"] == "" {
		return nil, nil
	}
	dup := []string{"user:" + con["user"],
		"role:" + con["role"],
		"type:" + con["type"],
	}

	if con["level"] != "" {
		dup = append(dup, "level:"+con["level"])
	}

	return dup, nil
}

// DisableSecOpt returns a security opt that can be used to disable SELinux
// labeling support for future container processes.
func DisableSecOpt() []string {
	return []string{"disable"}
}
