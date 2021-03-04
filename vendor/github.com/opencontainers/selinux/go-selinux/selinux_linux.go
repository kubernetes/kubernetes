// +build selinux,linux

package selinux

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/opencontainers/selinux/pkg/pwalk"
	"github.com/pkg/errors"
	"github.com/willf/bitset"
	"golang.org/x/sys/unix"
)

const (
	minSensLen       = 2
	contextFile      = "/usr/share/containers/selinux/contexts"
	selinuxDir       = "/etc/selinux/"
	selinuxConfig    = selinuxDir + "config"
	selinuxfsMount   = "/sys/fs/selinux"
	selinuxTypeTag   = "SELINUXTYPE"
	selinuxTag       = "SELINUX"
	xattrNameSelinux = "security.selinux"
)

type selinuxState struct {
	enabledSet    bool
	enabled       bool
	selinuxfsOnce sync.Once
	selinuxfs     string
	mcsList       map[string]bool
	sync.Mutex
}

type level struct {
	sens uint
	cats *bitset.BitSet
}

type mlsRange struct {
	low  *level
	high *level
}

type levelItem byte

const (
	sensitivity levelItem = 's'
	category    levelItem = 'c'
)

var (
	assignRegex       = regexp.MustCompile(`^([^=]+)=(.*)$`)
	readOnlyFileLabel string
	state             = selinuxState{
		mcsList: make(map[string]bool),
	}

	// for attrPath()
	attrPathOnce   sync.Once
	haveThreadSelf bool
)

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

// setDisabled disables SELinux support for the package
func setDisabled() {
	state.setEnable(false)
}

func verifySELinuxfsMount(mnt string) bool {
	var buf unix.Statfs_t
	for {
		err := unix.Statfs(mnt, &buf)
		if err == nil {
			break
		}
		if err == unix.EAGAIN {
			continue
		}
		return false
	}

	if uint32(buf.Type) != uint32(unix.SELINUX_MAGIC) {
		return false
	}
	if (buf.Flags & unix.ST_RDONLY) != 0 {
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
		txt := s.Bytes()
		// The first field after - is fs type.
		// Safe as spaces in mountpoints are encoded as \040
		if !bytes.Contains(txt, []byte(" - selinuxfs ")) {
			continue
		}
		const mPos = 5 // mount point is 5th field
		fields := bytes.SplitN(txt, []byte(" "), mPos+1)
		if len(fields) < mPos+1 {
			continue
		}
		return string(fields[mPos-1])
	}

	return ""
}

func (s *selinuxState) getSELinuxfs() string {
	s.selinuxfsOnce.Do(func() {
		s.selinuxfs = findSELinuxfs()
	})

	return s.selinuxfs
}

// getSelinuxMountPoint returns the path to the mountpoint of an selinuxfs
// filesystem or an empty string if no mountpoint is found.  Selinuxfs is
// a proc-like pseudo-filesystem that exposes the SELinux policy API to
// processes.  The existence of an selinuxfs mount is used to determine
// whether SELinux is currently enabled or not.
func getSelinuxMountPoint() string {
	return state.getSELinuxfs()
}

// getEnabled returns whether SELinux is currently enabled.
func getEnabled() bool {
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

func isProcHandle(fh *os.File) error {
	var buf unix.Statfs_t
	err := unix.Fstatfs(int(fh.Fd()), &buf)
	if err != nil {
		return errors.Wrapf(err, "statfs(%q) failed", fh.Name())
	}
	if buf.Type != unix.PROC_SUPER_MAGIC {
		return errors.Errorf("file %q is not on procfs", fh.Name())
	}

	return nil
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

	if err := isProcHandle(in); err != nil {
		return "", err
	}

	var retval string
	if _, err := fmt.Fscanf(in, "%s", &retval); err != nil {
		return "", err
	}
	return strings.Trim(retval, "\x00"), nil
}

// classIndex returns the int index for an object class in the loaded policy,
// or -1 and an error
func classIndex(class string) (int, error) {
	permpath := fmt.Sprintf("class/%s/index", class)
	indexpath := filepath.Join(getSelinuxMountPoint(), permpath)

	indexB, err := ioutil.ReadFile(indexpath)
	if err != nil {
		return -1, err
	}
	index, err := strconv.Atoi(string(indexB))
	if err != nil {
		return -1, err
	}

	return index, nil
}

// setFileLabel sets the SELinux label for this path or returns an error.
func setFileLabel(fpath string, label string) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	if err := unix.Lsetxattr(fpath, xattrNameSelinux, []byte(label), 0); err != nil {
		return errors.Wrapf(err, "failed to set file label on %s", fpath)
	}
	return nil
}

// fileLabel returns the SELinux label for this path or returns an error.
func fileLabel(fpath string) (string, error) {
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

// setFSCreateLabel tells kernel the label to create all file system objects
// created by this task. Setting label="" to return to default.
func setFSCreateLabel(label string) error {
	return writeAttr("fscreate", label)
}

// fsCreateLabel returns the default label the kernel which the kernel is using
// for file system objects created by this task. "" indicates default.
func fsCreateLabel() (string, error) {
	return readAttr("fscreate")
}

// currentLabel returns the SELinux label of the current process thread, or an error.
func currentLabel() (string, error) {
	return readAttr("current")
}

// pidLabel returns the SELinux label of the given pid, or an error.
func pidLabel(pid int) (string, error) {
	return readCon(fmt.Sprintf("/proc/%d/attr/current", pid))
}

// ExecLabel returns the SELinux label that the kernel will use for any programs
// that are executed by the current process thread, or an error.
func execLabel() (string, error) {
	return readAttr("exec")
}

func writeCon(fpath, val string) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	if val == "" {
		if !getEnabled() {
			return nil
		}
	}

	out, err := os.OpenFile(fpath, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer out.Close()

	if err := isProcHandle(out); err != nil {
		return err
	}

	if val != "" {
		_, err = out.Write([]byte(val))
	} else {
		_, err = out.Write(nil)
	}
	if err != nil {
		return errors.Wrapf(err, "failed to set %s on procfs", fpath)
	}
	return nil
}

func attrPath(attr string) string {
	// Linux >= 3.17 provides this
	const threadSelfPrefix = "/proc/thread-self/attr"

	attrPathOnce.Do(func() {
		st, err := os.Stat(threadSelfPrefix)
		if err == nil && st.Mode().IsDir() {
			haveThreadSelf = true
		}
	})

	if haveThreadSelf {
		return path.Join(threadSelfPrefix, attr)
	}

	return path.Join("/proc/self/task/", strconv.Itoa(unix.Gettid()), "/attr/", attr)
}

func readAttr(attr string) (string, error) {
	return readCon(attrPath(attr))
}

func writeAttr(attr, val string) error {
	return writeCon(attrPath(attr), val)
}

// canonicalizeContext takes a context string and writes it to the kernel
// the function then returns the context that the kernel will use. Use this
// function to check if two contexts are equivalent
func canonicalizeContext(val string) (string, error) {
	return readWriteCon(filepath.Join(getSelinuxMountPoint(), "context"), val)
}

// computeCreateContext requests the type transition from source to target for
// class from the kernel.
func computeCreateContext(source string, target string, class string) (string, error) {
	classidx, err := classIndex(class)
	if err != nil {
		return "", err
	}

	return readWriteCon(filepath.Join(getSelinuxMountPoint(), "create"), fmt.Sprintf("%s %s %d", source, target, classidx))
}

// catsToBitset stores categories in a bitset.
func catsToBitset(cats string) (*bitset.BitSet, error) {
	bitset := &bitset.BitSet{}

	catlist := strings.Split(cats, ",")
	for _, r := range catlist {
		ranges := strings.SplitN(r, ".", 2)
		if len(ranges) > 1 {
			catstart, err := parseLevelItem(ranges[0], category)
			if err != nil {
				return nil, err
			}
			catend, err := parseLevelItem(ranges[1], category)
			if err != nil {
				return nil, err
			}
			for i := catstart; i <= catend; i++ {
				bitset.Set(i)
			}
		} else {
			cat, err := parseLevelItem(ranges[0], category)
			if err != nil {
				return nil, err
			}
			bitset.Set(cat)
		}
	}

	return bitset, nil
}

// parseLevelItem parses and verifies that a sensitivity or category are valid
func parseLevelItem(s string, sep levelItem) (uint, error) {
	if len(s) < minSensLen || levelItem(s[0]) != sep {
		return 0, ErrLevelSyntax
	}
	val, err := strconv.ParseUint(s[1:], 10, 32)
	if err != nil {
		return 0, err
	}

	return uint(val), nil
}

// parseLevel fills a level from a string that contains
// a sensitivity and categories
func (l *level) parseLevel(levelStr string) error {
	lvl := strings.SplitN(levelStr, ":", 2)
	sens, err := parseLevelItem(lvl[0], sensitivity)
	if err != nil {
		return errors.Wrap(err, "failed to parse sensitivity")
	}
	l.sens = sens
	if len(lvl) > 1 {
		cats, err := catsToBitset(lvl[1])
		if err != nil {
			return errors.Wrap(err, "failed to parse categories")
		}
		l.cats = cats
	}

	return nil
}

// rangeStrToMLSRange marshals a string representation of a range.
func rangeStrToMLSRange(rangeStr string) (*mlsRange, error) {
	mlsRange := &mlsRange{}
	levelSlice := strings.SplitN(rangeStr, "-", 2)

	switch len(levelSlice) {
	// rangeStr that has a low and a high level, e.g. s4:c0.c1023-s6:c0.c1023
	case 2:
		mlsRange.high = &level{}
		if err := mlsRange.high.parseLevel(levelSlice[1]); err != nil {
			return nil, errors.Wrapf(err, "failed to parse high level %q", levelSlice[1])
		}
		fallthrough
	// rangeStr that is single level, e.g. s6:c0,c3,c5,c30.c1023
	case 1:
		mlsRange.low = &level{}
		if err := mlsRange.low.parseLevel(levelSlice[0]); err != nil {
			return nil, errors.Wrapf(err, "failed to parse low level %q", levelSlice[0])
		}
	}

	if mlsRange.high == nil {
		mlsRange.high = mlsRange.low
	}

	return mlsRange, nil
}

// bitsetToStr takes a category bitset and returns it in the
// canonical selinux syntax
func bitsetToStr(c *bitset.BitSet) string {
	var str string
	i, e := c.NextSet(0)
	len := 0
	for e {
		if len == 0 {
			if str != "" {
				str += ","
			}
			str += "c" + strconv.Itoa(int(i))
		}

		next, e := c.NextSet(i + 1)
		if e {
			// consecutive cats
			if next == i+1 {
				len++
				i = next
				continue
			}
		}
		if len == 1 {
			str += ",c" + strconv.Itoa(int(i))
		} else if len > 1 {
			str += ".c" + strconv.Itoa(int(i))
		}
		if !e {
			break
		}
		len = 0
		i = next
	}

	return str
}

func (l1 *level) equal(l2 *level) bool {
	if l2 == nil || l1 == nil {
		return l1 == l2
	}
	if l1.sens != l2.sens {
		return false
	}
	return l1.cats.Equal(l2.cats)
}

// String returns an mlsRange as a string.
func (m mlsRange) String() string {
	low := "s" + strconv.Itoa(int(m.low.sens))
	if m.low.cats != nil && m.low.cats.Count() > 0 {
		low += ":" + bitsetToStr(m.low.cats)
	}

	if m.low.equal(m.high) {
		return low
	}

	high := "s" + strconv.Itoa(int(m.high.sens))
	if m.high.cats != nil && m.high.cats.Count() > 0 {
		high += ":" + bitsetToStr(m.high.cats)
	}

	return low + "-" + high
}

func max(a, b uint) uint {
	if a > b {
		return a
	}
	return b
}

func min(a, b uint) uint {
	if a < b {
		return a
	}
	return b
}

// calculateGlbLub computes the glb (greatest lower bound) and lub (least upper bound)
// of a source and target range.
// The glblub is calculated as the greater of the low sensitivities and
// the lower of the high sensitivities and the and of each category bitset.
func calculateGlbLub(sourceRange, targetRange string) (string, error) {
	s, err := rangeStrToMLSRange(sourceRange)
	if err != nil {
		return "", err
	}
	t, err := rangeStrToMLSRange(targetRange)
	if err != nil {
		return "", err
	}

	if s.high.sens < t.low.sens || t.high.sens < s.low.sens {
		/* these ranges have no common sensitivities */
		return "", ErrIncomparable
	}

	outrange := &mlsRange{low: &level{}, high: &level{}}

	/* take the greatest of the low */
	outrange.low.sens = max(s.low.sens, t.low.sens)

	/* take the least of the high */
	outrange.high.sens = min(s.high.sens, t.high.sens)

	/* find the intersecting categories */
	if s.low.cats != nil && t.low.cats != nil {
		outrange.low.cats = s.low.cats.Intersection(t.low.cats)
	}
	if s.high.cats != nil && t.high.cats != nil {
		outrange.high.cats = s.high.cats.Intersection(t.high.cats)
	}

	return outrange.String(), nil
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

// setExecLabel sets the SELinux label that the kernel will use for any programs
// that are executed by the current process thread, or an error.
func setExecLabel(label string) error {
	return writeAttr("exec", label)
}

// setTaskLabel sets the SELinux label for the current thread, or an error.
// This requires the dyntransition permission.
func setTaskLabel(label string) error {
	return writeAttr("current", label)
}

// setSocketLabel takes a process label and tells the kernel to assign the
// label to the next socket that gets created
func setSocketLabel(label string) error {
	return writeAttr("sockcreate", label)
}

// socketLabel retrieves the current socket label setting
func socketLabel() (string, error) {
	return readAttr("sockcreate")
}

// peerLabel retrieves the label of the client on the other side of a socket
func peerLabel(fd uintptr) (string, error) {
	return unix.GetsockoptString(int(fd), unix.SOL_SOCKET, unix.SO_PEERSEC)
}

// setKeyLabel takes a process label and tells the kernel to assign the
// label to the next kernel keyring that gets created
func setKeyLabel(label string) error {
	err := writeCon("/proc/self/attr/keycreate", label)
	if os.IsNotExist(errors.Cause(err)) {
		return nil
	}
	if label == "" && os.IsPermission(errors.Cause(err)) {
		return nil
	}
	return err
}

// keyLabel retrieves the current kernel keyring label setting
func keyLabel() (string, error) {
	return readCon("/proc/self/attr/keycreate")
}

// get returns the Context as a string
func (c Context) get() string {
	if c["level"] != "" {
		return fmt.Sprintf("%s:%s:%s:%s", c["user"], c["role"], c["type"], c["level"])
	}
	return fmt.Sprintf("%s:%s:%s", c["user"], c["role"], c["type"])
}

// newContext creates a new Context struct from the specified label
func newContext(label string) (Context, error) {
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

// clearLabels clears all reserved labels
func clearLabels() {
	state.Lock()
	state.mcsList = make(map[string]bool)
	state.Unlock()
}

// reserveLabel reserves the MLS/MCS level component of the specified label
func reserveLabel(label string) {
	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		if len(con) > 3 {
			mcsAdd(con[3])
		}
	}
}

func selinuxEnforcePath() string {
	return path.Join(getSelinuxMountPoint(), "enforce")
}

// enforceMode returns the current SELinux mode Enforcing, Permissive, Disabled
func enforceMode() int {
	var enforce int

	enforceB, err := ioutil.ReadFile(selinuxEnforcePath())
	if err != nil {
		return -1
	}
	enforce, err = strconv.Atoi(string(enforceB))
	if err != nil {
		return -1
	}
	return enforce
}

// setEnforceMode sets the current SELinux mode Enforcing, Permissive.
// Disabled is not valid, since this needs to be set at boot time.
func setEnforceMode(mode int) error {
	return ioutil.WriteFile(selinuxEnforcePath(), []byte(strconv.Itoa(mode)), 0644)
}

// defaultEnforceMode returns the systems default SELinux mode Enforcing,
// Permissive or Disabled. Note this is is just the default at boot time.
// EnforceMode tells you the systems current mode.
func defaultEnforceMode() int {
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

// releaseLabel un-reserves the MLS/MCS Level field of the specified label,
// allowing it to be used by another process.
func releaseLabel(label string) {
	if len(label) != 0 {
		con := strings.SplitN(label, ":", 4)
		if len(con) > 3 {
			mcsDelete(con[3])
		}
	}
}

// roFileLabel returns the specified SELinux readonly file label
func roFileLabel() string {
	return readOnlyFileLabel
}

func openContextFile() (*os.File, error) {
	if f, err := os.Open(contextFile); err == nil {
		return f, nil
	}
	lxcPath := filepath.Join(getSELinuxPolicyRoot(), "/contexts/lxc_contexts")
	return os.Open(lxcPath)
}

var labels = loadLabels()

func loadLabels() map[string]string {
	var (
		val, key string
		bufin    *bufio.Reader
	)

	labels := make(map[string]string)
	in, err := openContextFile()
	if err != nil {
		return labels
	}
	defer in.Close()

	bufin = bufio.NewReader(in)

	for done := false; !done; {
		var line string
		if line, err = bufin.ReadString('\n'); err != nil {
			if err == io.EOF {
				done = true
			} else {
				break
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
			labels[key] = strings.Trim(val, "\"")
		}
	}

	return labels
}

// kvmContainerLabels returns the default processLabel and mountLabel to be used
// for kvm containers by the calling process.
func kvmContainerLabels() (string, string) {
	processLabel := labels["kvm_process"]
	if processLabel == "" {
		processLabel = labels["process"]
	}

	return addMcs(processLabel, labels["file"])
}

// initContainerLabels returns the default processLabel and file labels to be
// used for containers running an init system like systemd by the calling process.
func initContainerLabels() (string, string) {
	processLabel := labels["init_process"]
	if processLabel == "" {
		processLabel = labels["process"]
	}

	return addMcs(processLabel, labels["file"])
}

// containerLabels returns an allocated processLabel and fileLabel to be used for
// container labeling by the calling process.
func containerLabels() (processLabel string, fileLabel string) {
	if !getEnabled() {
		return "", ""
	}

	processLabel = labels["process"]
	fileLabel = labels["file"]
	readOnlyFileLabel = labels["ro_file"]

	if processLabel == "" || fileLabel == "" {
		return "", fileLabel
	}

	if readOnlyFileLabel == "" {
		readOnlyFileLabel = fileLabel
	}

	return addMcs(processLabel, fileLabel)
}

func addMcs(processLabel, fileLabel string) (string, string) {
	scon, _ := NewContext(processLabel)
	if scon["level"] != "" {
		mcs := uniqMcs(CategoryRange)
		scon["level"] = mcs
		processLabel = scon.Get()
		scon, _ = NewContext(fileLabel)
		scon["level"] = mcs
		fileLabel = scon.Get()
	}
	return processLabel, fileLabel
}

// securityCheckContext validates that the SELinux label is understood by the kernel
func securityCheckContext(val string) error {
	return ioutil.WriteFile(path.Join(getSelinuxMountPoint(), "context"), []byte(val), 0644)
}

// copyLevel returns a label with the MLS/MCS level from src label replaced on
// the dest label.
func copyLevel(src, dest string) (string, error) {
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

// Prevent users from relabeling system files
func badPrefix(fpath string) error {
	if fpath == "" {
		return ErrEmptyPath
	}

	badPrefixes := []string{"/usr"}
	for _, prefix := range badPrefixes {
		if strings.HasPrefix(fpath, prefix) {
			return errors.Errorf("relabeling content in %s is not allowed", prefix)
		}
	}
	return nil
}

// chcon changes the fpath file object to the SELinux label label.
// If fpath is a directory and recurse is true, then chcon walks the
// directory tree setting the label.
func chcon(fpath string, label string, recurse bool) error {
	if fpath == "" {
		return ErrEmptyPath
	}
	if label == "" {
		return nil
	}
	if err := badPrefix(fpath); err != nil {
		return err
	}

	if !recurse {
		return SetFileLabel(fpath, label)
	}

	return pwalk.Walk(fpath, func(p string, info os.FileInfo, err error) error {
		e := SetFileLabel(p, label)
		// Walk a file tree can race with removal, so ignore ENOENT
		if os.IsNotExist(errors.Cause(e)) {
			return nil
		}
		return e
	})
}

// dupSecOpt takes an SELinux process label and returns security options that
// can be used to set the SELinux Type and Level for future container processes.
func dupSecOpt(src string) ([]string, error) {
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

// disableSecOpt returns a security opt that can be used to disable SELinux
// labeling support for future container processes.
func disableSecOpt() []string {
	return []string{"disable"}
}
