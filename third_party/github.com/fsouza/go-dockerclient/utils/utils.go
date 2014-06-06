// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package utils

import (
	"bytes"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"index/suffixarray"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	IAMSTATIC bool   // whether or not Docker itself was compiled statically via ./hack/make.sh binary
	INITSHA1  string // sha1sum of separate static dockerinit, if Docker itself was compiled dynamically via ./hack/make.sh dynbinary
	INITPATH  string // custom location to search for a valid dockerinit binary (available for packagers as a last resort escape hatch)
)

// A common interface to access the Fatal method of
// both testing.B and testing.T.
type Fataler interface {
	Fatal(args ...interface{})
}

// Go is a basic promise implementation: it wraps calls a function in a goroutine,
// and returns a channel which will later return the function's return value.
func Go(f func() error) chan error {
	ch := make(chan error)
	go func() {
		ch <- f()
	}()
	return ch
}

// Request a given URL and return an io.Reader
func Download(url string) (*http.Response, error) {
	var resp *http.Response
	var err error
	if resp, err = http.Get(url); err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, errors.New("Got HTTP status code >= 400: " + resp.Status)
	}
	return resp, nil
}

func logf(level string, format string, a ...interface{}) {
	// Retrieve the stack infos
	_, file, line, ok := runtime.Caller(2)
	if !ok {
		file = "<unknown>"
		line = -1
	} else {
		file = file[strings.LastIndex(file, "/")+1:]
	}

	fmt.Fprintf(os.Stderr, fmt.Sprintf("[%s] %s:%d %s\n", level, file, line, format), a...)
}

// Debug function, if the debug flag is set, then display. Do nothing otherwise
// If Docker is in damon mode, also send the debug info on the socket
func Debugf(format string, a ...interface{}) {
	if os.Getenv("DEBUG") != "" {
		logf("debug", format, a...)
	}
}

func Errorf(format string, a ...interface{}) {
	logf("error", format, a...)
}

// HumanDuration returns a human-readable approximation of a duration
// (eg. "About a minute", "4 hours ago", etc.)
func HumanDuration(d time.Duration) string {
	if seconds := int(d.Seconds()); seconds < 1 {
		return "Less than a second"
	} else if seconds < 60 {
		return fmt.Sprintf("%d seconds", seconds)
	} else if minutes := int(d.Minutes()); minutes == 1 {
		return "About a minute"
	} else if minutes < 60 {
		return fmt.Sprintf("%d minutes", minutes)
	} else if hours := int(d.Hours()); hours == 1 {
		return "About an hour"
	} else if hours < 48 {
		return fmt.Sprintf("%d hours", hours)
	} else if hours < 24*7*2 {
		return fmt.Sprintf("%d days", hours/24)
	} else if hours < 24*30*3 {
		return fmt.Sprintf("%d weeks", hours/24/7)
	} else if hours < 24*365*2 {
		return fmt.Sprintf("%d months", hours/24/30)
	}
	return fmt.Sprintf("%f years", d.Hours()/24/365)
}

// HumanSize returns a human-readable approximation of a size
// using SI standard (eg. "44kB", "17MB")
func HumanSize(size int64) string {
	i := 0
	var sizef float64
	sizef = float64(size)
	units := []string{"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"}
	for sizef >= 1000.0 {
		sizef = sizef / 1000.0
		i++
	}
	return fmt.Sprintf("%.4g %s", sizef, units[i])
}

// Parses a human-readable string representing an amount of RAM
// in bytes, kibibytes, mebibytes or gibibytes, and returns the
// number of bytes, or -1 if the string is unparseable.
// Units are case-insensitive, and the 'b' suffix is optional.
func RAMInBytes(size string) (bytes int64, err error) {
	re, error := regexp.Compile("^(\\d+)([kKmMgG])?[bB]?$")
	if error != nil {
		return -1, error
	}

	matches := re.FindStringSubmatch(size)

	if len(matches) != 3 {
		return -1, fmt.Errorf("Invalid size: '%s'", size)
	}

	memLimit, error := strconv.ParseInt(matches[1], 10, 0)
	if error != nil {
		return -1, error
	}

	unit := strings.ToLower(matches[2])

	if unit == "k" {
		memLimit *= 1024
	} else if unit == "m" {
		memLimit *= 1024 * 1024
	} else if unit == "g" {
		memLimit *= 1024 * 1024 * 1024
	}

	return memLimit, nil
}

func Trunc(s string, maxlen int) string {
	if len(s) <= maxlen {
		return s
	}
	return s[:maxlen]
}

// Figure out the absolute path of our own binary (if it's still around).
func SelfPath() string {
	path, err := exec.LookPath(os.Args[0])
	if err != nil {
		if os.IsNotExist(err) {
			return ""
		}
		if execErr, ok := err.(*exec.Error); ok && os.IsNotExist(execErr.Err) {
			return ""
		}
		panic(err)
	}
	path, err = filepath.Abs(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ""
		}
		panic(err)
	}
	return path
}

func dockerInitSha1(target string) string {
	f, err := os.Open(target)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha1.New()
	_, err = io.Copy(h, f)
	if err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

func isValidDockerInitPath(target string, selfPath string) bool { // target and selfPath should be absolute (InitPath and SelfPath already do this)
	if target == "" {
		return false
	}
	if IAMSTATIC {
		if selfPath == "" {
			return false
		}
		if target == selfPath {
			return true
		}
		targetFileInfo, err := os.Lstat(target)
		if err != nil {
			return false
		}
		selfPathFileInfo, err := os.Lstat(selfPath)
		if err != nil {
			return false
		}
		return os.SameFile(targetFileInfo, selfPathFileInfo)
	}
	return INITSHA1 != "" && dockerInitSha1(target) == INITSHA1
}

// Figure out the path of our dockerinit (which may be SelfPath())
func DockerInitPath(localCopy string) string {
	selfPath := SelfPath()
	if isValidDockerInitPath(selfPath, selfPath) {
		// if we're valid, don't bother checking anything else
		return selfPath
	}
	var possibleInits = []string{
		localCopy,
		INITPATH,
		filepath.Join(filepath.Dir(selfPath), "dockerinit"),

		// FHS 3.0 Draft: "/usr/libexec includes internal binaries that are not intended to be executed directly by users or shell scripts. Applications may use a single subdirectory under /usr/libexec."
		// http://www.linuxbase.org/betaspecs/fhs/fhs.html#usrlibexec
		"/usr/libexec/docker/dockerinit",
		"/usr/local/libexec/docker/dockerinit",

		// FHS 2.3: "/usr/lib includes object files, libraries, and internal binaries that are not intended to be executed directly by users or shell scripts."
		// http://refspecs.linuxfoundation.org/FHS_2.3/fhs-2.3.html#USRLIBLIBRARIESFORPROGRAMMINGANDPA
		"/usr/lib/docker/dockerinit",
		"/usr/local/lib/docker/dockerinit",
	}
	for _, dockerInit := range possibleInits {
		if dockerInit == "" {
			continue
		}
		path, err := exec.LookPath(dockerInit)
		if err == nil {
			path, err = filepath.Abs(path)
			if err != nil {
				// LookPath already validated that this file exists and is executable (following symlinks), so how could Abs fail?
				panic(err)
			}
			if isValidDockerInitPath(path, selfPath) {
				return path
			}
		}
	}
	return ""
}

type NopWriter struct{}

func (*NopWriter) Write(buf []byte) (int, error) {
	return len(buf), nil
}

type nopWriteCloser struct {
	io.Writer
}

func (w *nopWriteCloser) Close() error { return nil }

func NopWriteCloser(w io.Writer) io.WriteCloser {
	return &nopWriteCloser{w}
}

type bufReader struct {
	sync.Mutex
	buf    *bytes.Buffer
	reader io.Reader
	err    error
	wait   sync.Cond
}

func NewBufReader(r io.Reader) *bufReader {
	reader := &bufReader{
		buf:    &bytes.Buffer{},
		reader: r,
	}
	reader.wait.L = &reader.Mutex
	go reader.drain()
	return reader
}

func (r *bufReader) drain() {
	buf := make([]byte, 1024)
	for {
		n, err := r.reader.Read(buf)
		r.Lock()
		if err != nil {
			r.err = err
		} else {
			r.buf.Write(buf[0:n])
		}
		r.wait.Signal()
		r.Unlock()
		if err != nil {
			break
		}
	}
}

func (r *bufReader) Read(p []byte) (n int, err error) {
	r.Lock()
	defer r.Unlock()
	for {
		n, err = r.buf.Read(p)
		if n > 0 {
			return n, err
		}
		if r.err != nil {
			return 0, r.err
		}
		r.wait.Wait()
	}
}

func (r *bufReader) Close() error {
	closer, ok := r.reader.(io.ReadCloser)
	if !ok {
		return nil
	}
	return closer.Close()
}

type WriteBroadcaster struct {
	sync.Mutex
	buf     *bytes.Buffer
	writers map[StreamWriter]bool
}

type StreamWriter struct {
	wc     io.WriteCloser
	stream string
}

func (w *WriteBroadcaster) AddWriter(writer io.WriteCloser, stream string) {
	w.Lock()
	sw := StreamWriter{wc: writer, stream: stream}
	w.writers[sw] = true
	w.Unlock()
}

type JSONLog struct {
	Log     string    `json:"log,omitempty"`
	Stream  string    `json:"stream,omitempty"`
	Created time.Time `json:"time"`
}

func (w *WriteBroadcaster) Write(p []byte) (n int, err error) {
	w.Lock()
	defer w.Unlock()
	w.buf.Write(p)
	for sw := range w.writers {
		lp := p
		if sw.stream != "" {
			lp = nil
			for {
				line, err := w.buf.ReadString('\n')
				if err != nil {
					w.buf.Write([]byte(line))
					break
				}
				b, err := json.Marshal(&JSONLog{Log: line, Stream: sw.stream, Created: time.Now().UTC()})
				if err != nil {
					// On error, evict the writer
					delete(w.writers, sw)
					continue
				}
				lp = append(lp, b...)
				lp = append(lp, '\n')
			}
		}
		if n, err := sw.wc.Write(lp); err != nil || n != len(lp) {
			// On error, evict the writer
			delete(w.writers, sw)
		}
	}
	return len(p), nil
}

func (w *WriteBroadcaster) CloseWriters() error {
	w.Lock()
	defer w.Unlock()
	for sw := range w.writers {
		sw.wc.Close()
	}
	w.writers = make(map[StreamWriter]bool)
	return nil
}

func NewWriteBroadcaster() *WriteBroadcaster {
	return &WriteBroadcaster{writers: make(map[StreamWriter]bool), buf: bytes.NewBuffer(nil)}
}

func GetTotalUsedFds() int {
	if fds, err := ioutil.ReadDir(fmt.Sprintf("/proc/%d/fd", os.Getpid())); err != nil {
		Errorf("Error opening /proc/%d/fd: %s", os.Getpid(), err)
	} else {
		return len(fds)
	}
	return -1
}

// TruncIndex allows the retrieval of string identifiers by any of their unique prefixes.
// This is used to retrieve image and container IDs by more convenient shorthand prefixes.
type TruncIndex struct {
	index *suffixarray.Index
	ids   map[string]bool
	bytes []byte
}

func NewTruncIndex() *TruncIndex {
	return &TruncIndex{
		index: suffixarray.New([]byte{' '}),
		ids:   make(map[string]bool),
		bytes: []byte{' '},
	}
}

func (idx *TruncIndex) Add(id string) error {
	if strings.Contains(id, " ") {
		return fmt.Errorf("Illegal character: ' '")
	}
	if _, exists := idx.ids[id]; exists {
		return fmt.Errorf("Id already exists: %s", id)
	}
	idx.ids[id] = true
	idx.bytes = append(idx.bytes, []byte(id+" ")...)
	idx.index = suffixarray.New(idx.bytes)
	return nil
}

func (idx *TruncIndex) Delete(id string) error {
	if _, exists := idx.ids[id]; !exists {
		return fmt.Errorf("No such id: %s", id)
	}
	before, after, err := idx.lookup(id)
	if err != nil {
		return err
	}
	delete(idx.ids, id)
	idx.bytes = append(idx.bytes[:before], idx.bytes[after:]...)
	idx.index = suffixarray.New(idx.bytes)
	return nil
}

func (idx *TruncIndex) lookup(s string) (int, int, error) {
	offsets := idx.index.Lookup([]byte(" "+s), -1)
	//log.Printf("lookup(%s): %v (index bytes: '%s')\n", s, offsets, idx.index.Bytes())
	if offsets == nil || len(offsets) == 0 || len(offsets) > 1 {
		return -1, -1, fmt.Errorf("No such id: %s", s)
	}
	offsetBefore := offsets[0] + 1
	offsetAfter := offsetBefore + strings.Index(string(idx.bytes[offsetBefore:]), " ")
	return offsetBefore, offsetAfter, nil
}

func (idx *TruncIndex) Get(s string) (string, error) {
	before, after, err := idx.lookup(s)
	//log.Printf("Get(%s) bytes=|%s| before=|%d| after=|%d|\n", s, idx.bytes, before, after)
	if err != nil {
		return "", err
	}
	return string(idx.bytes[before:after]), err
}

// TruncateID returns a shorthand version of a string identifier for convenience.
// A collision with other shorthands is very unlikely, but possible.
// In case of a collision a lookup with TruncIndex.Get() will fail, and the caller
// will need to use a langer prefix, or the full-length Id.
func TruncateID(id string) string {
	shortLen := 12
	if len(id) < shortLen {
		shortLen = len(id)
	}
	return id[:shortLen]
}

// Code c/c from io.Copy() modified to handle escape sequence
func CopyEscapable(dst io.Writer, src io.ReadCloser) (written int64, err error) {
	buf := make([]byte, 32*1024)
	for {
		nr, er := src.Read(buf)
		if nr > 0 {
			// ---- Docker addition
			// char 16 is C-p
			if nr == 1 && buf[0] == 16 {
				nr, er = src.Read(buf)
				// char 17 is C-q
				if nr == 1 && buf[0] == 17 {
					if err := src.Close(); err != nil {
						return 0, err
					}
					return 0, nil
				}
			}
			// ---- End of docker
			nw, ew := dst.Write(buf[0:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if ew != nil {
				err = ew
				break
			}
			if nr != nw {
				err = io.ErrShortWrite
				break
			}
		}
		if er == io.EOF {
			break
		}
		if er != nil {
			err = er
			break
		}
	}
	return written, err
}

func HashData(src io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, src); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

type KernelVersionInfo struct {
	Kernel int
	Major  int
	Minor  int
	Flavor string
}

func (k *KernelVersionInfo) String() string {
	flavor := ""
	if len(k.Flavor) > 0 {
		flavor = fmt.Sprintf("-%s", k.Flavor)
	}
	return fmt.Sprintf("%d.%d.%d%s", k.Kernel, k.Major, k.Minor, flavor)
}

// Compare two KernelVersionInfo struct.
// Returns -1 if a < b, = if a == b, 1 it a > b
func CompareKernelVersion(a, b *KernelVersionInfo) int {
	if a.Kernel < b.Kernel {
		return -1
	} else if a.Kernel > b.Kernel {
		return 1
	}

	if a.Major < b.Major {
		return -1
	} else if a.Major > b.Major {
		return 1
	}

	if a.Minor < b.Minor {
		return -1
	} else if a.Minor > b.Minor {
		return 1
	}

	return 0
}

func GetKernelVersion() (*KernelVersionInfo, error) {
	var (
		err error
	)

	uts, err := uname()
	if err != nil {
		return nil, err
	}

	release := make([]byte, len(uts.Release))

	i := 0
	for _, c := range uts.Release {
		release[i] = byte(c)
		i++
	}

	// Remove the \x00 from the release for Atoi to parse correctly
	release = release[:bytes.IndexByte(release, 0)]

	return ParseRelease(string(release))
}

func ParseRelease(release string) (*KernelVersionInfo, error) {
	var (
		flavor               string
		kernel, major, minor int
		err                  error
	)

	tmp := strings.SplitN(release, "-", 2)
	tmp2 := strings.Split(tmp[0], ".")

	if len(tmp2) > 0 {
		kernel, err = strconv.Atoi(tmp2[0])
		if err != nil {
			return nil, err
		}
	}

	if len(tmp2) > 1 {
		major, err = strconv.Atoi(tmp2[1])
		if err != nil {
			return nil, err
		}
	}

	if len(tmp2) > 2 {
		// Removes "+" because git kernels might set it
		minorUnparsed := strings.Trim(tmp2[2], "+")
		minor, err = strconv.Atoi(minorUnparsed)
		if err != nil {
			return nil, err
		}
	}

	if len(tmp) == 2 {
		flavor = tmp[1]
	} else {
		flavor = ""
	}

	return &KernelVersionInfo{
		Kernel: kernel,
		Major:  major,
		Minor:  minor,
		Flavor: flavor,
	}, nil
}

// FIXME: this is deprecated by CopyWithTar in archive.go
func CopyDirectory(source, dest string) error {
	if output, err := exec.Command("cp", "-ra", source, dest).CombinedOutput(); err != nil {
		return fmt.Errorf("Error copy: %s (%s)", err, output)
	}
	return nil
}

type NopFlusher struct{}

func (f *NopFlusher) Flush() {}

type WriteFlusher struct {
	sync.Mutex
	w       io.Writer
	flusher http.Flusher
}

func (wf *WriteFlusher) Write(b []byte) (n int, err error) {
	wf.Lock()
	defer wf.Unlock()
	n, err = wf.w.Write(b)
	wf.flusher.Flush()
	return n, err
}

// Flush the stream immediately.
func (wf *WriteFlusher) Flush() {
	wf.Lock()
	defer wf.Unlock()
	wf.flusher.Flush()
}

func NewWriteFlusher(w io.Writer) *WriteFlusher {
	var flusher http.Flusher
	if f, ok := w.(http.Flusher); ok {
		flusher = f
	} else {
		flusher = &NopFlusher{}
	}
	return &WriteFlusher{w: w, flusher: flusher}
}

func IsURL(str string) bool {
	return strings.HasPrefix(str, "http://") || strings.HasPrefix(str, "https://")
}

func IsGIT(str string) bool {
	return strings.HasPrefix(str, "git://") || strings.HasPrefix(str, "github.com/")
}

// GetResolvConf opens and read the content of /etc/resolv.conf.
// It returns it as byte slice.
func GetResolvConf() ([]byte, error) {
	resolv, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		Errorf("Error openning resolv.conf: %s", err)
		return nil, err
	}
	return resolv, nil
}

// CheckLocalDns looks into the /etc/resolv.conf,
// it returns true if there is a local nameserver or if there is no nameserver.
func CheckLocalDns(resolvConf []byte) bool {
	var parsedResolvConf = StripComments(resolvConf, []byte("#"))
	if !bytes.Contains(parsedResolvConf, []byte("nameserver")) {
		return true
	}
	for _, ip := range [][]byte{
		[]byte("127.0.0.1"),
		[]byte("127.0.1.1"),
	} {
		if bytes.Contains(parsedResolvConf, ip) {
			return true
		}
	}
	return false
}

// StripComments parses input into lines and strips away comments.
func StripComments(input []byte, commentMarker []byte) []byte {
	lines := bytes.Split(input, []byte("\n"))
	var output []byte
	for _, currentLine := range lines {
		var commentIndex = bytes.Index(currentLine, commentMarker)
		if commentIndex == -1 {
			output = append(output, currentLine...)
		} else {
			output = append(output, currentLine[:commentIndex]...)
		}
		output = append(output, []byte("\n")...)
	}
	return output
}

// GetNameserversAsCIDR returns nameservers (if any) listed in
// /etc/resolv.conf as CIDR blocks (e.g., "1.2.3.4/32")
// This function's output is intended for net.ParseCIDR
func GetNameserversAsCIDR(resolvConf []byte) []string {
	var parsedResolvConf = StripComments(resolvConf, []byte("#"))
	nameservers := []string{}
	re := regexp.MustCompile(`^\s*nameserver\s*(([0-9]+\.){3}([0-9]+))\s*$`)
	for _, line := range bytes.Split(parsedResolvConf, []byte("\n")) {
		var ns = re.FindSubmatch(line)
		if len(ns) > 0 {
			nameservers = append(nameservers, string(ns[1])+"/32")
		}
	}

	return nameservers
}

// FIXME: Change this not to receive default value as parameter
func ParseHost(defaultHost string, defaultPort int, defaultUnix, addr string) (string, error) {
	var (
		proto string
		host  string
		port  int
	)
	addr = strings.TrimSpace(addr)
	switch {
	case strings.HasPrefix(addr, "unix://"):
		proto = "unix"
		addr = strings.TrimPrefix(addr, "unix://")
		if addr == "" {
			addr = defaultUnix
		}
	case strings.HasPrefix(addr, "tcp://"):
		proto = "tcp"
		addr = strings.TrimPrefix(addr, "tcp://")
	case addr == "":
		proto = "unix"
		addr = defaultUnix
	default:
		if strings.Contains(addr, "://") {
			return "", fmt.Errorf("Invalid bind address protocol: %s", addr)
		}
		proto = "tcp"
	}

	if proto != "unix" && strings.Contains(addr, ":") {
		hostParts := strings.Split(addr, ":")
		if len(hostParts) != 2 {
			return "", fmt.Errorf("Invalid bind address format: %s", addr)
		}
		if hostParts[0] != "" {
			host = hostParts[0]
		} else {
			host = defaultHost
		}

		if p, err := strconv.Atoi(hostParts[1]); err == nil && p != 0 {
			port = p
		} else {
			port = defaultPort
		}

	} else {
		host = addr
		port = defaultPort
	}
	if proto == "unix" {
		return fmt.Sprintf("%s://%s", proto, host), nil
	}
	return fmt.Sprintf("%s://%s:%d", proto, host, port), nil
}

func GetReleaseVersion() string {
	resp, err := http.Get("http://get.docker.io/latest")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.ContentLength > 24 || resp.StatusCode != 200 {
		return ""
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(body))
}

// Get a repos name and returns the right reposName + tag
// The tag can be confusing because of a port in a repository name.
//     Ex: localhost.localdomain:5000/samalba/hipache:latest
func ParseRepositoryTag(repos string) (string, string) {
	n := strings.LastIndex(repos, ":")
	if n < 0 {
		return repos, ""
	}
	if tag := repos[n+1:]; !strings.Contains(tag, "/") {
		return repos[:n], tag
	}
	return repos, ""
}

type User struct {
	Uid      string // user id
	Gid      string // primary group id
	Username string
	Name     string
	HomeDir  string
}

// UserLookup check if the given username or uid is present in /etc/passwd
// and returns the user struct.
// If the username is not found, an error is returned.
func UserLookup(uid string) (*User, error) {
	file, err := ioutil.ReadFile("/etc/passwd")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(file), "\n") {
		data := strings.Split(line, ":")
		if len(data) > 5 && (data[0] == uid || data[2] == uid) {
			return &User{
				Uid:      data[2],
				Gid:      data[3],
				Username: data[0],
				Name:     data[4],
				HomeDir:  data[5],
			}, nil
		}
	}
	return nil, fmt.Errorf("User not found in /etc/passwd")
}

type DependencyGraph struct {
	nodes map[string]*DependencyNode
}

type DependencyNode struct {
	id   string
	deps map[*DependencyNode]bool
}

func NewDependencyGraph() DependencyGraph {
	return DependencyGraph{
		nodes: map[string]*DependencyNode{},
	}
}

func (graph *DependencyGraph) addNode(node *DependencyNode) string {
	if graph.nodes[node.id] == nil {
		graph.nodes[node.id] = node
	}
	return node.id
}

func (graph *DependencyGraph) NewNode(id string) string {
	if graph.nodes[id] != nil {
		return id
	}
	nd := &DependencyNode{
		id:   id,
		deps: map[*DependencyNode]bool{},
	}
	graph.addNode(nd)
	return id
}

func (graph *DependencyGraph) AddDependency(node, to string) error {
	if graph.nodes[node] == nil {
		return fmt.Errorf("Node %s does not belong to this graph", node)
	}

	if graph.nodes[to] == nil {
		return fmt.Errorf("Node %s does not belong to this graph", to)
	}

	if node == to {
		return fmt.Errorf("Dependency loops are forbidden!")
	}

	graph.nodes[node].addDependency(graph.nodes[to])
	return nil
}

func (node *DependencyNode) addDependency(to *DependencyNode) bool {
	node.deps[to] = true
	return node.deps[to]
}

func (node *DependencyNode) Degree() int {
	return len(node.deps)
}

// The magic happens here ::
func (graph *DependencyGraph) GenerateTraversalMap() ([][]string, error) {
	Debugf("Generating traversal map. Nodes: %d", len(graph.nodes))
	result := [][]string{}
	processed := map[*DependencyNode]bool{}
	// As long as we haven't processed all nodes...
	for len(processed) < len(graph.nodes) {
		// Use a temporary buffer for processed nodes, otherwise
		// nodes that depend on each other could end up in the same round.
		tmpProcessed := []*DependencyNode{}
		for _, node := range graph.nodes {
			// If the node has more dependencies than what we have cleared,
			// it won't be valid for this round.
			if node.Degree() > len(processed) {
				continue
			}
			// If it's already processed, get to the next one
			if processed[node] {
				continue
			}
			// It's not been processed yet and has 0 deps. Add it!
			// (this is a shortcut for what we're doing below)
			if node.Degree() == 0 {
				tmpProcessed = append(tmpProcessed, node)
				continue
			}
			// If at least one dep hasn't been processed yet, we can't
			// add it.
			ok := true
			for dep := range node.deps {
				if !processed[dep] {
					ok = false
					break
				}
			}
			// All deps have already been processed. Add it!
			if ok {
				tmpProcessed = append(tmpProcessed, node)
			}
		}
		Debugf("Round %d: found %d available nodes", len(result), len(tmpProcessed))
		// If no progress has been made this round,
		// that means we have circular dependencies.
		if len(tmpProcessed) == 0 {
			return nil, fmt.Errorf("Could not find a solution to this dependency graph")
		}
		round := []string{}
		for _, nd := range tmpProcessed {
			round = append(round, nd.id)
			processed[nd] = true
		}
		result = append(result, round)
	}
	return result, nil
}

// An StatusError reports an unsuccessful exit by a command.
type StatusError struct {
	Status     string
	StatusCode int
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("Status: %s, Code: %d", e.Status, e.StatusCode)
}

func quote(word string, buf *bytes.Buffer) {
	// Bail out early for "simple" strings
	if word != "" && !strings.ContainsAny(word, "\\'\"`${[|&;<>()~*?! \t\n") {
		buf.WriteString(word)
		return
	}

	buf.WriteString("'")

	for i := 0; i < len(word); i++ {
		b := word[i]
		if b == '\'' {
			// Replace literal ' with a close ', a \', and a open '
			buf.WriteString("'\\''")
		} else {
			buf.WriteByte(b)
		}
	}

	buf.WriteString("'")
}

// Take a list of strings and escape them so they will be handled right
// when passed as arguments to an program via a shell
func ShellQuoteArguments(args []string) string {
	var buf bytes.Buffer
	for i, arg := range args {
		if i != 0 {
			buf.WriteByte(' ')
		}
		quote(arg, &buf)
	}
	return buf.String()
}

func IsClosedError(err error) bool {
	/* This comparison is ugly, but unfortunately, net.go doesn't export errClosing.
	 * See:
	 * http://golang.org/src/pkg/net/net.go
	 * https://code.google.com/p/go/issues/detail?id=4337
	 * https://groups.google.com/forum/#!msg/golang-nuts/0_aaCvBmOcM/SptmDyX1XJMJ
	 */
	return strings.HasSuffix(err.Error(), "use of closed network connection")
}

func PartParser(template, data string) (map[string]string, error) {
	// ip:public:private
	var (
		templateParts = strings.Split(template, ":")
		parts         = strings.Split(data, ":")
		out           = make(map[string]string, len(templateParts))
	)
	if len(parts) != len(templateParts) {
		return nil, fmt.Errorf("Invalid format to parse.  %s should match template %s", data, template)
	}

	for i, t := range templateParts {
		value := ""
		if len(parts) > i {
			value = parts[i]
		}
		out[t] = value
	}
	return out, nil
}

var globalTestID string

// GetCallerName introspects the call stack and returns the name of the
// function `depth` levels down in the stack.
func GetCallerName(depth int) string {
	// Use the caller function name as a prefix.
	// This helps trace temp directories back to their test.
	pc, _, _, _ := runtime.Caller(depth + 1)
	callerLongName := runtime.FuncForPC(pc).Name()
	parts := strings.Split(callerLongName, ".")
	callerShortName := parts[len(parts)-1]
	return callerShortName
}

func CopyFile(src, dst string) (int64, error) {
	if src == dst {
		return 0, nil
	}
	sf, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer sf.Close()
	if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
		return 0, err
	}
	df, err := os.Create(dst)
	if err != nil {
		return 0, err
	}
	defer df.Close()
	return io.Copy(df, sf)
}
