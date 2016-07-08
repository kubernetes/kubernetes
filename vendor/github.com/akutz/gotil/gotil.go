package gotil

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/akutz/goof"
	"github.com/kardianos/osext"
)

const (
	trimPattern          = `(?s)^\s*(.*?)\s*$`
	networkAdressPattern = `(?i)^((?:(?:tcp|udp|ip)[46]?)|(?:unix(?:gram|packet)?))://(.+)$`

	letterBytes     = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	letterIndexBits = 6
	letterIndexMask = 1<<letterIndexBits - 1
	letterIndexMax  = 63 / letterIndexBits
)

var (
	homeDir    string
	homeDirSet bool
	trimRx     *regexp.Regexp
	netAddrRx  *regexp.Regexp
)

func init() {
	trimRx = regexp.MustCompile(trimPattern)
	netAddrRx = regexp.MustCompile(networkAdressPattern)
}

// StringInSlice returns a flag indicating whether or not a provided string
// exists in a string slice. The string comparison is case-insensitive.
func StringInSlice(a string, list []string) bool {
	for _, b := range list {
		if strings.ToLower(a) == strings.ToLower(b) {
			return true
		}
	}
	return false
}

// StringInSliceCS returns a flag indicating whether or not a provided string
// exists in a string slice. The string comparison is case-sensitive.
func StringInSliceCS(a string, list []string) bool {
	for _, b := range list {
		if a == b {
			return true
		}
	}
	return false
}

// WriteStringToFile writes the string to the file at the provided path.
func WriteStringToFile(text, path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0644)
	defer f.Close()

	if err != nil {
		return err
	}

	f.WriteString(text)
	return nil
}

// ReadFileToString reads the file at the provided path to a string.
func ReadFileToString(path string) (string, error) {

	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Scan()

	return scanner.Text(), nil
}

// IsDirEmpty returns a flag indicating whether or not a directory has any
// child objects such as files or directories in it.
func IsDirEmpty(name string) (bool, error) {
	f, err := os.Open(name)
	if err != nil {
		return false, err
	}
	defer f.Close()

	_, err = f.Readdir(1)
	if err == io.EOF {
		return true, nil
	}
	return false, err
}

func lineReader(f func() (io.Reader, func(), error)) (<-chan string, error) {

	r, done, err := f()
	if err != nil {
		return nil, err
	}

	c := make(chan string)
	go func() {
		if r != nil {
			s := bufio.NewScanner(r)
			for s.Scan() {
				c <- s.Text()
			}
		}
		close(c)
		if done != nil {
			done()
		}
	}()

	return c, nil
}

// LineReader returns a channel that reads the contents of a reader
// line-by-line.
func LineReader(r io.Reader) (<-chan string, error) {
	return lineReader(func() (io.Reader, func(), error) { return r, nil, nil })
}

// LineReaderFrom returns a channel that reads the contents of a file
// line-by-line.
func LineReaderFrom(path string) (<-chan string, error) {
	return lineReader(func() (io.Reader, func(), error) {
		if !FileExists(path) {
			return nil, nil, nil
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, nil, err
		}
		return f, func() { f.Close() }, nil
	})
}

// FileExists returns a flag indicating whether a provided file path exists.
func FileExists(filePath string) bool {
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		return true
	}
	return false
}

// FileExistsInPath returns a flag indicating whether the provided file exists
// in the current path.
func FileExistsInPath(fileName string) bool {
	_, err := exec.LookPath(fileName)
	return err == nil
}

// GetPathParts returns the absolute directory path, the file name, and the
// absolute path of the provided path string.
func GetPathParts(path string) (dirPath, fileName, absPath string) {
	lookup, lookupErr := exec.LookPath(path)
	if lookupErr == nil {
		path = lookup
	}
	absPath, _ = filepath.Abs(path)
	dirPath = filepath.Dir(absPath)
	fileName = filepath.Base(absPath)
	return
}

// GetThisPathParts returns the same information as GetPathParts for the
// current executable.
func GetThisPathParts() (dirPath, fileName, absPath string) {
	exeFile, _ := osext.Executable()
	return GetPathParts(exeFile)
}

// RandomString generates a random set of characters with the given lenght.
func RandomString(length int) string {
	src := rand.NewSource(time.Now().UnixNano())
	b := make([]byte, length)
	for i, cache, remain := length-1, src.Int63(), letterIndexMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIndexMax
		}
		if idx := int(cache & letterIndexMask); idx < len(letterBytes) {
			b[i] = letterBytes[idx]
			i--
		}
		cache >>= letterIndexBits
		remain--
	}

	return string(b)
}

// GetLocalIP returns the non loopback local IP of the host
func GetLocalIP() (ip string) {
	addrs, _ := net.InterfaceAddrs()
	for _, address := range addrs {
		// check the address type and if it is not a loopback the display it
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				ip = ipnet.IP.String()
				break
			}
		}
	}
	return
}

// ParseAddress parses a standard golang network address and returns the
// protocol and path.
func ParseAddress(addr string) (proto string, path string, err error) {
	m := netAddrRx.FindStringSubmatch(addr)
	if m == nil {
		return "", "", goof.WithField("address", addr, "invalid address")
	}
	return m[1], m[2], nil
}

// Trim removes all leading and trailing whitespace, including tab, newline,
// and carriage return characters.
func Trim(text string) string {
	return trimRx.FindStringSubmatch(text)[1]
}

// WriteIndentedN indents all lines n spaces.
func WriteIndentedN(w io.Writer, b []byte, n int) error {
	s := bufio.NewScanner(bytes.NewReader(b))
	if !s.Scan() {
		return nil
	}
	l := s.Text()
	for {
		for x := 0; x < n; x++ {
			if _, err := fmt.Fprint(w, " "); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprint(w, l); err != nil {
			return err
		}
		if !s.Scan() {
			break
		}
		l = s.Text()
		if _, err := fmt.Fprint(w, "\n"); err != nil {
			return err
		}
	}
	return nil
}

// WriteIndented indents all lines four spaces.
func WriteIndented(w io.Writer, b []byte) error {
	return WriteIndentedN(w, b, 4)
}

// HomeDir returns the home directory of the user that owns the current process.
func HomeDir() string {
	if homeDirSet {
		return homeDir
	}
	if user, err := user.Current(); err == nil {
		homeDir = user.HomeDir
	}
	homeDirSet = true
	return homeDir
}

const (
	minTCPPort         = 0
	maxTCPPort         = 65535
	maxReservedTCPPort = 1024
	maxRandTCPPort     = maxTCPPort - (maxReservedTCPPort + 1)
)

var (
	tcpPortRand = rand.New(rand.NewSource(time.Now().UnixNano()))
)

// IsTCPPortAvailable returns a flag indicating whether or not a TCP port is
// available.
func IsTCPPortAvailable(port int) bool {
	if port < minTCPPort || port > maxTCPPort {
		return false
	}
	conn, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// RandomTCPPort gets a free, random TCP port between 1025-65535. If no free
// ports are available -1 is returned.
func RandomTCPPort() int {
	for i := maxReservedTCPPort; i < maxTCPPort; i++ {
		p := tcpPortRand.Intn(maxRandTCPPort) + maxReservedTCPPort + 1
		if IsTCPPortAvailable(p) {
			return p
		}
	}
	return -1
}
