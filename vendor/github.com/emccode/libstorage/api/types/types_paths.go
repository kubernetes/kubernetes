package types

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"runtime"
	"sync"
	"time"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gotil"
)

type fileKey int

const (

	// Home is the application home directory.
	Home fileKey = iota

	minDirKey

	// Etc is the application etc directory.
	Etc

	// Lib is the application lib directory.
	Lib

	// Log is the application log directory.
	Log

	// Run is the application run directory.
	Run

	maxDirKey

	minFileKey

	// LSX is the path to the libStorage executor.
	LSX

	maxFileKey
)

func (k fileKey) isFileKeyMarker() bool {
	return k == minDirKey || k == maxDirKey ||
		k == minFileKey || k == maxFileKey
}

// Exists returns a flag indicating whether or not the file/directory exists.
func (k fileKey) Exists() bool {
	return gotil.FileExists(k.String())
}

// Format may call Sprint(f) or Fprint(f) etc. to generate its output.
func (k fileKey) Format(f fmt.State, c rune) {
	fs := &bytes.Buffer{}
	fs.WriteRune('%')
	if f.Flag('+') {
		fs.WriteRune('+')
	}
	if f.Flag('-') {
		fs.WriteRune('-')
	}
	if f.Flag('#') {
		fs.WriteRune('#')
	}
	if f.Flag(' ') {
		fs.WriteRune(' ')
	}
	if f.Flag('0') {
		fs.WriteRune('0')
	}
	if w, ok := f.Width(); ok {
		fs.WriteString(fmt.Sprintf("%d", w))
	}
	if p, ok := f.Precision(); ok {
		fs.WriteString(fmt.Sprintf("%d", p))
	}
	var (
		s  string
		cc = c
	)
	if c == 'k' {
		s = k.key()
		cc = 's'
	} else {
		s = k.String()
	}
	fs.WriteRune(cc)
	fmt.Fprintf(f, fs.String(), s)
}

func (k fileKey) parent() fileKey {
	if k < maxDirKey {
		return Home
	}
	return Lib
}

func (k fileKey) perms() os.FileMode {
	if k <= LSX {
		return 0755
	}
	return 0644
}

func (k fileKey) key() string {
	switch k {
	case Home:
		return "home"
	case Etc:
		return "etc"
	case Lib:
		return "lib"
	case Log:
		return "log"
	case Run:
		return "run"
	case LSX:
		return "lsx"
	}
	return ""
}

func (k fileKey) defaultVal() string {
	switch k {
	case Home:
		return libstorageHome
	case Etc:
		return "/etc/libstorage"
	case Lib:
		return "/var/lib/libstorage"
	case Log:
		return "/var/log/libstorage"
	case Run:
		return "/var/run/libstorage"
	case LSX:
		switch runtime.GOOS {
		case "linux", "darwin":
			return fmt.Sprintf("lsx-%s", runtime.GOOS)
		case "windows":
			return "lsx-windows.exe"
		}
	}
	return ""
}

func (k fileKey) get() string {
	if v, ok := fileKeyCache[k]; ok {
		return v
	}
	if k == Home {
		return libstorageHome
	}

	if p, ok := fileKeyCache[k.parent()]; ok {
		return path.Join(p, k.defaultVal())
	}
	return k.defaultVal()
}

// Join concatenates the filePath's fully-qualified path with the provided
// path elements using the local operating system's path separator.
func (k fileKey) Join(elem ...string) string {
	log.WithField("elem", elem).Debug("enter join")

	var elems []string
	if _, ok := fileKeyCache[k]; !ok {
		elems = []string{Home.String()}
	}
	if k != Home {
		elems = append(elems, k.String())
	}
	elems = append(elems, elem...)
	log.WithField("elem", elems).Debug("exit join")
	return path.Join(elems...)
}

// Name returns the last part of the fileKey's fully-qualified path.
func (k fileKey) Name() string {
	return path.Base(k.Path())
}

// Path returns the fileKey's fully-qualified path.
func (k fileKey) Path() string {
	if k == Home {
		homeLock.Lock()
		defer homeLock.Unlock()
	}

	if v, ok := fileKeyCache[k]; ok {
		return v
	}

	log.WithFields(log.Fields{
		"key": k.key(),
	}).Debug("must init path")

	k.init()
	k.cache()

	return fileKeyCache[k]
}

// String delegates to the fileKey's Path function.
func (k fileKey) String() string {
	return k.Path()
}

func (k fileKey) cache() {
	fileKeyCache[k] = k.get()
	log.WithFields(log.Fields{
		"key":  k.key(),
		"path": k.get(),
	}).Debug("cached key")
}

func (k fileKey) init() {

	if k == Home {
		if !checkPerms(k, false) {
			failedPath := k.get()
			libstorageHome = path.Join(gotil.HomeDir(), ".libstorage")
			log.WithFields(log.Fields{
				"failedPath": failedPath,
				"newPath":    k.get(),
			}).Debug("first make homedir failed, trying again")
			checkPerms(k, true)
		}
		return
	}

	checkPerms(k, true)
}

var (
	libstorageHome = os.Getenv("LIBSTORAGE_HOME")
	fileKeyCache   = map[fileKey]string{}
	homeLock       = &sync.Mutex{}
	thisExeDir     string
	thisExeName    string
	thisExeAbsPath string
	slashRX        = regexp.MustCompile(`^((?:/)|(?:[a-zA-Z]\:\\?))?$`)
)

func initPaths() {

	thisExeDir, thisExeName, thisExeAbsPath = gotil.GetThisPathParts()

	if libstorageHome == "" {
		libstorageHome = "/"
	}

	// if not root and home is /, change home to user's home dir
	if os.Geteuid() != 0 && libstorageHome == "/" {
		libstorageHome = path.Join(gotil.HomeDir(), ".libstorage")
	}

	for i := Home; i < maxFileKey; i++ {
		if i.isFileKeyMarker() {
			continue
		}

		// this initialized the value
		_ = i.String()

		if Debug {
			log.WithField(i.key(), i.String()).Info("libStorage path")
		}
	}
}

func checkPerms(k fileKey, mustPerm bool) bool {
	if k > maxDirKey {
		return true
	}

	p := k.get()

	fields := log.Fields{
		"path":     p,
		"perms":    k.perms(),
		"mustPerm": mustPerm,
	}

	if gotil.FileExists(p) {
		if log.GetLevel() == log.DebugLevel {
			log.WithField("path", p).Debug("file exists")
		}
	} else {
		log.WithFields(fields).Info("making libStorage directory")
		noPermMkdirErr := fmt.Sprintf("mkdir %s: permission denied", p)
		if err := os.MkdirAll(p, k.perms()); err != nil {
			if err.Error() == noPermMkdirErr {
				if mustPerm {
					log.WithFields(fields).Panic(noPermMkdirErr)
				}
				return false
			}
		}
	}

	touchFilePath := path.Join(p, fmt.Sprintf(".touch-%v", time.Now().Unix()))
	defer os.RemoveAll(touchFilePath)

	noPermTouchErr := fmt.Sprintf("open %s: permission denied", touchFilePath)

	if _, err := os.Create(touchFilePath); err != nil {
		if err.Error() == noPermTouchErr {
			if mustPerm {
				log.WithFields(fields).Panic(noPermTouchErr)
			}
			return false
		}
	}

	return true
}

// logFile returns a writer to a file inside the log directory with the
// provided file name.
func logFile(fileName string) (io.Writer, error) {
	return os.OpenFile(
		Log.Join(fileName), os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
}

// stdOutAndLogFile returns a mutltiplexed writer for the current process's
// stdout descriptor and alog file with the provided name.
func stdOutAndLogFile(fileName string) (io.Writer, error) {
	lf, lfErr := logFile(fileName)
	if lfErr != nil {
		return nil, lfErr
	}
	return io.MultiWriter(os.Stdout, lf), nil
}
