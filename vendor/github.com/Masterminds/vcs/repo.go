// Package vcs provides the ability to work with varying version control systems
// (VCS),  also known as source control systems (SCM) though the same interface.
//
// This package includes a function that attempts to detect the repo type from
// the remote URL and return the proper type. For example,
//
//     remote := "https://github.com/Masterminds/vcs"
//     local, _ := ioutil.TempDir("", "go-vcs")
//     repo, err := NewRepo(remote, local)
//
// In this case repo will be a GitRepo instance. NewRepo can detect the VCS for
// numerous popular VCS and from the URL. For example, a URL ending in .git
// that's not from one of the popular VCS will be detected as a Git repo and
// the correct type will be returned.
//
// If you know the repository type and would like to create an instance of a
// specific type you can use one of constructors for a type. They are NewGitRepo,
// NewSvnRepo, NewBzrRepo, and NewHgRepo. The definition and usage is the same
// as NewRepo.
//
// Once you have an object implementing the Repo interface the operations are
// the same no matter which VCS you're using. There are some caveats. For
// example, each VCS has its own version formats that need to be respected and
// checkout out branches, if a branch is being worked with, is different in
// each VCS.
package vcs

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// Logger is where you can provide a logger, implementing the log.Logger interface,
// where verbose output from each VCS will be written. The default logger does
// not log data. To log data supply your own logger or change the output location
// of the provided logger.
var Logger *log.Logger

func init() {
	// Initialize the logger to one that does not actually log anywhere. This is
	// to be overridden by the package user by setting vcs.Logger to a different
	// logger.
	Logger = log.New(ioutil.Discard, "go-vcs", log.LstdFlags)
}

const longForm = "2006-01-02 15:04:05 -0700"

// Type describes the type of VCS
type Type string

// VCS types
const (
	NoVCS Type = ""
	Git   Type = "git"
	Svn   Type = "svn"
	Bzr   Type = "bzr"
	Hg    Type = "hg"
)

// Repo provides an interface to work with repositories using different source
// control systems such as Git, Bzr, Mercurial, and SVN. For implementations
// of this interface see BzrRepo, GitRepo, HgRepo, and SvnRepo.
type Repo interface {

	// Vcs retrieves the underlying VCS being implemented.
	Vcs() Type

	// Remote retrieves the remote location for a repo.
	Remote() string

	// LocalPath retrieves the local file system location for a repo.
	LocalPath() string

	// Get is used to perform an initial clone/checkout of a repository.
	Get() error

	// Initializes a new repository locally.
	Init() error

	// Update performs an update to an existing checkout of a repository.
	Update() error

	// UpdateVersion sets the version of a package of a repository.
	UpdateVersion(string) error

	// Version retrieves the current version.
	Version() (string, error)

	// Current retrieves the current version-ish. This is different from the
	// Version method. The output could be a branch name if on the tip of a
	// branch (git), a tag if on a tag, a revision if on a specific revision
	// that's not the tip of the branch. The values here vary based on the VCS.
	Current() (string, error)

	// Date retrieves the date on the latest commit.
	Date() (time.Time, error)

	// CheckLocal verifies the local location is of the correct VCS type
	CheckLocal() bool

	// Branches returns a list of available branches on the repository.
	Branches() ([]string, error)

	// Tags returns a list of available tags on the repository.
	Tags() ([]string, error)

	// IsReference returns if a string is a reference. A reference can be a
	// commit id, branch, or tag.
	IsReference(string) bool

	// IsDirty returns if the checkout has been modified from the checked
	// out reference.
	IsDirty() bool

	// CommitInfo retrieves metadata about a commit.
	CommitInfo(string) (*CommitInfo, error)

	// TagsFromCommit retrieves tags from a commit id.
	TagsFromCommit(string) ([]string, error)

	// Ping returns if remote location is accessible.
	Ping() bool

	// RunFromDir executes a command from repo's directory.
	RunFromDir(cmd string, args ...string) ([]byte, error)

	// CmdFromDir creates a new command that will be executed from repo's
	// directory.
	CmdFromDir(cmd string, args ...string) *exec.Cmd

	// ExportDir exports the current revision to the passed in directory.
	ExportDir(string) error
}

// NewRepo returns a Repo based on trying to detect the source control from the
// remote and local locations. The appropriate implementation will be returned
// or an ErrCannotDetectVCS if the VCS type cannot be detected.
// Note, this function may make calls to the Internet to determind help determine
// the VCS.
func NewRepo(remote, local string) (Repo, error) {
	vtype, remote, err := detectVcsFromRemote(remote)

	// From the remote URL the VCS could not be detected. See if the local
	// repo contains enough information to figure out the VCS. The reason the
	// local repo is not checked first is because of the potential for VCS type
	// switches which will be detected in each of the type builders.
	if err == ErrCannotDetectVCS {
		vtype, err = DetectVcsFromFS(local)
	}

	if err != nil {
		return nil, err
	}

	switch vtype {
	case Git:
		return NewGitRepo(remote, local)
	case Svn:
		return NewSvnRepo(remote, local)
	case Hg:
		return NewHgRepo(remote, local)
	case Bzr:
		return NewBzrRepo(remote, local)
	}

	// Should never fall through to here but just in case.
	return nil, ErrCannotDetectVCS
}

// CommitInfo contains metadata about a commit.
type CommitInfo struct {
	// The commit id
	Commit string

	// Who authored the commit
	Author string

	// Date of the commit
	Date time.Time

	// Commit message
	Message string
}

type base struct {
	remote, local string
	Logger        *log.Logger
}

func (b *base) log(v interface{}) {
	b.Logger.Printf("%s", v)
}

// Remote retrieves the remote location for a repo.
func (b *base) Remote() string {
	return b.remote
}

// LocalPath retrieves the local file system location for a repo.
func (b *base) LocalPath() string {
	return b.local
}

func (b *base) setRemote(remote string) {
	b.remote = remote
}

func (b *base) setLocalPath(local string) {
	b.local = local
}

func (b base) run(cmd string, args ...string) ([]byte, error) {
	out, err := exec.Command(cmd, args...).CombinedOutput()
	b.log(out)
	if err != nil {
		err = fmt.Errorf("%s: %s", out, err)
	}
	return out, err
}

func (b *base) CmdFromDir(cmd string, args ...string) *exec.Cmd {
	c := exec.Command(cmd, args...)
	c.Dir = b.local
	c.Env = envForDir(c.Dir)
	return c
}

func (b *base) RunFromDir(cmd string, args ...string) ([]byte, error) {
	c := b.CmdFromDir(cmd, args...)
	out, err := c.CombinedOutput()
	return out, err
}

func (b *base) referenceList(c, r string) []string {
	var out []string
	re := regexp.MustCompile(r)
	for _, m := range re.FindAllStringSubmatch(c, -1) {
		out = append(out, m[1])
	}

	return out
}

func envForDir(dir string) []string {
	env := os.Environ()
	return mergeEnvLists([]string{"PWD=" + dir}, env)
}

func mergeEnvLists(in, out []string) []string {
NextVar:
	for _, inkv := range in {
		k := strings.SplitAfterN(inkv, "=", 2)[0]
		for i, outkv := range out {
			if strings.HasPrefix(outkv, k) {
				out[i] = inkv
				continue NextVar
			}
		}
		out = append(out, inkv)
	}
	return out
}

func depInstalled(name string) bool {
	if _, err := exec.LookPath(name); err != nil {
		return false
	}

	return true
}
