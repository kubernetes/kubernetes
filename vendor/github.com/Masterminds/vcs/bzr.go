package vcs

import (
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var bzrDetectURL = regexp.MustCompile("parent branch: (?P<foo>.+)\n")

// NewBzrRepo creates a new instance of BzrRepo. The remote and local directories
// need to be passed in.
func NewBzrRepo(remote, local string) (*BzrRepo, error) {
	ins := depInstalled("bzr")
	if !ins {
		return nil, NewLocalError("bzr is not installed", nil, "")
	}
	ltype, err := DetectVcsFromFS(local)

	// Found a VCS other than Bzr. Need to report an error.
	if err == nil && ltype != Bzr {
		return nil, ErrWrongVCS
	}

	r := &BzrRepo{}
	r.setRemote(remote)
	r.setLocalPath(local)
	r.Logger = Logger

	// With the other VCS we can check if the endpoint locally is different
	// from the one configured internally. But, with Bzr you can't. For example,
	// if you do `bzr branch https://launchpad.net/govcstestbzrrepo` and then
	// use `bzr info` to get the parent branch you'll find it set to
	// http://bazaar.launchpad.net/~mattfarina/govcstestbzrrepo/trunk/. Notice
	// the change from https to http and the path chance.
	// Here we set the remote to be the local one if none is passed in.
	if err == nil && r.CheckLocal() && remote == "" {
		c := exec.Command("bzr", "info")
		c.Dir = local
		c.Env = envForDir(c.Dir)
		out, err := c.CombinedOutput()
		if err != nil {
			return nil, NewLocalError("Unable to retrieve local repo information", err, string(out))
		}
		m := bzrDetectURL.FindStringSubmatch(string(out))

		// If no remote was passed in but one is configured for the locally
		// checked out Bzr repo use that one.
		if m[1] != "" {
			r.setRemote(m[1])
		}
	}

	return r, nil
}

// BzrRepo implements the Repo interface for the Bzr source control.
type BzrRepo struct {
	base
}

// Vcs retrieves the underlying VCS being implemented.
func (s BzrRepo) Vcs() Type {
	return Bzr
}

// Get is used to perform an initial clone of a repository.
func (s *BzrRepo) Get() error {

	basePath := filepath.Dir(filepath.FromSlash(s.LocalPath()))
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		err = os.MkdirAll(basePath, 0755)
		if err != nil {
			return NewLocalError("Unable to create directory", err, "")
		}
	}

	out, err := s.run("bzr", "branch", s.Remote(), s.LocalPath())
	if err != nil {
		return NewRemoteError("Unable to get repository", err, string(out))
	}

	return nil
}

// Init initializes a bazaar repository at local location.
func (s *BzrRepo) Init() error {
	out, err := s.run("bzr", "init", s.LocalPath())

	// There are some windows cases where bazaar cannot create the parent
	// directory if it does not already exist, to the location it's trying
	// to create the repo. Catch that error and try to handle it.
	if err != nil && s.isUnableToCreateDir(err) {

		basePath := filepath.Dir(filepath.FromSlash(s.LocalPath()))
		if _, err := os.Stat(basePath); os.IsNotExist(err) {
			err = os.MkdirAll(basePath, 0755)
			if err != nil {
				return NewLocalError("Unable to initialize repository", err, "")
			}

			out, err = s.run("bzr", "init", s.LocalPath())
			if err != nil {
				return NewLocalError("Unable to initialize repository", err, string(out))
			}
			return nil
		}

	} else if err != nil {
		return NewLocalError("Unable to initialize repository", err, string(out))
	}

	return nil
}

// Update performs a Bzr pull and update to an existing checkout.
func (s *BzrRepo) Update() error {
	out, err := s.RunFromDir("bzr", "pull")
	if err != nil {
		return NewRemoteError("Unable to update repository", err, string(out))
	}
	out, err = s.RunFromDir("bzr", "update")
	if err != nil {
		return NewRemoteError("Unable to update repository", err, string(out))
	}
	return nil
}

// UpdateVersion sets the version of a package currently checked out via Bzr.
func (s *BzrRepo) UpdateVersion(version string) error {
	out, err := s.RunFromDir("bzr", "update", "-r", version)
	if err != nil {
		return NewLocalError("Unable to update checked out version", err, string(out))
	}
	return nil
}

// Version retrieves the current version.
func (s *BzrRepo) Version() (string, error) {

	out, err := s.RunFromDir("bzr", "revno", "--tree")
	if err != nil {
		return "", NewLocalError("Unable to retrieve checked out version", err, string(out))
	}

	return strings.TrimSpace(string(out)), nil
}

// Current returns the current version-ish. This means:
// * -1 if on the tip of the branch (this is the Bzr value for HEAD)
// * A tag if on a tag
// * Otherwise a revision
func (s *BzrRepo) Current() (string, error) {
	tip, err := s.CommitInfo("-1")
	if err != nil {
		return "", err
	}

	curr, err := s.Version()
	if err != nil {
		return "", err
	}

	if tip.Commit == curr {
		return "-1", nil
	}

	ts, err := s.TagsFromCommit(curr)
	if err != nil {
		return "", err
	}
	if len(ts) > 0 {
		return ts[0], nil
	}

	return curr, nil
}

// Date retrieves the date on the latest commit.
func (s *BzrRepo) Date() (time.Time, error) {
	out, err := s.RunFromDir("bzr", "version-info", "--custom", "--template={date}")
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	t, err := time.Parse(longForm, string(out))
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	return t, nil
}

// CheckLocal verifies the local location is a Bzr repo.
func (s *BzrRepo) CheckLocal() bool {
	if _, err := os.Stat(s.LocalPath() + "/.bzr"); err == nil {
		return true
	}

	return false
}

// Branches returns a list of available branches on the repository.
// In Bazaar (Bzr) clones and branches are the same. A different branch will
// have a different URL location which we cannot detect from the repo. This
// is a little different from other VCS.
func (s *BzrRepo) Branches() ([]string, error) {
	var branches []string
	return branches, nil
}

// Tags returns a list of available tags on the repository.
func (s *BzrRepo) Tags() ([]string, error) {
	out, err := s.RunFromDir("bzr", "tags")
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}
	tags := s.referenceList(string(out), `(?m-s)^(\S+)`)
	return tags, nil
}

// IsReference returns if a string is a reference. A reference can be a
// commit id or tag.
func (s *BzrRepo) IsReference(r string) bool {
	_, err := s.RunFromDir("bzr", "revno", "-r", r)
	return err == nil
}

// IsDirty returns if the checkout has been modified from the checked
// out reference.
func (s *BzrRepo) IsDirty() bool {
	out, err := s.RunFromDir("bzr", "diff")
	return err != nil || len(out) != 0
}

// CommitInfo retrieves metadata about a commit.
func (s *BzrRepo) CommitInfo(id string) (*CommitInfo, error) {
	r := "-r" + id
	out, err := s.RunFromDir("bzr", "log", r, "--log-format=long")
	if err != nil {
		return nil, ErrRevisionUnavailable
	}

	ci := &CommitInfo{}
	lines := strings.Split(string(out), "\n")
	const format = "Mon 2006-01-02 15:04:05 -0700"
	var track int
	var trackOn bool

	// Note, bzr does not appear to use i18m.
	for i, l := range lines {
		if strings.HasPrefix(l, "revno:") {
			ci.Commit = strings.TrimSpace(strings.TrimPrefix(l, "revno:"))
		} else if strings.HasPrefix(l, "committer:") {
			ci.Author = strings.TrimSpace(strings.TrimPrefix(l, "committer:"))
		} else if strings.HasPrefix(l, "timestamp:") {
			ts := strings.TrimSpace(strings.TrimPrefix(l, "timestamp:"))
			ci.Date, err = time.Parse(format, ts)
			if err != nil {
				return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
			}
		} else if strings.TrimSpace(l) == "message:" {
			track = i
			trackOn = true
		} else if trackOn && i > track {
			ci.Message = ci.Message + l
		}
	}
	ci.Message = strings.TrimSpace(ci.Message)

	// Didn't find the revision
	if ci.Author == "" {
		return nil, ErrRevisionUnavailable
	}

	return ci, nil
}

// TagsFromCommit retrieves tags from a commit id.
func (s *BzrRepo) TagsFromCommit(id string) ([]string, error) {
	out, err := s.RunFromDir("bzr", "tags", "-r", id)
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}

	tags := s.referenceList(string(out), `(?m-s)^(\S+)`)
	return tags, nil
}

// Ping returns if remote location is accessible.
func (s *BzrRepo) Ping() bool {

	// Running bzr info is slow. Many of the projects are on launchpad which
	// has a public 1.0 API we can use.
	u, err := url.Parse(s.Remote())
	if err == nil {
		if u.Host == "launchpad.net" {
			try := strings.TrimPrefix(u.Path, "/")

			// get returns the body and an err. If the status code is not a 200
			// an error is returned. Launchpad returns a 404 for a codebase that
			// does not exist. Otherwise it returns a JSON object describing it.
			_, er := get("https://api.launchpad.net/1.0/" + try)
			return er == nil
		}
	}

	// This is the same command that Go itself uses but it's not fast (or fast
	// enough by my standards). A faster method would be useful.
	_, err = s.run("bzr", "info", s.Remote())
	return err == nil
}

// ExportDir exports the current revision to the passed in directory.
func (s *BzrRepo) ExportDir(dir string) error {
	out, err := s.RunFromDir("bzr", "export", dir)
	s.log(out)
	if err != nil {
		return NewLocalError("Unable to export source", err, string(out))
	}

	return nil
}

// Multi-lingual manner check for the VCS error that it couldn't create directory.
// https://bazaar.launchpad.net/~bzr-pqm/bzr/bzr.dev/files/head:/po/
func (s *BzrRepo) isUnableToCreateDir(err error) bool {
	msg := err.Error()

	if strings.HasPrefix(msg, fmt.Sprintf("Parent directory of %s does not exist.", s.LocalPath())) ||
		strings.HasPrefix(msg, fmt.Sprintf("Nadřazený adresář %s neexistuje.", s.LocalPath())) ||
		strings.HasPrefix(msg, fmt.Sprintf("El directorio padre de %s no existe.", s.LocalPath())) ||
		strings.HasPrefix(msg, fmt.Sprintf("%s の親ディレクトリがありません。", s.LocalPath())) ||
		strings.HasPrefix(msg, fmt.Sprintf("Родительская директория для %s не существует.", s.LocalPath())) {
		return true
	}

	return false
}
