package vcs

import (
	"encoding/xml"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

var hgDetectURL = regexp.MustCompile("default = (?P<foo>.+)\n")

// NewHgRepo creates a new instance of HgRepo. The remote and local directories
// need to be passed in.
func NewHgRepo(remote, local string) (*HgRepo, error) {
	ins := depInstalled("hg")
	if !ins {
		return nil, NewLocalError("hg is not installed", nil, "")
	}
	ltype, err := DetectVcsFromFS(local)

	// Found a VCS other than Hg. Need to report an error.
	if err == nil && ltype != Hg {
		return nil, ErrWrongVCS
	}

	r := &HgRepo{}
	r.setRemote(remote)
	r.setLocalPath(local)
	r.Logger = Logger

	// Make sure the local Hg repo is configured the same as the remote when
	// A remote value was passed in.
	if err == nil && r.CheckLocal() {
		// An Hg repo was found so test that the URL there matches
		// the repo passed in here.
		c := exec.Command("hg", "paths")
		c.Dir = local
		c.Env = envForDir(c.Dir)
		out, err := c.CombinedOutput()
		if err != nil {
			return nil, NewLocalError("Unable to retrieve local repo information", err, string(out))
		}

		m := hgDetectURL.FindStringSubmatch(string(out))
		if m[1] != "" && m[1] != remote {
			return nil, ErrWrongRemote
		}

		// If no remote was passed in but one is configured for the locally
		// checked out Hg repo use that one.
		if remote == "" && m[1] != "" {
			r.setRemote(m[1])
		}
	}

	return r, nil
}

// HgRepo implements the Repo interface for the Mercurial source control.
type HgRepo struct {
	base
}

// Vcs retrieves the underlying VCS being implemented.
func (s HgRepo) Vcs() Type {
	return Hg
}

// Get is used to perform an initial clone of a repository.
func (s *HgRepo) Get() error {
	out, err := s.run("hg", "clone", s.Remote(), s.LocalPath())
	if err != nil {
		return NewRemoteError("Unable to get repository", err, string(out))
	}
	return nil
}

// Init will initialize a mercurial repository at local location.
func (s *HgRepo) Init() error {
	out, err := s.run("hg", "init", s.LocalPath())
	if err != nil {
		return NewLocalError("Unable to initialize repository", err, string(out))
	}
	return nil
}

// Update performs a Mercurial pull to an existing checkout.
func (s *HgRepo) Update() error {
	return s.UpdateVersion(``)
}

// UpdateVersion sets the version of a package currently checked out via Hg.
func (s *HgRepo) UpdateVersion(version string) error {
	out, err := s.RunFromDir("hg", "pull")
	if err != nil {
		return NewLocalError("Unable to update checked out version", err, string(out))
	}
	if len(strings.TrimSpace(version)) > 0 {
		out, err = s.RunFromDir("hg", "update", version)
	} else {
		out, err = s.RunFromDir("hg", "update")
	}
	if err != nil {
		return NewLocalError("Unable to update checked out version", err, string(out))
	}
	return nil
}

// Version retrieves the current version.
func (s *HgRepo) Version() (string, error) {
	out, err := s.RunFromDir("hg", "--debug", "identify")
	if err != nil {
		return "", NewLocalError("Unable to retrieve checked out version", err, string(out))
	}

	parts := strings.SplitN(string(out), " ", 2)
	sha := parts[0]
	return strings.TrimSpace(sha), nil
}

// Current returns the current version-ish. This means:
// * Branch name if on the tip of the branch
// * Tag if on a tag
// * Otherwise a revision id
func (s *HgRepo) Current() (string, error) {
	out, err := s.RunFromDir("hg", "branch")
	if err != nil {
		return "", err
	}
	branch := strings.TrimSpace(string(out))

	tip, err := s.CommitInfo("max(branch(" + branch + "))")
	if err != nil {
		return "", err
	}

	curr, err := s.Version()
	if err != nil {
		return "", err
	}

	if tip.Commit == curr {

		return branch, nil
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
func (s *HgRepo) Date() (time.Time, error) {
	version, err := s.Version()
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, "")
	}
	out, err := s.RunFromDir("hg", "log", "-r", version, "--template", "{date|isodatesec}")
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	t, err := time.Parse(longForm, string(out))
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	return t, nil
}

// CheckLocal verifies the local location is a Git repo.
func (s *HgRepo) CheckLocal() bool {
	if _, err := os.Stat(s.LocalPath() + "/.hg"); err == nil {
		return true
	}

	return false
}

// Branches returns a list of available branches
func (s *HgRepo) Branches() ([]string, error) {
	out, err := s.RunFromDir("hg", "branches")
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve branches", err, string(out))
	}
	branches := s.referenceList(string(out), `(?m-s)^(\S+)`)
	return branches, nil
}

// Tags returns a list of available tags
func (s *HgRepo) Tags() ([]string, error) {
	out, err := s.RunFromDir("hg", "tags")
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}
	tags := s.referenceList(string(out), `(?m-s)^(\S+)`)
	return tags, nil
}

// IsReference returns if a string is a reference. A reference can be a
// commit id, branch, or tag.
func (s *HgRepo) IsReference(r string) bool {
	_, err := s.RunFromDir("hg", "log", "-r", r)
	return err == nil
}

// IsDirty returns if the checkout has been modified from the checked
// out reference.
func (s *HgRepo) IsDirty() bool {
	out, err := s.RunFromDir("hg", "diff")
	return err != nil || len(out) != 0
}

// CommitInfo retrieves metadata about a commit.
func (s *HgRepo) CommitInfo(id string) (*CommitInfo, error) {
	out, err := s.RunFromDir("hg", "log", "-r", id, "--style=xml")
	if err != nil {
		return nil, ErrRevisionUnavailable
	}

	type Author struct {
		Name  string `xml:",chardata"`
		Email string `xml:"email,attr"`
	}
	type Logentry struct {
		Node   string `xml:"node,attr"`
		Author Author `xml:"author"`
		Date   string `xml:"date"`
		Msg    string `xml:"msg"`
	}
	type Log struct {
		XMLName xml.Name   `xml:"log"`
		Logs    []Logentry `xml:"logentry"`
	}

	logs := &Log{}
	err = xml.Unmarshal(out, &logs)
	if err != nil {
		return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
	}
	if len(logs.Logs) == 0 {
		return nil, ErrRevisionUnavailable
	}

	ci := &CommitInfo{
		Commit:  logs.Logs[0].Node,
		Author:  logs.Logs[0].Author.Name + " <" + logs.Logs[0].Author.Email + ">",
		Message: logs.Logs[0].Msg,
	}

	if logs.Logs[0].Date != "" {
		ci.Date, err = time.Parse(time.RFC3339, logs.Logs[0].Date)
		if err != nil {
			return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
		}
	}

	return ci, nil
}

// TagsFromCommit retrieves tags from a commit id.
func (s *HgRepo) TagsFromCommit(id string) ([]string, error) {
	// Hg has a single tag per commit. If a second tag is added to a commit a
	// new commit is created and the tag is attached to that new commit.
	out, err := s.RunFromDir("hg", "log", "-r", id, "--style=xml")
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}

	type Logentry struct {
		Node string `xml:"node,attr"`
		Tag  string `xml:"tag"`
	}
	type Log struct {
		XMLName xml.Name   `xml:"log"`
		Logs    []Logentry `xml:"logentry"`
	}

	logs := &Log{}
	err = xml.Unmarshal(out, &logs)
	if err != nil {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}
	if len(logs.Logs) == 0 {
		return []string{}, NewLocalError("Unable to retrieve tags", err, string(out))
	}

	t := strings.TrimSpace(logs.Logs[0].Tag)
	if t != "" {
		return []string{t}, nil
	}
	return []string{}, nil
}

// Ping returns if remote location is accessible.
func (s *HgRepo) Ping() bool {
	_, err := s.run("hg", "identify", s.Remote())
	return err == nil
}

// ExportDir exports the current revision to the passed in directory.
func (s *HgRepo) ExportDir(dir string) error {

	out, err := s.RunFromDir("hg", "archive", dir)
	s.log(out)
	if err != nil {
		return NewLocalError("Unable to export source", err, string(out))
	}

	return nil
}
