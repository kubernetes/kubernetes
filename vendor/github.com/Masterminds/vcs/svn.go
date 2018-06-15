package vcs

import (
	"encoding/xml"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// NewSvnRepo creates a new instance of SvnRepo. The remote and local directories
// need to be passed in. The remote location should include the branch for SVN.
// For example, if the package is https://github.com/Masterminds/cookoo/ the remote
// should be https://github.com/Masterminds/cookoo/trunk for the trunk branch.
func NewSvnRepo(remote, local string) (*SvnRepo, error) {
	ins := depInstalled("svn")
	if !ins {
		return nil, NewLocalError("svn is not installed", nil, "")
	}
	ltype, err := DetectVcsFromFS(local)

	// Found a VCS other than Svn. Need to report an error.
	if err == nil && ltype != Svn {
		return nil, ErrWrongVCS
	}

	r := &SvnRepo{}
	r.setRemote(remote)
	r.setLocalPath(local)
	r.Logger = Logger

	// Make sure the local SVN repo is configured the same as the remote when
	// A remote value was passed in.
	if err == nil && r.CheckLocal() {
		// An SVN repo was found so test that the URL there matches
		// the repo passed in here.
		out, err := exec.Command("svn", "info", local).CombinedOutput()
		if err != nil {
			return nil, NewLocalError("Unable to retrieve local repo information", err, string(out))
		}

		detectedRemote, err := detectRemoteFromInfoCommand(string(out))
		if err != nil {
			return nil, NewLocalError("Unable to retrieve local repo information", err, string(out))
		}
		if detectedRemote != "" && remote != "" && detectedRemote != remote {
			return nil, ErrWrongRemote
		}

		// If no remote was passed in but one is configured for the locally
		// checked out Svn repo use that one.
		if remote == "" && detectedRemote != "" {
			r.setRemote(detectedRemote)
		}
	}

	return r, nil
}

// SvnRepo implements the Repo interface for the Svn source control.
type SvnRepo struct {
	base
}

// Vcs retrieves the underlying VCS being implemented.
func (s SvnRepo) Vcs() Type {
	return Svn
}

// Get is used to perform an initial checkout of a repository.
// Note, because SVN isn't distributed this is a checkout without
// a clone.
func (s *SvnRepo) Get() error {
	remote := s.Remote()
	if strings.HasPrefix(remote, "/") {
		remote = "file://" + remote
	} else if runtime.GOOS == "windows" && filepath.VolumeName(remote) != "" {
		remote = "file:///" + remote
	}
	out, err := s.run("svn", "checkout", remote, s.LocalPath())
	if err != nil {
		return NewRemoteError("Unable to get repository", err, string(out))
	}
	return nil
}

// Init will create a svn repository at remote location.
func (s *SvnRepo) Init() error {
	out, err := s.run("svnadmin", "create", s.Remote())

	if err != nil && s.isUnableToCreateDir(err) {

		basePath := filepath.Dir(filepath.FromSlash(s.Remote()))
		if _, err := os.Stat(basePath); os.IsNotExist(err) {
			err = os.MkdirAll(basePath, 0755)
			if err != nil {
				return NewLocalError("Unable to initialize repository", err, "")
			}

			out, err = s.run("svnadmin", "create", s.Remote())
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

// Update performs an SVN update to an existing checkout.
func (s *SvnRepo) Update() error {
	out, err := s.RunFromDir("svn", "update")
	if err != nil {
		return NewRemoteError("Unable to update repository", err, string(out))
	}
	return err
}

// UpdateVersion sets the version of a package currently checked out via SVN.
func (s *SvnRepo) UpdateVersion(version string) error {
	out, err := s.RunFromDir("svn", "update", "-r", version)
	if err != nil {
		return NewRemoteError("Unable to update checked out version", err, string(out))
	}
	return nil
}

// Version retrieves the current version.
func (s *SvnRepo) Version() (string, error) {
	type Commit struct {
		Revision string `xml:"revision,attr"`
	}
	type Info struct {
		Commit Commit `xml:"entry>commit"`
	}

	out, err := s.RunFromDir("svn", "info", "--xml")
	if err != nil {
		return "", NewLocalError("Unable to retrieve checked out version", err, string(out))
	}
	s.log(out)
	infos := &Info{}
	err = xml.Unmarshal(out, &infos)
	if err != nil {
		return "", NewLocalError("Unable to retrieve checked out version", err, string(out))
	}

	return infos.Commit.Revision, nil
}

// Current returns the current version-ish. This means:
// * HEAD if on the tip.
// * Otherwise a revision id
func (s *SvnRepo) Current() (string, error) {
	tip, err := s.CommitInfo("HEAD")
	if err != nil {
		return "", err
	}

	curr, err := s.Version()
	if err != nil {
		return "", err
	}

	if tip.Commit == curr {
		return "HEAD", nil
	}

	return curr, nil
}

// Date retrieves the date on the latest commit.
func (s *SvnRepo) Date() (time.Time, error) {
	version, err := s.Version()
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, "")
	}
	out, err := s.RunFromDir("svn", "pget", "svn:date", "--revprop", "-r", version)
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	const longForm = "2006-01-02T15:04:05.000000Z"
	t, err := time.Parse(longForm, strings.TrimSpace(string(out)))
	if err != nil {
		return time.Time{}, NewLocalError("Unable to retrieve revision date", err, string(out))
	}
	return t, nil
}

// CheckLocal verifies the local location is an SVN repo.
func (s *SvnRepo) CheckLocal() bool {
	pth, err := filepath.Abs(s.LocalPath())
	if err != nil {
		s.log(err.Error())
		return false
	}

	if _, err := os.Stat(filepath.Join(pth, ".svn")); err == nil {
		return true
	}

	oldpth := pth
	for oldpth != pth {
		pth = filepath.Dir(pth)
		if _, err := os.Stat(filepath.Join(pth, ".svn")); err == nil {
			return true
		}
	}

	return false
}

// Tags returns []string{} as there are no formal tags in SVN. Tags are a
// convention in SVN. They are typically implemented as a copy of the trunk and
// placed in the /tags/[tag name] directory. Since this is a convention the
// expectation is to checkout a tag the correct subdirectory will be used
// as the path. For more information see:
// http://svnbook.red-bean.com/en/1.7/svn.branchmerge.tags.html
func (s *SvnRepo) Tags() ([]string, error) {
	return []string{}, nil
}

// Branches returns []string{} as there are no formal branches in SVN. Branches
// are a convention. They are typically implemented as a copy of the trunk and
// placed in the /branches/[tag name] directory. Since this is a convention the
// expectation is to checkout a branch the correct subdirectory will be used
// as the path. For more information see:
// http://svnbook.red-bean.com/en/1.7/svn.branchmerge.using.html
func (s *SvnRepo) Branches() ([]string, error) {
	return []string{}, nil
}

// IsReference returns if a string is a reference. A reference is a commit id.
// Branches and tags are part of the path.
func (s *SvnRepo) IsReference(r string) bool {
	out, err := s.RunFromDir("svn", "log", "-r", r)

	// This is a complete hack. There must be a better way to do this. Pull
	// requests welcome. When the reference isn't real you get a line of
	// repeated - followed by an empty line. If the reference is real there
	// is commit information in addition to those. So, we look for responses
	// over 2 lines long.
	lines := strings.Split(string(out), "\n")
	if err == nil && len(lines) > 2 {
		return true
	}

	return false
}

// IsDirty returns if the checkout has been modified from the checked
// out reference.
func (s *SvnRepo) IsDirty() bool {
	out, err := s.RunFromDir("svn", "diff")
	return err != nil || len(out) != 0
}

// CommitInfo retrieves metadata about a commit.
func (s *SvnRepo) CommitInfo(id string) (*CommitInfo, error) {

	// There are cases where Svn log doesn't return anything for HEAD or BASE.
	// svn info does provide details for these but does not have elements like
	// the commit message.
	if id == "HEAD" || id == "BASE" {
		type Commit struct {
			Revision string `xml:"revision,attr"`
		}
		type Info struct {
			Commit Commit `xml:"entry>commit"`
		}

		out, err := s.RunFromDir("svn", "info", "-r", id, "--xml")
		if err != nil {
			return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
		}
		infos := &Info{}
		err = xml.Unmarshal(out, &infos)
		if err != nil {
			return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
		}

		id = infos.Commit.Revision
		if id == "" {
			return nil, ErrRevisionUnavailable
		}
	}

	out, err := s.RunFromDir("svn", "log", "-r", id, "--xml")
	if err != nil {
		return nil, NewRemoteError("Unable to retrieve commit information", err, string(out))
	}

	type Logentry struct {
		Author string `xml:"author"`
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
		Commit:  id,
		Author:  logs.Logs[0].Author,
		Message: logs.Logs[0].Msg,
	}

	if len(logs.Logs[0].Date) > 0 {
		ci.Date, err = time.Parse(time.RFC3339Nano, logs.Logs[0].Date)
		if err != nil {
			return nil, NewLocalError("Unable to retrieve commit information", err, string(out))
		}
	}

	return ci, nil
}

// TagsFromCommit retrieves tags from a commit id.
func (s *SvnRepo) TagsFromCommit(id string) ([]string, error) {
	// Svn tags are a convention implemented as paths. See the details on the
	// Tag() method for more information.
	return []string{}, nil
}

// Ping returns if remote location is accessible.
func (s *SvnRepo) Ping() bool {
	_, err := s.run("svn", "--non-interactive", "info", s.Remote())
	return err == nil
}

// ExportDir exports the current revision to the passed in directory.
func (s *SvnRepo) ExportDir(dir string) error {

	out, err := s.RunFromDir("svn", "export", ".", dir)
	s.log(out)
	if err != nil {
		return NewLocalError("Unable to export source", err, string(out))
	}

	return nil
}

// isUnableToCreateDir checks for an error in Init() to see if an error
// where the parent directory of the VCS local path doesn't exist.
func (s *SvnRepo) isUnableToCreateDir(err error) bool {
	msg := err.Error()
	return strings.HasPrefix(msg, "E000002")
}

// detectRemoteFromInfoCommand finds the remote url from the `svn info`
// command's output without using  a regex. We avoid regex because URLs
// are notoriously complex to accurately match with a regex and
// splitting strings is less complex and often faster
func detectRemoteFromInfoCommand(infoOut string) (string, error) {
	sBytes := []byte(infoOut)
	urlIndex := strings.Index(infoOut, "URL: ")
	if urlIndex == -1 {
		return "", fmt.Errorf("Remote not specified in svn info")
	}
	urlEndIndex := strings.Index(string(sBytes[urlIndex:]), "\n")
	if urlEndIndex == -1 {
		urlEndIndex = strings.Index(string(sBytes[urlIndex:]), "\r")
		if urlEndIndex == -1 {
			return "", fmt.Errorf("Unable to parse remote URL for svn info")
		}
	}

	return string(sBytes[(urlIndex + 5):(urlIndex + urlEndIndex)]), nil
}
