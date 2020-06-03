package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

var (
	UpstreamSummaryPattern = regexp.MustCompile(`^UPSTREAM: (revert: )?(([\w\.-]+\/[\w-\.-]+)?: )?(\d+:|<carry>:|<drop>:)`)
	BumpSummaryPattern     = regexp.MustCompile(`^bump[\(\w].*`)

	// patchRegexps holds regexps for paths inside vendor dir that are allowed to be patched directly.
	// It must corresponds to published repositories.
	PatchRegexps = []*regexp.Regexp{
		regexp.MustCompile("^k8s.io/kubernetes/.*"),
	}

	// supportedHosts maps source hosts to the number of path segments that
	// represent the account/repo for that host. This is necessary because we
	// can't tell just by looking at an import path whether the repo is identified
	// by the first 2 or 3 path segments.
	//
	// If dependencies are introduced from new hosts, they'll need to be added
	// here.
	SupportedHosts = map[string]int{
		"bitbucket.org":     3,
		"cloud.google.com":  2,
		"code.google.com":   3,
		"github.com":        3,
		"golang.org":        3,
		"google.golang.org": 2,
		"gopkg.in":          2,
		"k8s.io":            2,
		"speter.net":        2,
	}
)

func RegexpsToStrings(a []*regexp.Regexp) []string {
	var res []string
	for _, r := range a {
		res = append(res, r.String())
	}
	return res
}

type File string

func (f File) HasVendoredCodeChanges() bool {
	return strings.HasPrefix(string(f), "vendor")
}

func (f File) IsPatch() bool {
	if !strings.HasPrefix(string(f), "vendor/") {
		return false
	}

	for _, r := range PatchRegexps {
		if r.Match([]byte(strings.TrimPrefix(string(f), "vendor/"))) {
			return true
		}
	}

	return false
}

func (f File) VendorRepo() (string, error) {
	if !strings.HasPrefix(string(f), "vendor/") {
		return "", fmt.Errorf("file %q doesn't appear to be a vendor change", string(f))
	}

	p := strings.TrimPrefix(string(f), "vendor/")

	parts := strings.Split(p, string(os.PathSeparator))

	if len(parts) < 1 {
		return "", fmt.Errorf("invalid file %q", string(f))
	}

	numSegments, ok := SupportedHosts[parts[0]]
	if !ok {
		return "", fmt.Errorf("unsupported host for file %q", string(f))
	}

	if numSegments < 1 {
		return "", fmt.Errorf("invalid number of segments %d when processing file path %q", numSegments, string(f))
	}

	return strings.Join(parts[0:numSegments], string(os.PathSeparator)), nil
}

type Commit struct {
	Sha         string
	Summary     string
	Description []string
	Files       []File
	Email       string
}

func (c Commit) MatchesUpstreamSummaryPattern() bool {
	return UpstreamSummaryPattern.MatchString(c.Summary)
}

func (c Commit) MatchesBumpSummaryPattern() bool {
	return BumpSummaryPattern.MatchString(c.Summary)
}

func (c Commit) DeclaredUpstreamRepo() (string, error) {
	if !c.MatchesUpstreamSummaryPattern() {
		return "", fmt.Errorf("commit doesn't match the upstream commit summary pattern")
	}
	groups := UpstreamSummaryPattern.FindStringSubmatch(c.Summary)
	repo := groups[3]
	if len(repo) == 0 {
		repo = "k8s.io/kubernetes"
	}
	return repo, nil
}

func (c Commit) HasVendoredCodeChanges() bool {
	for _, file := range c.Files {
		if file.HasVendoredCodeChanges() {
			return true
		}
	}
	return false
}

func (c Commit) HasNonVendoredCodeChanges() bool {
	for _, file := range c.Files {
		if !file.HasVendoredCodeChanges() {
			return true
		}
	}
	return false
}

func (c Commit) HasPatches() bool {
	for _, f := range c.Files {
		if f.IsPatch() {
			return true
		}
	}
	return false
}

func (c Commit) HasBumpedFiles() bool {
	for _, f := range c.Files {
		if f.HasVendoredCodeChanges() && !f.IsPatch() {
			return true
		}
	}
	return false
}

func (c Commit) PatchedRepos() ([]string, error) {
	var repos []string
	seenKeys := map[string]struct{}{}
	for _, f := range c.Files {
		if f.IsPatch() {
			repo, err := f.VendorRepo()
			if err != nil {
				return nil, err
			}
			_, ok := seenKeys[repo]
			if !ok {
				repos = append(repos, repo)
				seenKeys[repo] = struct{}{}
			}
		}
	}
	return repos, nil
}

func IsCommit(a string) bool {
	if _, _, err := run("git", "rev-parse", a); err != nil {
		return false
	}
	return true
}

var ErrNotCommit = fmt.Errorf("one or both of the provided commits was not a valid commit")

func CommitsBetween(a, b string) ([]Commit, error) {
	commits := []Commit{}
	stdout, stderr, err := run("git", "log", "--oneline", fmt.Sprintf("%s..%s", a, b))
	if err != nil {
		if !IsCommit(a) || !IsCommit(b) {
			return nil, ErrNotCommit
		}
		return nil, fmt.Errorf("error executing git log: %s: %s", stderr, err)
	}
	for _, log := range strings.Split(stdout, "\n") {
		if len(log) == 0 {
			continue
		}
		commit, err := NewCommitFromOnelineLog(log)
		if err != nil {
			return nil, err
		}
		commits = append(commits, commit)
	}
	return commits, nil
}

func NewCommitFromOnelineLog(log string) (Commit, error) {
	var commit Commit
	var err error
	parts := strings.Split(log, " ")
	if len(parts) < 2 {
		return commit, fmt.Errorf("invalid log entry: %s", log)
	}
	commit.Sha = parts[0]
	commit.Summary = strings.Join(parts[1:], " ")
	commit.Description, err = descriptionInCommit(commit.Sha)
	if err != nil {
		return commit, err
	}
	files, err := filesInCommit(commit.Sha)
	if err != nil {
		return commit, err
	}
	commit.Files = files
	commit.Email, err = emailInCommit(commit.Sha)
	if err != nil {
		return commit, err
	}
	return commit, nil
}

func FetchRepo(repoDir string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	defer os.Chdir(cwd)

	if err := os.Chdir(repoDir); err != nil {
		return err
	}

	if stdout, stderr, err := run("git", "fetch", "origin"); err != nil {
		return fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	}
	return nil
}

func IsAncestor(commit1, commit2, repoDir string) (bool, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return false, err
	}
	defer os.Chdir(cwd)

	if err := os.Chdir(repoDir); err != nil {
		return false, err
	}

	if stdout, stderr, err := run("git", "merge-base", "--is-ancestor", commit1, commit2); err != nil {
		return false, fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	}

	return true, nil
}

func CommitDate(commit, repoDir string) (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	defer os.Chdir(cwd)

	if err := os.Chdir(repoDir); err != nil {
		return "", err
	}

	if stdout, stderr, err := run("git", "fetch", "origin"); err != nil {
		return "", fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	}

	if stdout, stderr, err := run("git", "show", "-s", "--format=%ci", commit); err != nil {
		return "", fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	} else {
		return strings.TrimSpace(stdout), nil
	}
}

func Checkout(commit, repoDir string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	defer os.Chdir(cwd)

	if err := os.Chdir(repoDir); err != nil {
		return err
	}

	if stdout, stderr, err := run("git", "checkout", commit); err != nil {
		return fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	}
	return nil
}

func CurrentRev(repoDir string) (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	defer os.Chdir(cwd)

	if err := os.Chdir(repoDir); err != nil {
		return "", err
	}

	if stdout, stderr, err := run("git", "rev-parse", "HEAD"); err != nil {
		return "", fmt.Errorf("out=%s, err=%s, %s", strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
	} else {
		return strings.TrimSpace(stdout), nil
	}
}

func emailInCommit(sha string) (string, error) {
	stdout, stderr, err := run("git", "show", `--format=%ae`, "-s", sha)
	if err != nil {
		return "", fmt.Errorf("%s: %s", stderr, err)
	}
	return strings.TrimSpace(stdout), nil
}

func filesInCommit(sha string) ([]File, error) {
	files := []File{}
	stdout, stderr, err := run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
	if err != nil {
		return nil, fmt.Errorf("%s: %s", stderr, err)
	}
	for _, filename := range strings.Split(stdout, "\n") {
		if len(filename) == 0 {
			continue
		}
		files = append(files, File(filename))
	}
	return files, nil
}

func descriptionInCommit(sha string) ([]string, error) {
	descriptionLines := []string{}
	stdout, stderr, err := run("git", "log", "--pretty=%b", "-1", sha)
	if err != nil {
		return descriptionLines, fmt.Errorf("%s: %s", stderr, err)
	}

	for _, commitLine := range strings.Split(stdout, "\n") {
		if len(commitLine) == 0 {
			continue
		}
		descriptionLines = append(descriptionLines, commitLine)
	}
	return descriptionLines, nil
}

func run(args ...string) (string, string, error) {
	cmd := exec.Command(args[0], args[1:]...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}
