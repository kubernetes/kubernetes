package git

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/pkg/urlutil"
	"github.com/pkg/errors"
)

type gitRepo struct {
	remote string
	ref    string
	subdir string
}

// Clone clones a repository into a newly created directory which
// will be under "docker-build-git"
func Clone(remoteURL string) (string, error) {
	repo, err := parseRemoteURL(remoteURL)

	if err != nil {
		return "", err
	}

	fetch := fetchArgs(repo.remote, repo.ref)

	root, err := ioutil.TempDir("", "docker-build-git")
	if err != nil {
		return "", err
	}

	if out, err := gitWithinDir(root, "init"); err != nil {
		return "", errors.Wrapf(err, "failed to init repo at %s: %s", root, out)
	}

	// Add origin remote for compatibility with previous implementation that
	// used "git clone" and also to make sure local refs are created for branches
	if out, err := gitWithinDir(root, "remote", "add", "origin", repo.remote); err != nil {
		return "", errors.Wrapf(err, "failed add origin repo at %s: %s", repo.remote, out)
	}

	if output, err := gitWithinDir(root, fetch...); err != nil {
		return "", errors.Wrapf(err, "error fetching: %s", output)
	}

	return checkoutGit(root, repo.ref, repo.subdir)
}

func parseRemoteURL(remoteURL string) (gitRepo, error) {
	repo := gitRepo{}

	if !isGitTransport(remoteURL) {
		remoteURL = "https://" + remoteURL
	}

	var fragment string
	if strings.HasPrefix(remoteURL, "git@") {
		// git@.. is not an URL, so cannot be parsed as URL
		parts := strings.SplitN(remoteURL, "#", 2)

		repo.remote = parts[0]
		if len(parts) == 2 {
			fragment = parts[1]
		}
		repo.ref, repo.subdir = getRefAndSubdir(fragment)
	} else {
		u, err := url.Parse(remoteURL)
		if err != nil {
			return repo, err
		}

		repo.ref, repo.subdir = getRefAndSubdir(u.Fragment)
		u.Fragment = ""
		repo.remote = u.String()
	}
	return repo, nil
}

func getRefAndSubdir(fragment string) (ref string, subdir string) {
	refAndDir := strings.SplitN(fragment, ":", 2)
	ref = "master"
	if len(refAndDir[0]) != 0 {
		ref = refAndDir[0]
	}
	if len(refAndDir) > 1 && len(refAndDir[1]) != 0 {
		subdir = refAndDir[1]
	}
	return
}

func fetchArgs(remoteURL string, ref string) []string {
	args := []string{"fetch", "--recurse-submodules=yes"}
	shallow := true

	if urlutil.IsURL(remoteURL) {
		res, err := http.Head(fmt.Sprintf("%s/info/refs?service=git-upload-pack", remoteURL))
		if err != nil || res.Header.Get("Content-Type") != "application/x-git-upload-pack-advertisement" {
			shallow = false
		}
	}

	if shallow {
		args = append(args, "--depth", "1")
	}

	return append(args, "origin", ref)
}

func checkoutGit(root, ref, subdir string) (string, error) {
	// Try checking out by ref name first. This will work on branches and sets
	// .git/HEAD to the current branch name
	if output, err := gitWithinDir(root, "checkout", ref); err != nil {
		// If checking out by branch name fails check out the last fetched ref
		if _, err2 := gitWithinDir(root, "checkout", "FETCH_HEAD"); err2 != nil {
			return "", errors.Wrapf(err, "error checking out %s: %s", ref, output)
		}
	}

	if subdir != "" {
		newCtx, err := symlink.FollowSymlinkInScope(filepath.Join(root, subdir), root)
		if err != nil {
			return "", errors.Wrapf(err, "error setting git context, %q not within git root", subdir)
		}

		fi, err := os.Stat(newCtx)
		if err != nil {
			return "", err
		}
		if !fi.IsDir() {
			return "", errors.Errorf("error setting git context, not a directory: %s", newCtx)
		}
		root = newCtx
	}

	return root, nil
}

func gitWithinDir(dir string, args ...string) ([]byte, error) {
	a := []string{"--work-tree", dir, "--git-dir", filepath.Join(dir, ".git")}
	return git(append(a, args...)...)
}

func git(args ...string) ([]byte, error) {
	return exec.Command("git", args...).CombinedOutput()
}

// isGitTransport returns true if the provided str is a git transport by inspecting
// the prefix of the string for known protocols used in git.
func isGitTransport(str string) bool {
	return urlutil.IsURL(str) || strings.HasPrefix(str, "git://") || strings.HasPrefix(str, "git@")
}
