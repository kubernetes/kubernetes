package utils

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
)

// GitClone clones a repository into a newly created directory which
// will be under "docker-build-git"
func GitClone(remoteURL string) (string, error) {
	if !urlutil.IsGitTransport(remoteURL) {
		remoteURL = "https://" + remoteURL
	}
	root, err := ioutil.TempDir("", "docker-build-git")
	if err != nil {
		return "", err
	}

	u, err := url.Parse(remoteURL)
	if err != nil {
		return "", err
	}

	fragment := u.Fragment
	clone := cloneArgs(u, root)

	if output, err := git(clone...); err != nil {
		return "", fmt.Errorf("Error trying to use git: %s (%s)", err, output)
	}

	return checkoutGit(fragment, root)
}

func cloneArgs(remoteURL *url.URL, root string) []string {
	args := []string{"clone", "--recursive"}
	shallow := len(remoteURL.Fragment) == 0

	if shallow && strings.HasPrefix(remoteURL.Scheme, "http") {
		res, err := http.Head(fmt.Sprintf("%s/info/refs?service=git-upload-pack", remoteURL))
		if err != nil || res.Header.Get("Content-Type") != "application/x-git-upload-pack-advertisement" {
			shallow = false
		}
	}

	if shallow {
		args = append(args, "--depth", "1")
	}

	if remoteURL.Fragment != "" {
		remoteURL.Fragment = ""
	}

	return append(args, remoteURL.String(), root)
}

func checkoutGit(fragment, root string) (string, error) {
	refAndDir := strings.SplitN(fragment, ":", 2)

	if len(refAndDir[0]) != 0 {
		if output, err := gitWithinDir(root, "checkout", refAndDir[0]); err != nil {
			return "", fmt.Errorf("Error trying to use git: %s (%s)", err, output)
		}
	}

	if len(refAndDir) > 1 && len(refAndDir[1]) != 0 {
		newCtx, err := symlink.FollowSymlinkInScope(filepath.Join(root, refAndDir[1]), root)
		if err != nil {
			return "", fmt.Errorf("Error setting git context, %q not within git root: %s", refAndDir[1], err)
		}

		fi, err := os.Stat(newCtx)
		if err != nil {
			return "", err
		}
		if !fi.IsDir() {
			return "", fmt.Errorf("Error setting git context, not a directory: %s", newCtx)
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
