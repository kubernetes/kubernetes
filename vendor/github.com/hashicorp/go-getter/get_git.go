package getter

import (
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	urlhelper "github.com/hashicorp/go-getter/helper/url"
	"github.com/hashicorp/go-safetemp"
	"github.com/hashicorp/go-version"
)

// GitGetter is a Getter implementation that will download a module from
// a git repository.
type GitGetter struct{}

func (g *GitGetter) ClientMode(_ *url.URL) (ClientMode, error) {
	return ClientModeDir, nil
}

func (g *GitGetter) Get(dst string, u *url.URL) error {
	if _, err := exec.LookPath("git"); err != nil {
		return fmt.Errorf("git must be available and on the PATH")
	}

	// Extract some query parameters we use
	var ref, sshKey string
	q := u.Query()
	if len(q) > 0 {
		ref = q.Get("ref")
		q.Del("ref")

		sshKey = q.Get("sshkey")
		q.Del("sshkey")

		// Copy the URL
		var newU url.URL = *u
		u = &newU
		u.RawQuery = q.Encode()
	}

	var sshKeyFile string
	if sshKey != "" {
		// Check that the git version is sufficiently new.
		if err := checkGitVersion("2.3"); err != nil {
			return fmt.Errorf("Error using ssh key: %v", err)
		}

		// We have an SSH key - decode it.
		raw, err := base64.StdEncoding.DecodeString(sshKey)
		if err != nil {
			return err
		}

		// Create a temp file for the key and ensure it is removed.
		fh, err := ioutil.TempFile("", "go-getter")
		if err != nil {
			return err
		}
		sshKeyFile = fh.Name()
		defer os.Remove(sshKeyFile)

		// Set the permissions prior to writing the key material.
		if err := os.Chmod(sshKeyFile, 0600); err != nil {
			return err
		}

		// Write the raw key into the temp file.
		_, err = fh.Write(raw)
		fh.Close()
		if err != nil {
			return err
		}
	}

	// Clone or update the repository
	_, err := os.Stat(dst)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err == nil {
		err = g.update(dst, sshKeyFile, ref)
	} else {
		err = g.clone(dst, sshKeyFile, u)
	}
	if err != nil {
		return err
	}

	// Next: check out the proper tag/branch if it is specified, and checkout
	if ref != "" {
		if err := g.checkout(dst, ref); err != nil {
			return err
		}
	}

	// Lastly, download any/all submodules.
	return g.fetchSubmodules(dst, sshKeyFile)
}

// GetFile for Git doesn't support updating at this time. It will download
// the file every time.
func (g *GitGetter) GetFile(dst string, u *url.URL) error {
	td, tdcloser, err := safetemp.Dir("", "getter")
	if err != nil {
		return err
	}
	defer tdcloser.Close()

	// Get the filename, and strip the filename from the URL so we can
	// just get the repository directly.
	filename := filepath.Base(u.Path)
	u.Path = filepath.Dir(u.Path)

	// Get the full repository
	if err := g.Get(td, u); err != nil {
		return err
	}

	// Copy the single file
	u, err = urlhelper.Parse(fmtFileURL(filepath.Join(td, filename)))
	if err != nil {
		return err
	}

	fg := &FileGetter{Copy: true}
	return fg.GetFile(dst, u)
}

func (g *GitGetter) checkout(dst string, ref string) error {
	cmd := exec.Command("git", "checkout", ref)
	cmd.Dir = dst
	return getRunCommand(cmd)
}

func (g *GitGetter) clone(dst, sshKeyFile string, u *url.URL) error {
	cmd := exec.Command("git", "clone", u.String(), dst)
	setupGitEnv(cmd, sshKeyFile)
	return getRunCommand(cmd)
}

func (g *GitGetter) update(dst, sshKeyFile, ref string) error {
	// Determine if we're a branch. If we're NOT a branch, then we just
	// switch to master prior to checking out
	cmd := exec.Command("git", "show-ref", "-q", "--verify", "refs/heads/"+ref)
	cmd.Dir = dst

	if getRunCommand(cmd) != nil {
		// Not a branch, switch to master. This will also catch non-existent
		// branches, in which case we want to switch to master and then
		// checkout the proper branch later.
		ref = "master"
	}

	// We have to be on a branch to pull
	if err := g.checkout(dst, ref); err != nil {
		return err
	}

	cmd = exec.Command("git", "pull", "--ff-only")
	cmd.Dir = dst
	setupGitEnv(cmd, sshKeyFile)
	return getRunCommand(cmd)
}

// fetchSubmodules downloads any configured submodules recursively.
func (g *GitGetter) fetchSubmodules(dst, sshKeyFile string) error {
	cmd := exec.Command("git", "submodule", "update", "--init", "--recursive")
	cmd.Dir = dst
	setupGitEnv(cmd, sshKeyFile)
	return getRunCommand(cmd)
}

// setupGitEnv sets up the environment for the given command. This is used to
// pass configuration data to git and ssh and enables advanced cloning methods.
func setupGitEnv(cmd *exec.Cmd, sshKeyFile string) {
	const gitSSHCommand = "GIT_SSH_COMMAND="
	var sshCmd []string

	// If we have an existing GIT_SSH_COMMAND, we need to append our options.
	// We will also remove our old entry to make sure the behavior is the same
	// with versions of Go < 1.9.
	env := os.Environ()
	for i, v := range env {
		if strings.HasPrefix(v, gitSSHCommand) {
			sshCmd = []string{v}

			env[i], env[len(env)-1] = env[len(env)-1], env[i]
			env = env[:len(env)-1]
			break
		}
	}

	if len(sshCmd) == 0 {
		sshCmd = []string{gitSSHCommand + "ssh"}
	}

	if sshKeyFile != "" {
		// We have an SSH key temp file configured, tell ssh about this.
		sshCmd = append(sshCmd, "-i", sshKeyFile)
	}

	env = append(env, strings.Join(sshCmd, " "))
	cmd.Env = env
}

// checkGitVersion is used to check the version of git installed on the system
// against a known minimum version. Returns an error if the installed version
// is older than the given minimum.
func checkGitVersion(min string) error {
	want, err := version.NewVersion(min)
	if err != nil {
		return err
	}

	out, err := exec.Command("git", "version").Output()
	if err != nil {
		return err
	}

	fields := strings.Fields(string(out))
	if len(fields) < 3 {
		return fmt.Errorf("Unexpected 'git version' output: %q", string(out))
	}

	have, err := version.NewVersion(fields[2])
	if err != nil {
		return err
	}

	if have.LessThan(want) {
		return fmt.Errorf("Required git version = %s, have %s", want, have)
	}

	return nil
}
