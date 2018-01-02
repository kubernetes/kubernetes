package seed

import (
	"fmt"
	"net/url"
	"os/exec"
	"path"
)

const GitRevision = "revision"

type Git struct {
	host     string
	revision string
	ready    bool
}

// String representation of this source
func (g *Git) String() string {
	return "github://" + g.host
}

// Load from URI into dest.
func (g *Git) Load(dest string) error {

	g.ready = false
	cmd := exec.Command("git", "clone", g.host, dest)

	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("'wd: %s git clone %s': %s: %s", dest, g.host, output, err)
	}
	if len(g.revision) == 0 {
		g.ready = true
		return nil
	}
	cmd = exec.Command("git", "checkout", g.revision)
	cmd.Dir = dest
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("wd %v 'git checkout %s': %s: %s", cmd.Dir, g.revision, output, err)
	}
	cmd = exec.Command("git", "reset", "--hard")
	cmd.Dir = dest
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("wd %v 'git reset --hard  %s': %s: %s", cmd.Dir, g.revision, output, err)
	}

	g.ready = true
	return nil
}

// Metadata for this source.
func (g *Git) MetadataRead(mdDir string) (string, error) {
	return "", nil
}

// MetadataWrite for this source.
func (g *Git) MetadataWrite(mdDir string) error {
	return nil
}

func NewGitSource(uri string, options map[string]string) (Source, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	if u.Scheme != "github" {
		return nil, ErrUnsupported
	}
	return &Git{
		host:     "http://" + path.Join(u.Host, u.Path),
		revision: options[GitRevision],
	}, nil
}
