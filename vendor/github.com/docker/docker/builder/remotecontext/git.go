package remotecontext

import (
	"os"

	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/remotecontext/git"
	"github.com/docker/docker/pkg/archive"
)

// MakeGitContext returns a Context from gitURL that is cloned in a temporary directory.
func MakeGitContext(gitURL string) (builder.Source, error) {
	root, err := git.Clone(gitURL)
	if err != nil {
		return nil, err
	}

	c, err := archive.Tar(root, archive.Uncompressed)
	if err != nil {
		return nil, err
	}

	defer func() {
		// TODO: print errors?
		c.Close()
		os.RemoveAll(root)
	}()
	return FromArchive(c)
}
