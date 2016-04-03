package client

import (
	"fmt"
	"net/url"

	"github.com/docker/docker/graph/tags"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/utils"
)

// CmdPull pulls an image or a repository from the registry.
//
// Usage: docker pull [OPTIONS] IMAGENAME[:TAG|@DIGEST]
func (cli *DockerCli) CmdPull(args ...string) error {
	cmd := cli.Subcmd("pull", []string{"NAME[:TAG|@DIGEST]"}, "Pull an image or a repository from a registry", true)
	allTags := cmd.Bool([]string{"a", "-all-tags"}, false, "Download all tagged images in the repository")
	cmd.Require(flag.Exact, 1)

	cmd.ParseFlags(args, true)

	var (
		v         = url.Values{}
		remote    = cmd.Arg(0)
		newRemote = remote
	)
	taglessRemote, tag := parsers.ParseRepositoryTag(remote)
	if tag == "" && !*allTags {
		newRemote = utils.ImageReference(taglessRemote, tags.DEFAULTTAG)
	}
	if tag != "" && *allTags {
		return fmt.Errorf("tag can't be used with --all-tags/-a")
	}

	v.Set("fromImage", newRemote)

	// Resolve the Repository name from fqn to RepositoryInfo
	repoInfo, err := registry.ParseRepositoryInfo(taglessRemote)
	if err != nil {
		return err
	}

	_, _, err = cli.clientRequestAttemptLogin("POST", "/images/create?"+v.Encode(), nil, cli.out, repoInfo.Index, "pull")
	return err
}
