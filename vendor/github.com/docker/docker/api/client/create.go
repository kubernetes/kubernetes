package client

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/graph/tags"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/utils"
)

func (cli *DockerCli) pullImage(image string) error {
	return cli.pullImageCustomOut(image, cli.out)
}

func (cli *DockerCli) pullImageCustomOut(image string, out io.Writer) error {
	v := url.Values{}
	repos, tag := parsers.ParseRepositoryTag(image)
	// pull only the image tagged 'latest' if no tag was specified
	if tag == "" {
		tag = tags.DEFAULTTAG
	}
	v.Set("fromImage", repos)
	v.Set("tag", tag)

	// Resolve the Repository name from fqn to RepositoryInfo
	repoInfo, err := registry.ParseRepositoryInfo(repos)
	if err != nil {
		return err
	}

	// Resolve the Auth config relevant for this server
	authConfig := registry.ResolveAuthConfig(cli.configFile, repoInfo.Index)
	buf, err := json.Marshal(authConfig)
	if err != nil {
		return err
	}

	registryAuthHeader := []string{
		base64.URLEncoding.EncodeToString(buf),
	}
	sopts := &streamOpts{
		rawTerminal: true,
		out:         out,
		headers:     map[string][]string{"X-Registry-Auth": registryAuthHeader},
	}
	if _, err := cli.stream("POST", "/images/create?"+v.Encode(), sopts); err != nil {
		return err
	}
	return nil
}

type cidFile struct {
	path    string
	file    *os.File
	written bool
}

func newCIDFile(path string) (*cidFile, error) {
	if _, err := os.Stat(path); err == nil {
		return nil, fmt.Errorf("Container ID file found, make sure the other container isn't running or delete %s", path)
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("Failed to create the container ID file: %s", err)
	}

	return &cidFile{path: path, file: f}, nil
}

func (cli *DockerCli) createContainer(config *runconfig.Config, hostConfig *runconfig.HostConfig, cidfile, name string) (*types.ContainerCreateResponse, error) {
	containerValues := url.Values{}
	if name != "" {
		containerValues.Set("name", name)
	}

	mergedConfig := runconfig.MergeConfigs(config, hostConfig)

	var containerIDFile *cidFile
	if cidfile != "" {
		var err error
		if containerIDFile, err = newCIDFile(cidfile); err != nil {
			return nil, err
		}
		defer containerIDFile.Close()
	}

	//create the container
	serverResp, err := cli.call("POST", "/containers/create?"+containerValues.Encode(), mergedConfig, nil)
	//if image not found try to pull it
	if serverResp.statusCode == 404 && strings.Contains(err.Error(), config.Image) {
		repo, tag := parsers.ParseRepositoryTag(config.Image)
		if tag == "" {
			tag = tags.DEFAULTTAG
		}
		fmt.Fprintf(cli.err, "Unable to find image '%s' locally\n", utils.ImageReference(repo, tag))

		// we don't want to write to stdout anything apart from container.ID
		if err = cli.pullImageCustomOut(config.Image, cli.err); err != nil {
			return nil, err
		}
		// Retry
		if serverResp, err = cli.call("POST", "/containers/create?"+containerValues.Encode(), mergedConfig, nil); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}

	defer serverResp.body.Close()

	var response types.ContainerCreateResponse
	if err := json.NewDecoder(serverResp.body).Decode(&response); err != nil {
		return nil, err
	}
	for _, warning := range response.Warnings {
		fmt.Fprintf(cli.err, "WARNING: %s\n", warning)
	}
	if containerIDFile != nil {
		if err = containerIDFile.Write(response.ID); err != nil {
			return nil, err
		}
	}
	return &response, nil
}

// CmdCreate creates a new container from a given image.
//
// Usage: docker create [OPTIONS] IMAGE [COMMAND] [ARG...]
func (cli *DockerCli) CmdCreate(args ...string) error {
	cmd := cli.Subcmd("create", []string{"IMAGE [COMMAND] [ARG...]"}, "Create a new container", true)

	// These are flags not stored in Config/HostConfig
	var (
		flName = cmd.String([]string{"-name"}, "", "Assign a name to the container")
	)

	config, hostConfig, cmd, err := runconfig.Parse(cmd, args)
	if err != nil {
		cmd.ReportError(err.Error(), true)
		os.Exit(1)
	}
	if config.Image == "" {
		cmd.Usage()
		return nil
	}
	response, err := cli.createContainer(config, hostConfig, hostConfig.ContainerIDFile, *flName)
	if err != nil {
		return err
	}
	fmt.Fprintf(cli.out, "%s\n", response.ID)
	return nil
}
