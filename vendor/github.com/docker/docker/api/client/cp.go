package client

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/archive"
	flag "github.com/docker/docker/pkg/mflag"
)

type copyDirection int

const (
	fromContainer copyDirection = (1 << iota)
	toContainer
	acrossContainers = fromContainer | toContainer
)

// CmdCp copies files/folders to or from a path in a container.
//
// When copying from a container, if LOCALPATH is '-' the data is written as a
// tar archive file to STDOUT.
//
// When copying to a container, if LOCALPATH is '-' the data is read as a tar
// archive file from STDIN, and the destination CONTAINER:PATH, must specify
// a directory.
//
// Usage:
// 	docker cp CONTAINER:PATH LOCALPATH|-
// 	docker cp LOCALPATH|- CONTAINER:PATH
func (cli *DockerCli) CmdCp(args ...string) error {
	cmd := cli.Subcmd(
		"cp",
		[]string{"CONTAINER:PATH LOCALPATH|-", "LOCALPATH|- CONTAINER:PATH"},
		strings.Join([]string{
			"Copy files/folders between a container and your host.\n",
			"Use '-' as the source to read a tar archive from stdin\n",
			"and extract it to a directory destination in a container.\n",
			"Use '-' as the destination to stream a tar archive of a\n",
			"container source to stdout.",
		}, ""),
		true,
	)

	cmd.Require(flag.Exact, 2)
	cmd.ParseFlags(args, true)

	if cmd.Arg(0) == "" {
		return fmt.Errorf("source can not be empty")
	}
	if cmd.Arg(1) == "" {
		return fmt.Errorf("destination can not be empty")
	}

	srcContainer, srcPath := splitCpArg(cmd.Arg(0))
	dstContainer, dstPath := splitCpArg(cmd.Arg(1))

	var direction copyDirection
	if srcContainer != "" {
		direction |= fromContainer
	}
	if dstContainer != "" {
		direction |= toContainer
	}

	switch direction {
	case fromContainer:
		return cli.copyFromContainer(srcContainer, srcPath, dstPath)
	case toContainer:
		return cli.copyToContainer(srcPath, dstContainer, dstPath)
	case acrossContainers:
		// Copying between containers isn't supported.
		return fmt.Errorf("copying between containers is not supported")
	default:
		// User didn't specify any container.
		return fmt.Errorf("must specify at least one container source")
	}
}

// We use `:` as a delimiter between CONTAINER and PATH, but `:` could also be
// in a valid LOCALPATH, like `file:name.txt`. We can resolve this ambiguity by
// requiring a LOCALPATH with a `:` to be made explicit with a relative or
// absolute path:
// 	`/path/to/file:name.txt` or `./file:name.txt`
//
// This is apparently how `scp` handles this as well:
// 	http://www.cyberciti.biz/faq/rsync-scp-file-name-with-colon-punctuation-in-it/
//
// We can't simply check for a filepath separator because container names may
// have a separator, e.g., "host0/cname1" if container is in a Docker cluster,
// so we have to check for a `/` or `.` prefix. Also, in the case of a Windows
// client, a `:` could be part of an absolute Windows path, in which case it
// is immediately proceeded by a backslash.
func splitCpArg(arg string) (container, path string) {
	if filepath.IsAbs(arg) {
		// Explicit local absolute path, e.g., `C:\foo` or `/foo`.
		return "", arg
	}

	parts := strings.SplitN(arg, ":", 2)

	if len(parts) == 1 || strings.HasPrefix(parts[0], ".") {
		// Either there's no `:` in the arg
		// OR it's an explicit local relative path like `./file:name.txt`.
		return "", arg
	}

	return parts[0], parts[1]
}

func (cli *DockerCli) statContainerPath(containerName, path string) (types.ContainerPathStat, error) {
	var stat types.ContainerPathStat

	query := make(url.Values, 1)
	query.Set("path", filepath.ToSlash(path)) // Normalize the paths used in the API.

	urlStr := fmt.Sprintf("/containers/%s/archive?%s", containerName, query.Encode())

	response, err := cli.call("HEAD", urlStr, nil, nil)
	if err != nil {
		return stat, err
	}
	defer response.body.Close()

	if response.statusCode != http.StatusOK {
		return stat, fmt.Errorf("unexpected status code from daemon: %d", response.statusCode)
	}

	return getContainerPathStatFromHeader(response.header)
}

func getContainerPathStatFromHeader(header http.Header) (types.ContainerPathStat, error) {
	var stat types.ContainerPathStat

	encodedStat := header.Get("X-Docker-Container-Path-Stat")
	statDecoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(encodedStat))

	err := json.NewDecoder(statDecoder).Decode(&stat)
	if err != nil {
		err = fmt.Errorf("unable to decode container path stat header: %s", err)
	}

	return stat, err
}

func resolveLocalPath(localPath string) (absPath string, err error) {
	if absPath, err = filepath.Abs(localPath); err != nil {
		return
	}

	return archive.PreserveTrailingDotOrSeparator(absPath, localPath), nil
}

func (cli *DockerCli) copyFromContainer(srcContainer, srcPath, dstPath string) (err error) {
	if dstPath != "-" {
		// Get an absolute destination path.
		dstPath, err = resolveLocalPath(dstPath)
		if err != nil {
			return err
		}
	}

	query := make(url.Values, 1)
	query.Set("path", filepath.ToSlash(srcPath)) // Normalize the paths used in the API.

	urlStr := fmt.Sprintf("/containers/%s/archive?%s", srcContainer, query.Encode())

	response, err := cli.call("GET", urlStr, nil, nil)
	if err != nil {
		return err
	}
	defer response.body.Close()

	if response.statusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code from daemon: %d", response.statusCode)
	}

	if dstPath == "-" {
		// Send the response to STDOUT.
		_, err = io.Copy(os.Stdout, response.body)

		return err
	}

	// In order to get the copy behavior right, we need to know information
	// about both the source and the destination. The response headers include
	// stat info about the source that we can use in deciding exactly how to
	// copy it locally. Along with the stat info about the local destination,
	// we have everything we need to handle the multiple possibilities there
	// can be when copying a file/dir from one location to another file/dir.
	stat, err := getContainerPathStatFromHeader(response.header)
	if err != nil {
		return fmt.Errorf("unable to get resource stat from response: %s", err)
	}

	// Prepare source copy info.
	srcInfo := archive.CopyInfo{
		Path:   srcPath,
		Exists: true,
		IsDir:  stat.Mode.IsDir(),
	}

	// See comments in the implementation of `archive.CopyTo` for exactly what
	// goes into deciding how and whether the source archive needs to be
	// altered for the correct copy behavior.
	return archive.CopyTo(response.body, srcInfo, dstPath)
}

func (cli *DockerCli) copyToContainer(srcPath, dstContainer, dstPath string) (err error) {
	if srcPath != "-" {
		// Get an absolute source path.
		srcPath, err = resolveLocalPath(srcPath)
		if err != nil {
			return err
		}
	}

	// In order to get the copy behavior right, we need to know information
	// about both the source and destination. The API is a simple tar
	// archive/extract API but we can use the stat info header about the
	// destination to be more informed about exactly what the destination is.

	// Prepare destination copy info by stat-ing the container path.
	dstInfo := archive.CopyInfo{Path: dstPath}
	dstStat, err := cli.statContainerPath(dstContainer, dstPath)
	// Ignore any error and assume that the parent directory of the destination
	// path exists, in which case the copy may still succeed. If there is any
	// type of conflict (e.g., non-directory overwriting an existing directory
	// or vice versia) the extraction will fail. If the destination simply did
	// not exist, but the parent directory does, the extraction will still
	// succeed.
	if err == nil {
		dstInfo.Exists, dstInfo.IsDir = true, dstStat.Mode.IsDir()
	}

	var content io.Reader
	if srcPath == "-" {
		// Use STDIN.
		content = os.Stdin
		if !dstInfo.IsDir {
			return fmt.Errorf("destination %q must be a directory", fmt.Sprintf("%s:%s", dstContainer, dstPath))
		}
	} else {
		srcArchive, err := archive.TarResource(srcPath)
		if err != nil {
			return err
		}
		defer srcArchive.Close()

		// With the stat info about the local source as well as the
		// destination, we have enough information to know whether we need to
		// alter the archive that we upload so that when the server extracts
		// it to the specified directory in the container we get the disired
		// copy behavior.

		// Prepare source copy info.
		srcInfo, err := archive.CopyInfoStatPath(srcPath, true)
		if err != nil {
			return err
		}

		// See comments in the implementation of `archive.PrepareArchiveCopy`
		// for exactly what goes into deciding how and whether the source
		// archive needs to be altered for the correct copy behavior when it is
		// extracted. This function also infers from the source and destination
		// info which directory to extract to, which may be the parent of the
		// destination that the user specified.
		dstDir, preparedArchive, err := archive.PrepareArchiveCopy(srcArchive, srcInfo, dstInfo)
		if err != nil {
			return err
		}
		defer preparedArchive.Close()

		dstPath = dstDir
		content = preparedArchive
	}

	query := make(url.Values, 2)
	query.Set("path", filepath.ToSlash(dstPath)) // Normalize the paths used in the API.
	// Do not allow for an existing directory to be overwritten by a non-directory and vice versa.
	query.Set("noOverwriteDirNonDir", "true")

	urlStr := fmt.Sprintf("/containers/%s/archive?%s", dstContainer, query.Encode())

	response, err := cli.stream("PUT", urlStr, &streamOpts{in: content})
	if err != nil {
		return err
	}
	defer response.body.Close()

	if response.statusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code from daemon: %d", response.statusCode)
	}

	return nil
}
