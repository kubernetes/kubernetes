package builder

// internals for handling commands. Covers many areas and a lot of
// non-contiguous functionality. Please read the comments.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/builder/parser"
	"github.com/docker/docker/cliconfig"
	"github.com/docker/docker/daemon"
	"github.com/docker/docker/graph"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/httputils"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/progressreader"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/tarsum"
	"github.com/docker/docker/pkg/urlutil"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/runconfig"
)

func (b *Builder) readContext(context io.Reader) (err error) {
	tmpdirPath, err := ioutil.TempDir("", "docker-build")
	if err != nil {
		return
	}

	// Make sure we clean-up upon error.  In the happy case the caller
	// is expected to manage the clean-up
	defer func() {
		if err != nil {
			if e := os.RemoveAll(tmpdirPath); e != nil {
				logrus.Debugf("[BUILDER] failed to remove temporary context: %s", e)
			}
		}
	}()

	decompressedStream, err := archive.DecompressStream(context)
	if err != nil {
		return
	}

	if b.context, err = tarsum.NewTarSum(decompressedStream, true, tarsum.Version1); err != nil {
		return
	}

	if err = chrootarchive.Untar(b.context, tmpdirPath, nil); err != nil {
		return
	}

	b.contextPath = tmpdirPath
	return
}

func (b *Builder) commit(id string, autoCmd *runconfig.Command, comment string) error {
	if b.disableCommit {
		return nil
	}
	if b.image == "" && !b.noBaseImage {
		return fmt.Errorf("Please provide a source image with `from` prior to commit")
	}
	b.Config.Image = b.image
	if id == "" {
		cmd := b.Config.Cmd
		if runtime.GOOS != "windows" {
			b.Config.Cmd = runconfig.NewCommand("/bin/sh", "-c", "#(nop) "+comment)
		} else {
			b.Config.Cmd = runconfig.NewCommand("cmd", "/S /C", "REM (nop) "+comment)
		}
		defer func(cmd *runconfig.Command) { b.Config.Cmd = cmd }(cmd)

		hit, err := b.probeCache()
		if err != nil {
			return err
		}
		if hit {
			return nil
		}

		container, err := b.create()
		if err != nil {
			return err
		}
		id = container.ID

		if err := container.Mount(); err != nil {
			return err
		}
		defer container.Unmount()
	}
	container, err := b.Daemon.Get(id)
	if err != nil {
		return err
	}

	// Note: Actually copy the struct
	autoConfig := *b.Config
	autoConfig.Cmd = autoCmd

	commitCfg := &daemon.ContainerCommitConfig{
		Author: b.maintainer,
		Pause:  true,
		Config: &autoConfig,
	}

	// Commit the container
	image, err := b.Daemon.Commit(container, commitCfg)
	if err != nil {
		return err
	}
	b.Daemon.Graph().Retain(b.id, image.ID)
	b.activeImages = append(b.activeImages, image.ID)
	b.image = image.ID
	return nil
}

type copyInfo struct {
	origPath   string
	destPath   string
	hash       string
	decompress bool
	tmpDir     string
}

func (b *Builder) runContextCommand(args []string, allowRemote bool, allowDecompression bool, cmdName string) error {
	if b.context == nil {
		return fmt.Errorf("No context given. Impossible to use %s", cmdName)
	}

	if len(args) < 2 {
		return fmt.Errorf("Invalid %s format - at least two arguments required", cmdName)
	}

	// Work in daemon-specific filepath semantics
	dest := filepath.FromSlash(args[len(args)-1]) // last one is always the dest

	copyInfos := []*copyInfo{}

	b.Config.Image = b.image

	defer func() {
		for _, ci := range copyInfos {
			if ci.tmpDir != "" {
				os.RemoveAll(ci.tmpDir)
			}
		}
	}()

	// Loop through each src file and calculate the info we need to
	// do the copy (e.g. hash value if cached).  Don't actually do
	// the copy until we've looked at all src files
	for _, orig := range args[0 : len(args)-1] {
		if err := calcCopyInfo(
			b,
			cmdName,
			&copyInfos,
			orig,
			dest,
			allowRemote,
			allowDecompression,
			true,
		); err != nil {
			return err
		}
	}

	if len(copyInfos) == 0 {
		return fmt.Errorf("No source files were specified")
	}
	if len(copyInfos) > 1 && !strings.HasSuffix(dest, string(os.PathSeparator)) {
		return fmt.Errorf("When using %s with more than one source file, the destination must be a directory and end with a /", cmdName)
	}

	// For backwards compat, if there's just one CI then use it as the
	// cache look-up string, otherwise hash 'em all into one
	var srcHash string
	var origPaths string

	if len(copyInfos) == 1 {
		srcHash = copyInfos[0].hash
		origPaths = copyInfos[0].origPath
	} else {
		var hashs []string
		var origs []string
		for _, ci := range copyInfos {
			hashs = append(hashs, ci.hash)
			origs = append(origs, ci.origPath)
		}
		hasher := sha256.New()
		hasher.Write([]byte(strings.Join(hashs, ",")))
		srcHash = "multi:" + hex.EncodeToString(hasher.Sum(nil))
		origPaths = strings.Join(origs, " ")
	}

	cmd := b.Config.Cmd
	if runtime.GOOS != "windows" {
		b.Config.Cmd = runconfig.NewCommand("/bin/sh", "-c", fmt.Sprintf("#(nop) %s %s in %s", cmdName, srcHash, dest))
	} else {
		b.Config.Cmd = runconfig.NewCommand("cmd", "/S /C", fmt.Sprintf("REM (nop) %s %s in %s", cmdName, srcHash, dest))
	}
	defer func(cmd *runconfig.Command) { b.Config.Cmd = cmd }(cmd)

	hit, err := b.probeCache()
	if err != nil {
		return err
	}

	if hit {
		return nil
	}

	container, _, err := b.Daemon.Create(b.Config, nil, "")
	if err != nil {
		return err
	}
	b.TmpContainers[container.ID] = struct{}{}

	if err := container.Mount(); err != nil {
		return err
	}
	defer container.Unmount()

	if err := container.PrepareStorage(); err != nil {
		return err
	}

	for _, ci := range copyInfos {
		if err := b.addContext(container, ci.origPath, ci.destPath, ci.decompress); err != nil {
			return err
		}
	}

	if err := container.CleanupStorage(); err != nil {
		return err
	}

	if err := b.commit(container.ID, cmd, fmt.Sprintf("%s %s in %s", cmdName, origPaths, dest)); err != nil {
		return err
	}
	return nil
}

func calcCopyInfo(b *Builder, cmdName string, cInfos *[]*copyInfo, origPath string, destPath string, allowRemote bool, allowDecompression bool, allowWildcards bool) error {

	// Work in daemon-specific OS filepath semantics. However, we save
	// the the origPath passed in here, as it might also be a URL which
	// we need to check for in this function.
	passedInOrigPath := origPath
	origPath = filepath.FromSlash(origPath)
	destPath = filepath.FromSlash(destPath)

	if origPath != "" && origPath[0] == os.PathSeparator && len(origPath) > 1 {
		origPath = origPath[1:]
	}
	origPath = strings.TrimPrefix(origPath, "."+string(os.PathSeparator))

	// Twiddle the destPath when its a relative path - meaning, make it
	// relative to the WORKINGDIR
	if !filepath.IsAbs(destPath) {
		hasSlash := strings.HasSuffix(destPath, string(os.PathSeparator))
		destPath = filepath.Join(string(os.PathSeparator), filepath.FromSlash(b.Config.WorkingDir), destPath)

		// Make sure we preserve any trailing slash
		if hasSlash {
			destPath += string(os.PathSeparator)
		}
	}

	// In the remote/URL case, download it and gen its hashcode
	if urlutil.IsURL(passedInOrigPath) {

		// As it's a URL, we go back to processing on what was passed in
		// to this function
		origPath = passedInOrigPath

		if !allowRemote {
			return fmt.Errorf("Source can't be a URL for %s", cmdName)
		}

		ci := copyInfo{}
		ci.origPath = origPath
		ci.hash = origPath // default to this but can change
		ci.destPath = destPath
		ci.decompress = false
		*cInfos = append(*cInfos, &ci)

		// Initiate the download
		resp, err := httputils.Download(ci.origPath)
		if err != nil {
			return err
		}

		// Create a tmp dir
		tmpDirName, err := ioutil.TempDir(b.contextPath, "docker-remote")
		if err != nil {
			return err
		}
		ci.tmpDir = tmpDirName

		// Create a tmp file within our tmp dir
		tmpFileName := filepath.Join(tmpDirName, "tmp")
		tmpFile, err := os.OpenFile(tmpFileName, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return err
		}

		// Download and dump result to tmp file
		if _, err := io.Copy(tmpFile, progressreader.New(progressreader.Config{
			In:        resp.Body,
			Out:       b.OutOld,
			Formatter: b.StreamFormatter,
			Size:      int(resp.ContentLength),
			NewLines:  true,
			ID:        "",
			Action:    "Downloading",
		})); err != nil {
			tmpFile.Close()
			return err
		}
		fmt.Fprintf(b.OutStream, "\n")
		tmpFile.Close()

		// Set the mtime to the Last-Modified header value if present
		// Otherwise just remove atime and mtime
		times := make([]syscall.Timespec, 2)

		lastMod := resp.Header.Get("Last-Modified")
		if lastMod != "" {
			mTime, err := http.ParseTime(lastMod)
			// If we can't parse it then just let it default to 'zero'
			// otherwise use the parsed time value
			if err == nil {
				times[1] = syscall.NsecToTimespec(mTime.UnixNano())
			}
		}

		if err := system.UtimesNano(tmpFileName, times); err != nil {
			return err
		}

		ci.origPath = filepath.Join(filepath.Base(tmpDirName), filepath.Base(tmpFileName))

		// If the destination is a directory, figure out the filename.
		if strings.HasSuffix(ci.destPath, string(os.PathSeparator)) {
			u, err := url.Parse(origPath)
			if err != nil {
				return err
			}
			path := u.Path
			if strings.HasSuffix(path, string(os.PathSeparator)) {
				path = path[:len(path)-1]
			}
			parts := strings.Split(path, string(os.PathSeparator))
			filename := parts[len(parts)-1]
			if filename == "" {
				return fmt.Errorf("cannot determine filename from url: %s", u)
			}
			ci.destPath = ci.destPath + filename
		}

		// Calc the checksum, even if we're using the cache
		r, err := archive.Tar(tmpFileName, archive.Uncompressed)
		if err != nil {
			return err
		}
		tarSum, err := tarsum.NewTarSum(r, true, tarsum.Version1)
		if err != nil {
			return err
		}
		if _, err := io.Copy(ioutil.Discard, tarSum); err != nil {
			return err
		}
		ci.hash = tarSum.Sum(nil)
		r.Close()

		return nil
	}

	// Deal with wildcards
	if allowWildcards && ContainsWildcards(origPath) {
		for _, fileInfo := range b.context.GetSums() {
			if fileInfo.Name() == "" {
				continue
			}
			match, _ := filepath.Match(origPath, fileInfo.Name())
			if !match {
				continue
			}

			// Note we set allowWildcards to false in case the name has
			// a * in it
			calcCopyInfo(b, cmdName, cInfos, fileInfo.Name(), destPath, allowRemote, allowDecompression, false)
		}
		return nil
	}

	// Must be a dir or a file

	if err := b.checkPathForAddition(origPath); err != nil {
		return err
	}
	fi, _ := os.Stat(filepath.Join(b.contextPath, origPath))

	ci := copyInfo{}
	ci.origPath = origPath
	ci.hash = origPath
	ci.destPath = destPath
	ci.decompress = allowDecompression
	*cInfos = append(*cInfos, &ci)

	// Deal with the single file case
	if !fi.IsDir() {
		// This will match first file in sums of the archive
		fis := b.context.GetSums().GetFile(ci.origPath)
		if fis != nil {
			ci.hash = "file:" + fis.Sum()
		}
		return nil
	}

	// Must be a dir
	var subfiles []string
	absOrigPath := filepath.Join(b.contextPath, ci.origPath)

	// Add a trailing / to make sure we only pick up nested files under
	// the dir and not sibling files of the dir that just happen to
	// start with the same chars
	if !strings.HasSuffix(absOrigPath, string(os.PathSeparator)) {
		absOrigPath += string(os.PathSeparator)
	}

	// Need path w/o slash too to find matching dir w/o trailing slash
	absOrigPathNoSlash := absOrigPath[:len(absOrigPath)-1]

	for _, fileInfo := range b.context.GetSums() {
		absFile := filepath.Join(b.contextPath, fileInfo.Name())
		// Any file in the context that starts with the given path will be
		// picked up and its hashcode used.  However, we'll exclude the
		// root dir itself.  We do this for a coupel of reasons:
		// 1 - ADD/COPY will not copy the dir itself, just its children
		//     so there's no reason to include it in the hash calc
		// 2 - the metadata on the dir will change when any child file
		//     changes.  This will lead to a miss in the cache check if that
		//     child file is in the .dockerignore list.
		if strings.HasPrefix(absFile, absOrigPath) && absFile != absOrigPathNoSlash {
			subfiles = append(subfiles, fileInfo.Sum())
		}
	}
	sort.Strings(subfiles)
	hasher := sha256.New()
	hasher.Write([]byte(strings.Join(subfiles, ",")))
	ci.hash = "dir:" + hex.EncodeToString(hasher.Sum(nil))

	return nil
}

func ContainsWildcards(name string) bool {
	for i := 0; i < len(name); i++ {
		ch := name[i]
		if ch == '\\' {
			i++
		} else if ch == '*' || ch == '?' || ch == '[' {
			return true
		}
	}
	return false
}

func (b *Builder) pullImage(name string) (*image.Image, error) {
	remote, tag := parsers.ParseRepositoryTag(name)
	if tag == "" {
		tag = "latest"
	}

	pullRegistryAuth := &cliconfig.AuthConfig{}
	if len(b.AuthConfigs) > 0 {
		// The request came with a full auth config file, we prefer to use that
		repoInfo, err := b.Daemon.RegistryService.ResolveRepository(remote)
		if err != nil {
			return nil, err
		}

		resolvedConfig := registry.ResolveAuthConfig(
			&cliconfig.ConfigFile{AuthConfigs: b.AuthConfigs},
			repoInfo.Index,
		)
		pullRegistryAuth = &resolvedConfig
	}

	imagePullConfig := &graph.ImagePullConfig{
		AuthConfig: pullRegistryAuth,
		OutStream:  ioutils.NopWriteCloser(b.OutOld),
	}

	if err := b.Daemon.Repositories().Pull(remote, tag, imagePullConfig); err != nil {
		return nil, err
	}

	image, err := b.Daemon.Repositories().LookupImage(name)
	if err != nil {
		return nil, err
	}

	return image, nil
}

func (b *Builder) processImageFrom(img *image.Image) error {
	b.image = img.ID

	if img.Config != nil {
		b.Config = img.Config
	}

	// The default path will be blank on Windows (set by HCS)
	if len(b.Config.Env) == 0 && daemon.DefaultPathEnv != "" {
		b.Config.Env = append(b.Config.Env, "PATH="+daemon.DefaultPathEnv)
	}

	// Process ONBUILD triggers if they exist
	if nTriggers := len(b.Config.OnBuild); nTriggers != 0 {
		fmt.Fprintf(b.ErrStream, "# Executing %d build triggers\n", nTriggers)
	}

	// Copy the ONBUILD triggers, and remove them from the config, since the config will be committed.
	onBuildTriggers := b.Config.OnBuild
	b.Config.OnBuild = []string{}

	// parse the ONBUILD triggers by invoking the parser
	for stepN, step := range onBuildTriggers {
		ast, err := parser.Parse(strings.NewReader(step))
		if err != nil {
			return err
		}

		for i, n := range ast.Children {
			switch strings.ToUpper(n.Value) {
			case "ONBUILD":
				return fmt.Errorf("Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed")
			case "MAINTAINER", "FROM":
				return fmt.Errorf("%s isn't allowed as an ONBUILD trigger", n.Value)
			}

			fmt.Fprintf(b.OutStream, "Trigger %d, %s\n", stepN, step)

			if err := b.dispatch(i, n); err != nil {
				return err
			}
		}
	}

	return nil
}

// probeCache checks to see if image-caching is enabled (`b.UtilizeCache`)
// and if so attempts to look up the current `b.image` and `b.Config` pair
// in the current server `b.Daemon`. If an image is found, probeCache returns
// `(true, nil)`. If no image is found, it returns `(false, nil)`. If there
// is any error, it returns `(false, err)`.
func (b *Builder) probeCache() (bool, error) {
	if !b.UtilizeCache || b.cacheBusted {
		return false, nil
	}

	cache, err := b.Daemon.ImageGetCached(b.image, b.Config)
	if err != nil {
		return false, err
	}
	if cache == nil {
		logrus.Debugf("[BUILDER] Cache miss")
		b.cacheBusted = true
		return false, nil
	}

	fmt.Fprintf(b.OutStream, " ---> Using cache\n")
	logrus.Debugf("[BUILDER] Use cached version")
	b.image = cache.ID
	b.Daemon.Graph().Retain(b.id, cache.ID)
	b.activeImages = append(b.activeImages, cache.ID)
	return true, nil
}

func (b *Builder) create() (*daemon.Container, error) {
	if b.image == "" && !b.noBaseImage {
		return nil, fmt.Errorf("Please provide a source image with `from` prior to run")
	}
	b.Config.Image = b.image

	hostConfig := &runconfig.HostConfig{
		CpuShares:    b.cpuShares,
		CpuPeriod:    b.cpuPeriod,
		CpuQuota:     b.cpuQuota,
		CpusetCpus:   b.cpuSetCpus,
		CpusetMems:   b.cpuSetMems,
		CgroupParent: b.cgroupParent,
		Memory:       b.memory,
		MemorySwap:   b.memorySwap,
	}

	config := *b.Config

	// Create the container
	c, warnings, err := b.Daemon.Create(b.Config, hostConfig, "")
	if err != nil {
		return nil, err
	}
	for _, warning := range warnings {
		fmt.Fprintf(b.OutStream, " ---> [Warning] %s\n", warning)
	}

	b.TmpContainers[c.ID] = struct{}{}
	fmt.Fprintf(b.OutStream, " ---> Running in %s\n", stringid.TruncateID(c.ID))

	if config.Cmd.Len() > 0 {
		// override the entry point that may have been picked up from the base image
		s := config.Cmd.Slice()
		c.Path = s[0]
		c.Args = s[1:]
	} else {
		config.Cmd = runconfig.NewCommand()
	}

	return c, nil
}

func (b *Builder) run(c *daemon.Container) error {
	var errCh chan error
	if b.Verbose {
		errCh = c.Attach(nil, b.OutStream, b.ErrStream)
	}

	//start the container
	if err := c.Start(); err != nil {
		return err
	}

	finished := make(chan struct{})
	defer close(finished)
	go func() {
		select {
		case <-b.cancelled:
			logrus.Debugln("Build cancelled, killing container:", c.ID)
			c.Kill()
		case <-finished:
		}
	}()

	if b.Verbose {
		// Block on reading output from container, stop on err or chan closed
		if err := <-errCh; err != nil {
			return err
		}
	}

	// Wait for it to finish
	if ret, _ := c.WaitStop(-1 * time.Second); ret != 0 {
		return &jsonmessage.JSONError{
			Message: fmt.Sprintf("The command '%s' returned a non-zero code: %d", b.Config.Cmd.ToString(), ret),
			Code:    ret,
		}
	}

	return nil
}

func (b *Builder) checkPathForAddition(orig string) error {
	origPath := filepath.Join(b.contextPath, orig)
	origPath, err := filepath.EvalSymlinks(origPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%s: no such file or directory", orig)
		}
		return err
	}
	contextPath, err := filepath.EvalSymlinks(b.contextPath)
	if err != nil {
		return err
	}
	if !strings.HasPrefix(origPath, contextPath) {
		return fmt.Errorf("Forbidden path outside the build context: %s (%s)", orig, origPath)
	}
	if _, err := os.Stat(origPath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%s: no such file or directory", orig)
		}
		return err
	}
	return nil
}

func (b *Builder) addContext(container *daemon.Container, orig, dest string, decompress bool) error {
	var (
		err        error
		destExists = true
		origPath   = filepath.Join(b.contextPath, orig)
		destPath   string
	)

	// Work in daemon-local OS specific file paths
	dest = filepath.FromSlash(dest)

	destPath, err = container.GetResourcePath(dest)
	if err != nil {
		return err
	}

	// Preserve the trailing slash
	if strings.HasSuffix(dest, string(os.PathSeparator)) || dest == "." {
		destPath = destPath + string(os.PathSeparator)
	}

	destStat, err := os.Stat(destPath)
	if err != nil {
		if !os.IsNotExist(err) {
			logrus.Errorf("Error performing os.Stat on %s. %s", destPath, err)
			return err
		}
		destExists = false
	}

	fi, err := os.Stat(origPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%s: no such file or directory", orig)
		}
		return err
	}

	if fi.IsDir() {
		return copyAsDirectory(origPath, destPath, destExists)
	}

	// If we are adding a remote file (or we've been told not to decompress), do not try to untar it
	if decompress {
		// First try to unpack the source as an archive
		// to support the untar feature we need to clean up the path a little bit
		// because tar is very forgiving.  First we need to strip off the archive's
		// filename from the path but this is only added if it does not end in slash
		tarDest := destPath
		if strings.HasSuffix(tarDest, string(os.PathSeparator)) {
			tarDest = filepath.Dir(destPath)
		}

		// try to successfully untar the orig
		if err := chrootarchive.UntarPath(origPath, tarDest); err == nil {
			return nil
		} else if err != io.EOF {
			logrus.Debugf("Couldn't untar %s to %s: %s", origPath, tarDest, err)
		}
	}

	if err := system.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return err
	}
	if err := chrootarchive.CopyWithTar(origPath, destPath); err != nil {
		return err
	}

	resPath := destPath
	if destExists && destStat.IsDir() {
		resPath = filepath.Join(destPath, filepath.Base(origPath))
	}

	return fixPermissions(origPath, resPath, 0, 0, destExists)
}

func copyAsDirectory(source, destination string, destExisted bool) error {
	if err := chrootarchive.CopyWithTar(source, destination); err != nil {
		return err
	}
	return fixPermissions(source, destination, 0, 0, destExisted)
}

func (b *Builder) clearTmp() {
	for c := range b.TmpContainers {
		rmConfig := &daemon.ContainerRmConfig{
			ForceRemove:  true,
			RemoveVolume: true,
		}
		if err := b.Daemon.ContainerRm(c, rmConfig); err != nil {
			fmt.Fprintf(b.OutStream, "Error removing intermediate container %s: %v\n", stringid.TruncateID(c), err)
			return
		}
		delete(b.TmpContainers, c)
		fmt.Fprintf(b.OutStream, "Removing intermediate container %s\n", stringid.TruncateID(c))
	}
}
