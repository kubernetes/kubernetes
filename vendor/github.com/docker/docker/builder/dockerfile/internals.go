package dockerfile

// internals for handling commands. Covers many areas and a lot of
// non-contiguous functionality. Please read the comments.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/stringid"
	"github.com/pkg/errors"
)

func (b *Builder) commit(dispatchState *dispatchState, comment string) error {
	if b.disableCommit {
		return nil
	}
	if !dispatchState.hasFromImage() {
		return errors.New("Please provide a source image with `from` prior to commit")
	}

	runConfigWithCommentCmd := copyRunConfig(dispatchState.runConfig, withCmdComment(comment, b.platform))
	hit, err := b.probeCache(dispatchState, runConfigWithCommentCmd)
	if err != nil || hit {
		return err
	}
	id, err := b.create(runConfigWithCommentCmd)
	if err != nil {
		return err
	}

	return b.commitContainer(dispatchState, id, runConfigWithCommentCmd)
}

func (b *Builder) commitContainer(dispatchState *dispatchState, id string, containerConfig *container.Config) error {
	if b.disableCommit {
		return nil
	}

	commitCfg := &backend.ContainerCommitConfig{
		ContainerCommitConfig: types.ContainerCommitConfig{
			Author: dispatchState.maintainer,
			Pause:  true,
			// TODO: this should be done by Commit()
			Config: copyRunConfig(dispatchState.runConfig),
		},
		ContainerConfig: containerConfig,
	}

	// Commit the container
	imageID, err := b.docker.Commit(id, commitCfg)
	if err != nil {
		return err
	}

	dispatchState.imageID = imageID
	b.buildStages.update(imageID)
	return nil
}

func (b *Builder) exportImage(state *dispatchState, imageMount *imageMount, runConfig *container.Config) error {
	newLayer, err := imageMount.Layer().Commit(b.platform)
	if err != nil {
		return err
	}

	// add an image mount without an image so the layer is properly unmounted
	// if there is an error before we can add the full mount with image
	b.imageSources.Add(newImageMount(nil, newLayer))

	parentImage, ok := imageMount.Image().(*image.Image)
	if !ok {
		return errors.Errorf("unexpected image type")
	}

	newImage := image.NewChildImage(parentImage, image.ChildConfig{
		Author:          state.maintainer,
		ContainerConfig: runConfig,
		DiffID:          newLayer.DiffID(),
		Config:          copyRunConfig(state.runConfig),
	}, parentImage.OS)

	// TODO: it seems strange to marshal this here instead of just passing in the
	// image struct
	config, err := newImage.MarshalJSON()
	if err != nil {
		return errors.Wrap(err, "failed to encode image config")
	}

	exportedImage, err := b.docker.CreateImage(config, state.imageID, parentImage.OS)
	if err != nil {
		return errors.Wrapf(err, "failed to export image")
	}

	state.imageID = exportedImage.ImageID()
	b.imageSources.Add(newImageMount(exportedImage, newLayer))
	b.buildStages.update(state.imageID)
	return nil
}

func (b *Builder) performCopy(state *dispatchState, inst copyInstruction) error {
	srcHash := getSourceHashFromInfos(inst.infos)

	// TODO: should this have been using origPaths instead of srcHash in the comment?
	runConfigWithCommentCmd := copyRunConfig(
		state.runConfig,
		withCmdCommentString(fmt.Sprintf("%s %s in %s ", inst.cmdName, srcHash, inst.dest), b.platform))
	hit, err := b.probeCache(state, runConfigWithCommentCmd)
	if err != nil || hit {
		return err
	}

	imageMount, err := b.imageSources.Get(state.imageID, true)
	if err != nil {
		return errors.Wrapf(err, "failed to get destination image %q", state.imageID)
	}
	destInfo, err := createDestInfo(state.runConfig.WorkingDir, inst, imageMount)
	if err != nil {
		return err
	}

	opts := copyFileOptions{
		decompress: inst.allowLocalDecompression,
		archiver:   b.archiver,
	}
	for _, info := range inst.infos {
		if err := performCopyForInfo(destInfo, info, opts); err != nil {
			return errors.Wrapf(err, "failed to copy files")
		}
	}
	return b.exportImage(state, imageMount, runConfigWithCommentCmd)
}

func createDestInfo(workingDir string, inst copyInstruction, imageMount *imageMount) (copyInfo, error) {
	// Twiddle the destination when it's a relative path - meaning, make it
	// relative to the WORKINGDIR
	dest, err := normaliseDest(workingDir, inst.dest)
	if err != nil {
		return copyInfo{}, errors.Wrapf(err, "invalid %s", inst.cmdName)
	}

	destMount, err := imageMount.Source()
	if err != nil {
		return copyInfo{}, errors.Wrapf(err, "failed to mount copy source")
	}

	return newCopyInfoFromSource(destMount, dest, ""), nil
}

// For backwards compat, if there's just one info then use it as the
// cache look-up string, otherwise hash 'em all into one
func getSourceHashFromInfos(infos []copyInfo) string {
	if len(infos) == 1 {
		return infos[0].hash
	}
	var hashs []string
	for _, info := range infos {
		hashs = append(hashs, info.hash)
	}
	return hashStringSlice("multi", hashs)
}

func hashStringSlice(prefix string, slice []string) string {
	hasher := sha256.New()
	hasher.Write([]byte(strings.Join(slice, ",")))
	return prefix + ":" + hex.EncodeToString(hasher.Sum(nil))
}

type runConfigModifier func(*container.Config)

func copyRunConfig(runConfig *container.Config, modifiers ...runConfigModifier) *container.Config {
	copy := *runConfig
	for _, modifier := range modifiers {
		modifier(&copy)
	}
	return &copy
}

func withCmd(cmd []string) runConfigModifier {
	return func(runConfig *container.Config) {
		runConfig.Cmd = cmd
	}
}

// withCmdComment sets Cmd to a nop comment string. See withCmdCommentString for
// why there are two almost identical versions of this.
func withCmdComment(comment string, platform string) runConfigModifier {
	return func(runConfig *container.Config) {
		runConfig.Cmd = append(getShell(runConfig, platform), "#(nop) ", comment)
	}
}

// withCmdCommentString exists to maintain compatibility with older versions.
// A few instructions (workdir, copy, add) used a nop comment that is a single arg
// where as all the other instructions used a two arg comment string. This
// function implements the single arg version.
func withCmdCommentString(comment string, platform string) runConfigModifier {
	return func(runConfig *container.Config) {
		runConfig.Cmd = append(getShell(runConfig, platform), "#(nop) "+comment)
	}
}

func withEnv(env []string) runConfigModifier {
	return func(runConfig *container.Config) {
		runConfig.Env = env
	}
}

// withEntrypointOverride sets an entrypoint on runConfig if the command is
// not empty. The entrypoint is left unmodified if command is empty.
//
// The dockerfile RUN instruction expect to run without an entrypoint
// so the runConfig entrypoint needs to be modified accordingly. ContainerCreate
// will change a []string{""} entrypoint to nil, so we probe the cache with the
// nil entrypoint.
func withEntrypointOverride(cmd []string, entrypoint []string) runConfigModifier {
	return func(runConfig *container.Config) {
		if len(cmd) > 0 {
			runConfig.Entrypoint = entrypoint
		}
	}
}

// getShell is a helper function which gets the right shell for prefixing the
// shell-form of RUN, ENTRYPOINT and CMD instructions
func getShell(c *container.Config, platform string) []string {
	if 0 == len(c.Shell) {
		return append([]string{}, defaultShellForPlatform(platform)[:]...)
	}
	return append([]string{}, c.Shell[:]...)
}

func (b *Builder) probeCache(dispatchState *dispatchState, runConfig *container.Config) (bool, error) {
	cachedID, err := b.imageProber.Probe(dispatchState.imageID, runConfig)
	if cachedID == "" || err != nil {
		return false, err
	}
	fmt.Fprint(b.Stdout, " ---> Using cache\n")

	dispatchState.imageID = string(cachedID)
	b.buildStages.update(dispatchState.imageID)
	return true, nil
}

var defaultLogConfig = container.LogConfig{Type: "none"}

func (b *Builder) probeAndCreate(dispatchState *dispatchState, runConfig *container.Config) (string, error) {
	if hit, err := b.probeCache(dispatchState, runConfig); err != nil || hit {
		return "", err
	}
	// Set a log config to override any default value set on the daemon
	hostConfig := &container.HostConfig{LogConfig: defaultLogConfig}
	container, err := b.containerManager.Create(runConfig, hostConfig, b.platform)
	return container.ID, err
}

func (b *Builder) create(runConfig *container.Config) (string, error) {
	hostConfig := hostConfigFromOptions(b.options)
	container, err := b.containerManager.Create(runConfig, hostConfig, b.platform)
	if err != nil {
		return "", err
	}
	// TODO: could this be moved into containerManager.Create() ?
	for _, warning := range container.Warnings {
		fmt.Fprintf(b.Stdout, " ---> [Warning] %s\n", warning)
	}
	fmt.Fprintf(b.Stdout, " ---> Running in %s\n", stringid.TruncateID(container.ID))
	return container.ID, nil
}

func hostConfigFromOptions(options *types.ImageBuildOptions) *container.HostConfig {
	resources := container.Resources{
		CgroupParent: options.CgroupParent,
		CPUShares:    options.CPUShares,
		CPUPeriod:    options.CPUPeriod,
		CPUQuota:     options.CPUQuota,
		CpusetCpus:   options.CPUSetCPUs,
		CpusetMems:   options.CPUSetMems,
		Memory:       options.Memory,
		MemorySwap:   options.MemorySwap,
		Ulimits:      options.Ulimits,
	}

	return &container.HostConfig{
		SecurityOpt: options.SecurityOpt,
		Isolation:   options.Isolation,
		ShmSize:     options.ShmSize,
		Resources:   resources,
		NetworkMode: container.NetworkMode(options.NetworkMode),
		// Set a log config to override any default value set on the daemon
		LogConfig:  defaultLogConfig,
		ExtraHosts: options.ExtraHosts,
	}
}
