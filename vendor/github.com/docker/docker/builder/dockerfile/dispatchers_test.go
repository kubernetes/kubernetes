package dockerfile

import (
	"fmt"
	"runtime"
	"testing"

	"bytes"
	"context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/strslice"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/dockerfile/parser"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/testutil"
	"github.com/docker/go-connections/nat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type commandWithFunction struct {
	name     string
	function func(args []string) error
}

func withArgs(f dispatcher) func([]string) error {
	return func(args []string) error {
		return f(dispatchRequest{args: args})
	}
}

func withBuilderAndArgs(builder *Builder, f dispatcher) func([]string) error {
	return func(args []string) error {
		return f(defaultDispatchReq(builder, args...))
	}
}

func defaultDispatchReq(builder *Builder, args ...string) dispatchRequest {
	return dispatchRequest{
		builder: builder,
		args:    args,
		flags:   NewBFlags(),
		shlex:   NewShellLex(parser.DefaultEscapeToken),
		state:   &dispatchState{runConfig: &container.Config{}},
	}
}

func newBuilderWithMockBackend() *Builder {
	mockBackend := &MockBackend{}
	ctx := context.Background()
	b := &Builder{
		options:       &types.ImageBuildOptions{},
		docker:        mockBackend,
		buildArgs:     newBuildArgs(make(map[string]*string)),
		Stdout:        new(bytes.Buffer),
		clientCtx:     ctx,
		disableCommit: true,
		imageSources: newImageSources(ctx, builderOptions{
			Options: &types.ImageBuildOptions{},
			Backend: mockBackend,
		}),
		buildStages:      newBuildStages(),
		imageProber:      newImageProber(mockBackend, nil, runtime.GOOS, false),
		containerManager: newContainerManager(mockBackend),
	}
	return b
}

func TestCommandsExactlyOneArgument(t *testing.T) {
	commands := []commandWithFunction{
		{"MAINTAINER", withArgs(maintainer)},
		{"WORKDIR", withArgs(workdir)},
		{"USER", withArgs(user)},
		{"STOPSIGNAL", withArgs(stopSignal)},
	}

	for _, command := range commands {
		err := command.function([]string{})
		assert.EqualError(t, err, errExactlyOneArgument(command.name).Error())
	}
}

func TestCommandsAtLeastOneArgument(t *testing.T) {
	commands := []commandWithFunction{
		{"ENV", withArgs(env)},
		{"LABEL", withArgs(label)},
		{"ONBUILD", withArgs(onbuild)},
		{"HEALTHCHECK", withArgs(healthcheck)},
		{"EXPOSE", withArgs(expose)},
		{"VOLUME", withArgs(volume)},
	}

	for _, command := range commands {
		err := command.function([]string{})
		assert.EqualError(t, err, errAtLeastOneArgument(command.name).Error())
	}
}

func TestCommandsAtLeastTwoArguments(t *testing.T) {
	commands := []commandWithFunction{
		{"ADD", withArgs(add)},
		{"COPY", withArgs(dispatchCopy)}}

	for _, command := range commands {
		err := command.function([]string{"arg1"})
		assert.EqualError(t, err, errAtLeastTwoArguments(command.name).Error())
	}
}

func TestCommandsTooManyArguments(t *testing.T) {
	commands := []commandWithFunction{
		{"ENV", withArgs(env)},
		{"LABEL", withArgs(label)}}

	for _, command := range commands {
		err := command.function([]string{"arg1", "arg2", "arg3"})
		assert.EqualError(t, err, errTooManyArguments(command.name).Error())
	}
}

func TestCommandsBlankNames(t *testing.T) {
	builder := newBuilderWithMockBackend()
	commands := []commandWithFunction{
		{"ENV", withBuilderAndArgs(builder, env)},
		{"LABEL", withBuilderAndArgs(builder, label)},
	}

	for _, command := range commands {
		err := command.function([]string{"", ""})
		assert.EqualError(t, err, errBlankCommandNames(command.name).Error())
	}
}

func TestEnv2Variables(t *testing.T) {
	b := newBuilderWithMockBackend()

	args := []string{"var1", "val1", "var2", "val2"}
	req := defaultDispatchReq(b, args...)
	err := env(req)
	require.NoError(t, err)

	expected := []string{
		fmt.Sprintf("%s=%s", args[0], args[1]),
		fmt.Sprintf("%s=%s", args[2], args[3]),
	}
	assert.Equal(t, expected, req.state.runConfig.Env)
}

func TestEnvValueWithExistingRunConfigEnv(t *testing.T) {
	b := newBuilderWithMockBackend()

	args := []string{"var1", "val1"}
	req := defaultDispatchReq(b, args...)
	req.state.runConfig.Env = []string{"var1=old", "var2=fromenv"}
	err := env(req)
	require.NoError(t, err)

	expected := []string{
		fmt.Sprintf("%s=%s", args[0], args[1]),
		"var2=fromenv",
	}
	assert.Equal(t, expected, req.state.runConfig.Env)
}

func TestMaintainer(t *testing.T) {
	maintainerEntry := "Some Maintainer <maintainer@example.com>"

	b := newBuilderWithMockBackend()
	req := defaultDispatchReq(b, maintainerEntry)
	err := maintainer(req)
	require.NoError(t, err)
	assert.Equal(t, maintainerEntry, req.state.maintainer)
}

func TestLabel(t *testing.T) {
	labelName := "label"
	labelValue := "value"

	labelEntry := []string{labelName, labelValue}
	b := newBuilderWithMockBackend()
	req := defaultDispatchReq(b, labelEntry...)
	err := label(req)
	require.NoError(t, err)

	require.Contains(t, req.state.runConfig.Labels, labelName)
	assert.Equal(t, req.state.runConfig.Labels[labelName], labelValue)
}

func TestFromScratch(t *testing.T) {
	b := newBuilderWithMockBackend()
	req := defaultDispatchReq(b, "scratch")
	err := from(req)

	if runtime.GOOS == "windows" && !system.LCOWSupported() {
		assert.EqualError(t, err, "Windows does not support FROM scratch")
		return
	}

	require.NoError(t, err)
	assert.True(t, req.state.hasFromImage())
	assert.Equal(t, "", req.state.imageID)
	// Windows does not set the default path. TODO @jhowardmsft LCOW support. This will need revisiting as we get further into the implementation
	expected := "PATH=" + system.DefaultPathEnv(runtime.GOOS)
	if runtime.GOOS == "windows" {
		expected = ""
	}
	assert.Equal(t, []string{expected}, req.state.runConfig.Env)
}

func TestFromWithArg(t *testing.T) {
	tag, expected := ":sometag", "expectedthisid"

	getImage := func(name string) (builder.Image, builder.ReleaseableLayer, error) {
		assert.Equal(t, "alpine"+tag, name)
		return &mockImage{id: "expectedthisid"}, nil, nil
	}
	b := newBuilderWithMockBackend()
	b.docker.(*MockBackend).getImageFunc = getImage

	require.NoError(t, arg(defaultDispatchReq(b, "THETAG="+tag)))
	req := defaultDispatchReq(b, "alpine${THETAG}")
	err := from(req)

	require.NoError(t, err)
	assert.Equal(t, expected, req.state.imageID)
	assert.Equal(t, expected, req.state.baseImage.ImageID())
	assert.Len(t, b.buildArgs.GetAllAllowed(), 0)
	assert.Len(t, b.buildArgs.GetAllMeta(), 1)
}

func TestFromWithUndefinedArg(t *testing.T) {
	tag, expected := "sometag", "expectedthisid"

	getImage := func(name string) (builder.Image, builder.ReleaseableLayer, error) {
		assert.Equal(t, "alpine", name)
		return &mockImage{id: "expectedthisid"}, nil, nil
	}
	b := newBuilderWithMockBackend()
	b.docker.(*MockBackend).getImageFunc = getImage
	b.options.BuildArgs = map[string]*string{"THETAG": &tag}

	req := defaultDispatchReq(b, "alpine${THETAG}")
	err := from(req)
	require.NoError(t, err)
	assert.Equal(t, expected, req.state.imageID)
}

func TestFromMultiStageWithScratchNamedStage(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows does not support scratch")
	}
	b := newBuilderWithMockBackend()
	req := defaultDispatchReq(b, "scratch", "AS", "base")

	require.NoError(t, from(req))
	assert.True(t, req.state.hasFromImage())

	req.args = []string{"base"}
	require.NoError(t, from(req))
	assert.True(t, req.state.hasFromImage())
}

func TestOnbuildIllegalTriggers(t *testing.T) {
	triggers := []struct{ command, expectedError string }{
		{"ONBUILD", "Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed"},
		{"MAINTAINER", "MAINTAINER isn't allowed as an ONBUILD trigger"},
		{"FROM", "FROM isn't allowed as an ONBUILD trigger"}}

	for _, trigger := range triggers {
		b := newBuilderWithMockBackend()

		err := onbuild(defaultDispatchReq(b, trigger.command))
		testutil.ErrorContains(t, err, trigger.expectedError)
	}
}

func TestOnbuild(t *testing.T) {
	b := newBuilderWithMockBackend()

	req := defaultDispatchReq(b, "ADD", ".", "/app/src")
	req.original = "ONBUILD ADD . /app/src"
	req.state.runConfig = &container.Config{}

	err := onbuild(req)
	require.NoError(t, err)
	assert.Equal(t, "ADD . /app/src", req.state.runConfig.OnBuild[0])
}

func TestWorkdir(t *testing.T) {
	b := newBuilderWithMockBackend()
	workingDir := "/app"
	if runtime.GOOS == "windows" {
		workingDir = "C:\app"
	}

	req := defaultDispatchReq(b, workingDir)
	err := workdir(req)
	require.NoError(t, err)
	assert.Equal(t, workingDir, req.state.runConfig.WorkingDir)
}

func TestCmd(t *testing.T) {
	b := newBuilderWithMockBackend()
	command := "./executable"

	req := defaultDispatchReq(b, command)
	err := cmd(req)
	require.NoError(t, err)

	var expectedCommand strslice.StrSlice
	if runtime.GOOS == "windows" {
		expectedCommand = strslice.StrSlice(append([]string{"cmd"}, "/S", "/C", command))
	} else {
		expectedCommand = strslice.StrSlice(append([]string{"/bin/sh"}, "-c", command))
	}

	assert.Equal(t, expectedCommand, req.state.runConfig.Cmd)
	assert.True(t, req.state.cmdSet)
}

func TestHealthcheckNone(t *testing.T) {
	b := newBuilderWithMockBackend()

	req := defaultDispatchReq(b, "NONE")
	err := healthcheck(req)
	require.NoError(t, err)

	require.NotNil(t, req.state.runConfig.Healthcheck)
	assert.Equal(t, []string{"NONE"}, req.state.runConfig.Healthcheck.Test)
}

func TestHealthcheckCmd(t *testing.T) {
	b := newBuilderWithMockBackend()

	args := []string{"CMD", "curl", "-f", "http://localhost/", "||", "exit", "1"}
	req := defaultDispatchReq(b, args...)
	err := healthcheck(req)
	require.NoError(t, err)

	require.NotNil(t, req.state.runConfig.Healthcheck)
	expectedTest := []string{"CMD-SHELL", "curl -f http://localhost/ || exit 1"}
	assert.Equal(t, expectedTest, req.state.runConfig.Healthcheck.Test)
}

func TestEntrypoint(t *testing.T) {
	b := newBuilderWithMockBackend()
	entrypointCmd := "/usr/sbin/nginx"

	req := defaultDispatchReq(b, entrypointCmd)
	err := entrypoint(req)
	require.NoError(t, err)
	require.NotNil(t, req.state.runConfig.Entrypoint)

	var expectedEntrypoint strslice.StrSlice
	if runtime.GOOS == "windows" {
		expectedEntrypoint = strslice.StrSlice(append([]string{"cmd"}, "/S", "/C", entrypointCmd))
	} else {
		expectedEntrypoint = strslice.StrSlice(append([]string{"/bin/sh"}, "-c", entrypointCmd))
	}
	assert.Equal(t, expectedEntrypoint, req.state.runConfig.Entrypoint)
}

func TestExpose(t *testing.T) {
	b := newBuilderWithMockBackend()

	exposedPort := "80"
	req := defaultDispatchReq(b, exposedPort)
	err := expose(req)
	require.NoError(t, err)

	require.NotNil(t, req.state.runConfig.ExposedPorts)
	require.Len(t, req.state.runConfig.ExposedPorts, 1)

	portsMapping, err := nat.ParsePortSpec(exposedPort)
	require.NoError(t, err)
	assert.Contains(t, req.state.runConfig.ExposedPorts, portsMapping[0].Port)
}

func TestUser(t *testing.T) {
	b := newBuilderWithMockBackend()
	userCommand := "foo"

	req := defaultDispatchReq(b, userCommand)
	err := user(req)
	require.NoError(t, err)
	assert.Equal(t, userCommand, req.state.runConfig.User)
}

func TestVolume(t *testing.T) {
	b := newBuilderWithMockBackend()

	exposedVolume := "/foo"

	req := defaultDispatchReq(b, exposedVolume)
	err := volume(req)
	require.NoError(t, err)

	require.NotNil(t, req.state.runConfig.Volumes)
	assert.Len(t, req.state.runConfig.Volumes, 1)
	assert.Contains(t, req.state.runConfig.Volumes, exposedVolume)
}

func TestStopSignal(t *testing.T) {
	b := newBuilderWithMockBackend()
	signal := "SIGKILL"

	req := defaultDispatchReq(b, signal)
	err := stopSignal(req)
	require.NoError(t, err)
	assert.Equal(t, signal, req.state.runConfig.StopSignal)
}

func TestArg(t *testing.T) {
	b := newBuilderWithMockBackend()

	argName := "foo"
	argVal := "bar"
	argDef := fmt.Sprintf("%s=%s", argName, argVal)

	err := arg(defaultDispatchReq(b, argDef))
	require.NoError(t, err)

	expected := map[string]string{argName: argVal}
	assert.Equal(t, expected, b.buildArgs.GetAllAllowed())
}

func TestShell(t *testing.T) {
	b := newBuilderWithMockBackend()

	shellCmd := "powershell"
	req := defaultDispatchReq(b, shellCmd)
	req.attributes = map[string]bool{"json": true}

	err := shell(req)
	require.NoError(t, err)

	expectedShell := strslice.StrSlice([]string{shellCmd})
	assert.Equal(t, expectedShell, req.state.runConfig.Shell)
}

func TestParseOptInterval(t *testing.T) {
	flInterval := &Flag{
		name:     "interval",
		flagType: stringType,
		Value:    "50ns",
	}
	_, err := parseOptInterval(flInterval)
	testutil.ErrorContains(t, err, "cannot be less than 1ms")

	flInterval.Value = "1ms"
	_, err = parseOptInterval(flInterval)
	require.NoError(t, err)
}

func TestPrependEnvOnCmd(t *testing.T) {
	buildArgs := newBuildArgs(nil)
	buildArgs.AddArg("NO_PROXY", nil)

	args := []string{"sorted=nope", "args=not", "http_proxy=foo", "NO_PROXY=YA"}
	cmd := []string{"foo", "bar"}
	cmdWithEnv := prependEnvOnCmd(buildArgs, args, cmd)
	expected := strslice.StrSlice([]string{
		"|3", "NO_PROXY=YA", "args=not", "sorted=nope", "foo", "bar"})
	assert.Equal(t, expected, cmdWithEnv)
}

func TestRunWithBuildArgs(t *testing.T) {
	b := newBuilderWithMockBackend()
	b.buildArgs.argsFromOptions["HTTP_PROXY"] = strPtr("FOO")
	b.disableCommit = false

	runConfig := &container.Config{}
	origCmd := strslice.StrSlice([]string{"cmd", "in", "from", "image"})
	cmdWithShell := strslice.StrSlice(append(getShell(runConfig, runtime.GOOS), "echo foo"))
	envVars := []string{"|1", "one=two"}
	cachedCmd := strslice.StrSlice(append(envVars, cmdWithShell...))

	imageCache := &mockImageCache{
		getCacheFunc: func(parentID string, cfg *container.Config) (string, error) {
			// Check the runConfig.Cmd sent to probeCache()
			assert.Equal(t, cachedCmd, cfg.Cmd)
			assert.Equal(t, strslice.StrSlice(nil), cfg.Entrypoint)
			return "", nil
		},
	}

	mockBackend := b.docker.(*MockBackend)
	mockBackend.makeImageCacheFunc = func(_ []string, _ string) builder.ImageCache {
		return imageCache
	}
	b.imageProber = newImageProber(mockBackend, nil, runtime.GOOS, false)
	mockBackend.getImageFunc = func(_ string) (builder.Image, builder.ReleaseableLayer, error) {
		return &mockImage{
			id:     "abcdef",
			config: &container.Config{Cmd: origCmd},
		}, nil, nil
	}
	mockBackend.containerCreateFunc = func(config types.ContainerCreateConfig) (container.ContainerCreateCreatedBody, error) {
		// Check the runConfig.Cmd sent to create()
		assert.Equal(t, cmdWithShell, config.Config.Cmd)
		assert.Contains(t, config.Config.Env, "one=two")
		assert.Equal(t, strslice.StrSlice{""}, config.Config.Entrypoint)
		return container.ContainerCreateCreatedBody{ID: "12345"}, nil
	}
	mockBackend.commitFunc = func(cID string, cfg *backend.ContainerCommitConfig) (string, error) {
		// Check the runConfig.Cmd sent to commit()
		assert.Equal(t, origCmd, cfg.Config.Cmd)
		assert.Equal(t, cachedCmd, cfg.ContainerConfig.Cmd)
		assert.Equal(t, strslice.StrSlice(nil), cfg.Config.Entrypoint)
		return "", nil
	}

	req := defaultDispatchReq(b, "abcdef")
	require.NoError(t, from(req))
	b.buildArgs.AddArg("one", strPtr("two"))

	req.args = []string{"echo foo"}
	require.NoError(t, run(req))

	// Check that runConfig.Cmd has not been modified by run
	assert.Equal(t, origCmd, req.state.runConfig.Cmd)
}
