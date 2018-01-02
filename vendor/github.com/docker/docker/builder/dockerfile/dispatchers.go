package dockerfile

// This file contains the dispatchers for each command. Note that
// `nullDispatch` is not actually a command, but support for commands we parse
// but do nothing with.
//
// See evaluator.go for a higher level discussion of the whole evaluator
// package.

import (
	"bytes"
	"fmt"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/strslice"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/dockerfile/parser"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/signal"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/go-connections/nat"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// ENV foo bar
//
// Sets the environment variable foo to bar, also makes interpolation
// in the dockerfile available from the next statement on via ${foo}.
//
func env(req dispatchRequest) error {
	if len(req.args) == 0 {
		return errAtLeastOneArgument("ENV")
	}

	if len(req.args)%2 != 0 {
		// should never get here, but just in case
		return errTooManyArguments("ENV")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	runConfig := req.state.runConfig
	commitMessage := bytes.NewBufferString("ENV")

	for j := 0; j < len(req.args); j += 2 {
		if len(req.args[j]) == 0 {
			return errBlankCommandNames("ENV")
		}
		name := req.args[j]
		value := req.args[j+1]
		newVar := name + "=" + value
		commitMessage.WriteString(" " + newVar)

		gotOne := false
		for i, envVar := range runConfig.Env {
			envParts := strings.SplitN(envVar, "=", 2)
			compareFrom := envParts[0]
			if equalEnvKeys(compareFrom, name) {
				runConfig.Env[i] = newVar
				gotOne = true
				break
			}
		}
		if !gotOne {
			runConfig.Env = append(runConfig.Env, newVar)
		}
	}

	return req.builder.commit(req.state, commitMessage.String())
}

// MAINTAINER some text <maybe@an.email.address>
//
// Sets the maintainer metadata.
func maintainer(req dispatchRequest) error {
	if len(req.args) != 1 {
		return errExactlyOneArgument("MAINTAINER")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	maintainer := req.args[0]
	req.state.maintainer = maintainer
	return req.builder.commit(req.state, "MAINTAINER "+maintainer)
}

// LABEL some json data describing the image
//
// Sets the Label variable foo to bar,
//
func label(req dispatchRequest) error {
	if len(req.args) == 0 {
		return errAtLeastOneArgument("LABEL")
	}
	if len(req.args)%2 != 0 {
		// should never get here, but just in case
		return errTooManyArguments("LABEL")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	commitStr := "LABEL"
	runConfig := req.state.runConfig

	if runConfig.Labels == nil {
		runConfig.Labels = map[string]string{}
	}

	for j := 0; j < len(req.args); j++ {
		name := req.args[j]
		if name == "" {
			return errBlankCommandNames("LABEL")
		}

		value := req.args[j+1]
		commitStr += " " + name + "=" + value

		runConfig.Labels[name] = value
		j++
	}
	return req.builder.commit(req.state, commitStr)
}

// ADD foo /path
//
// Add the file 'foo' to '/path'. Tarball and Remote URL (git, http) handling
// exist here. If you do not wish to have this automatic handling, use COPY.
//
func add(req dispatchRequest) error {
	if len(req.args) < 2 {
		return errAtLeastTwoArguments("ADD")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	downloader := newRemoteSourceDownloader(req.builder.Output, req.builder.Stdout)
	copier := copierFromDispatchRequest(req, downloader, nil)
	defer copier.Cleanup()
	copyInstruction, err := copier.createCopyInstruction(req.args, "ADD")
	if err != nil {
		return err
	}
	copyInstruction.allowLocalDecompression = true

	return req.builder.performCopy(req.state, copyInstruction)
}

// COPY foo /path
//
// Same as 'ADD' but without the tar and remote url handling.
//
func dispatchCopy(req dispatchRequest) error {
	if len(req.args) < 2 {
		return errAtLeastTwoArguments("COPY")
	}

	flFrom := req.flags.AddString("from", "")
	if err := req.flags.Parse(); err != nil {
		return err
	}

	im, err := req.builder.getImageMount(flFrom)
	if err != nil {
		return errors.Wrapf(err, "invalid from flag value %s", flFrom.Value)
	}

	copier := copierFromDispatchRequest(req, errOnSourceDownload, im)
	defer copier.Cleanup()
	copyInstruction, err := copier.createCopyInstruction(req.args, "COPY")
	if err != nil {
		return err
	}

	return req.builder.performCopy(req.state, copyInstruction)
}

func (b *Builder) getImageMount(fromFlag *Flag) (*imageMount, error) {
	if !fromFlag.IsUsed() {
		// TODO: this could return the source in the default case as well?
		return nil, nil
	}

	var localOnly bool
	imageRefOrID := fromFlag.Value
	stage, err := b.buildStages.get(fromFlag.Value)
	if err != nil {
		return nil, err
	}
	if stage != nil {
		imageRefOrID = stage.ImageID()
		localOnly = true
	}
	return b.imageSources.Get(imageRefOrID, localOnly)
}

// FROM imagename[:tag | @digest] [AS build-stage-name]
//
func from(req dispatchRequest) error {
	stageName, err := parseBuildStageName(req.args)
	if err != nil {
		return err
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	req.builder.imageProber.Reset()
	image, err := req.builder.getFromImage(req.shlex, req.args[0])
	if err != nil {
		return err
	}
	if err := req.builder.buildStages.add(stageName, image); err != nil {
		return err
	}
	req.state.beginStage(stageName, image)
	req.builder.buildArgs.ResetAllowed()
	if image.ImageID() == "" {
		// Typically this means they used "FROM scratch"
		return nil
	}

	return processOnBuild(req)
}

func parseBuildStageName(args []string) (string, error) {
	stageName := ""
	switch {
	case len(args) == 3 && strings.EqualFold(args[1], "as"):
		stageName = strings.ToLower(args[2])
		if ok, _ := regexp.MatchString("^[a-z][a-z0-9-_\\.]*$", stageName); !ok {
			return "", errors.Errorf("invalid name for build stage: %q, name can't start with a number or contain symbols", stageName)
		}
	case len(args) != 1:
		return "", errors.New("FROM requires either one or three arguments")
	}

	return stageName, nil
}

// scratchImage is used as a token for the empty base image.
var scratchImage builder.Image = &image.Image{}

func (b *Builder) getFromImage(shlex *ShellLex, name string) (builder.Image, error) {
	substitutionArgs := []string{}
	for key, value := range b.buildArgs.GetAllMeta() {
		substitutionArgs = append(substitutionArgs, key+"="+value)
	}

	name, err := shlex.ProcessWord(name, substitutionArgs)
	if err != nil {
		return nil, err
	}

	var localOnly bool
	if stage, ok := b.buildStages.getByName(name); ok {
		name = stage.ImageID()
		localOnly = true
	}

	// Windows cannot support a container with no base image unless it is LCOW.
	if name == api.NoBaseImageSpecifier {
		if runtime.GOOS == "windows" {
			if b.platform == "windows" || (b.platform != "windows" && !system.LCOWSupported()) {
				return nil, errors.New("Windows does not support FROM scratch")
			}
		}
		return scratchImage, nil
	}
	imageMount, err := b.imageSources.Get(name, localOnly)
	if err != nil {
		return nil, err
	}
	return imageMount.Image(), nil
}

func processOnBuild(req dispatchRequest) error {
	dispatchState := req.state
	// Process ONBUILD triggers if they exist
	if nTriggers := len(dispatchState.runConfig.OnBuild); nTriggers != 0 {
		word := "trigger"
		if nTriggers > 1 {
			word = "triggers"
		}
		fmt.Fprintf(req.builder.Stderr, "# Executing %d build %s...\n", nTriggers, word)
	}

	// Copy the ONBUILD triggers, and remove them from the config, since the config will be committed.
	onBuildTriggers := dispatchState.runConfig.OnBuild
	dispatchState.runConfig.OnBuild = []string{}

	// Reset stdin settings as all build actions run without stdin
	dispatchState.runConfig.OpenStdin = false
	dispatchState.runConfig.StdinOnce = false

	// parse the ONBUILD triggers by invoking the parser
	for _, step := range onBuildTriggers {
		dockerfile, err := parser.Parse(strings.NewReader(step))
		if err != nil {
			return err
		}

		for _, n := range dockerfile.AST.Children {
			if err := checkDispatch(n); err != nil {
				return err
			}

			upperCasedCmd := strings.ToUpper(n.Value)
			switch upperCasedCmd {
			case "ONBUILD":
				return errors.New("Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed")
			case "MAINTAINER", "FROM":
				return errors.Errorf("%s isn't allowed as an ONBUILD trigger", upperCasedCmd)
			}
		}

		if _, err := dispatchFromDockerfile(req.builder, dockerfile, dispatchState, req.source); err != nil {
			return err
		}
	}
	return nil
}

// ONBUILD RUN echo yo
//
// ONBUILD triggers run when the image is used in a FROM statement.
//
// ONBUILD handling has a lot of special-case functionality, the heading in
// evaluator.go and comments around dispatch() in the same file explain the
// special cases. search for 'OnBuild' in internals.go for additional special
// cases.
//
func onbuild(req dispatchRequest) error {
	if len(req.args) == 0 {
		return errAtLeastOneArgument("ONBUILD")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	triggerInstruction := strings.ToUpper(strings.TrimSpace(req.args[0]))
	switch triggerInstruction {
	case "ONBUILD":
		return errors.New("Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed")
	case "MAINTAINER", "FROM":
		return fmt.Errorf("%s isn't allowed as an ONBUILD trigger", triggerInstruction)
	}

	runConfig := req.state.runConfig
	original := regexp.MustCompile(`(?i)^\s*ONBUILD\s*`).ReplaceAllString(req.original, "")
	runConfig.OnBuild = append(runConfig.OnBuild, original)
	return req.builder.commit(req.state, "ONBUILD "+original)
}

// WORKDIR /tmp
//
// Set the working directory for future RUN/CMD/etc statements.
//
func workdir(req dispatchRequest) error {
	if len(req.args) != 1 {
		return errExactlyOneArgument("WORKDIR")
	}

	err := req.flags.Parse()
	if err != nil {
		return err
	}

	runConfig := req.state.runConfig
	// This is from the Dockerfile and will not necessarily be in platform
	// specific semantics, hence ensure it is converted.
	runConfig.WorkingDir, err = normaliseWorkdir(runConfig.WorkingDir, req.args[0])
	if err != nil {
		return err
	}

	// For performance reasons, we explicitly do a create/mkdir now
	// This avoids having an unnecessary expensive mount/unmount calls
	// (on Windows in particular) during each container create.
	// Prior to 1.13, the mkdir was deferred and not executed at this step.
	if req.builder.disableCommit {
		// Don't call back into the daemon if we're going through docker commit --change "WORKDIR /foo".
		// We've already updated the runConfig and that's enough.
		return nil
	}

	comment := "WORKDIR " + runConfig.WorkingDir
	runConfigWithCommentCmd := copyRunConfig(runConfig, withCmdCommentString(comment, req.builder.platform))
	containerID, err := req.builder.probeAndCreate(req.state, runConfigWithCommentCmd)
	if err != nil || containerID == "" {
		return err
	}
	if err := req.builder.docker.ContainerCreateWorkdir(containerID); err != nil {
		return err
	}

	return req.builder.commitContainer(req.state, containerID, runConfigWithCommentCmd)
}

// RUN some command yo
//
// run a command and commit the image. Args are automatically prepended with
// the current SHELL which defaults to 'sh -c' under linux or 'cmd /S /C' under
// Windows, in the event there is only one argument The difference in processing:
//
// RUN echo hi          # sh -c echo hi       (Linux and LCOW)
// RUN echo hi          # cmd /S /C echo hi   (Windows)
// RUN [ "echo", "hi" ] # echo hi
//
func run(req dispatchRequest) error {
	if !req.state.hasFromImage() {
		return errors.New("Please provide a source image with `from` prior to run")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	stateRunConfig := req.state.runConfig
	args := handleJSONArgs(req.args, req.attributes)
	if !req.attributes["json"] {
		args = append(getShell(stateRunConfig, req.builder.platform), args...)
	}
	cmdFromArgs := strslice.StrSlice(args)
	buildArgs := req.builder.buildArgs.FilterAllowed(stateRunConfig.Env)

	saveCmd := cmdFromArgs
	if len(buildArgs) > 0 {
		saveCmd = prependEnvOnCmd(req.builder.buildArgs, buildArgs, cmdFromArgs)
	}

	runConfigForCacheProbe := copyRunConfig(stateRunConfig,
		withCmd(saveCmd),
		withEntrypointOverride(saveCmd, nil))
	hit, err := req.builder.probeCache(req.state, runConfigForCacheProbe)
	if err != nil || hit {
		return err
	}

	runConfig := copyRunConfig(stateRunConfig,
		withCmd(cmdFromArgs),
		withEnv(append(stateRunConfig.Env, buildArgs...)),
		withEntrypointOverride(saveCmd, strslice.StrSlice{""}))

	// set config as already being escaped, this prevents double escaping on windows
	runConfig.ArgsEscaped = true

	logrus.Debugf("[BUILDER] Command to be executed: %v", runConfig.Cmd)
	cID, err := req.builder.create(runConfig)
	if err != nil {
		return err
	}
	if err := req.builder.containerManager.Run(req.builder.clientCtx, cID, req.builder.Stdout, req.builder.Stderr); err != nil {
		if err, ok := err.(*statusCodeError); ok {
			// TODO: change error type, because jsonmessage.JSONError assumes HTTP
			return &jsonmessage.JSONError{
				Message: fmt.Sprintf(
					"The command '%s' returned a non-zero code: %d",
					strings.Join(runConfig.Cmd, " "), err.StatusCode()),
				Code: err.StatusCode(),
			}
		}
		return err
	}

	return req.builder.commitContainer(req.state, cID, runConfigForCacheProbe)
}

// Derive the command to use for probeCache() and to commit in this container.
// Note that we only do this if there are any build-time env vars.  Also, we
// use the special argument "|#" at the start of the args array. This will
// avoid conflicts with any RUN command since commands can not
// start with | (vertical bar). The "#" (number of build envs) is there to
// help ensure proper cache matches. We don't want a RUN command
// that starts with "foo=abc" to be considered part of a build-time env var.
//
// remove any unreferenced built-in args from the environment variables.
// These args are transparent so resulting image should be the same regardless
// of the value.
func prependEnvOnCmd(buildArgs *buildArgs, buildArgVars []string, cmd strslice.StrSlice) strslice.StrSlice {
	var tmpBuildEnv []string
	for _, env := range buildArgVars {
		key := strings.SplitN(env, "=", 2)[0]
		if buildArgs.IsReferencedOrNotBuiltin(key) {
			tmpBuildEnv = append(tmpBuildEnv, env)
		}
	}

	sort.Strings(tmpBuildEnv)
	tmpEnv := append([]string{fmt.Sprintf("|%d", len(tmpBuildEnv))}, tmpBuildEnv...)
	return strslice.StrSlice(append(tmpEnv, cmd...))
}

// CMD foo
//
// Set the default command to run in the container (which may be empty).
// Argument handling is the same as RUN.
//
func cmd(req dispatchRequest) error {
	if err := req.flags.Parse(); err != nil {
		return err
	}

	runConfig := req.state.runConfig
	cmdSlice := handleJSONArgs(req.args, req.attributes)
	if !req.attributes["json"] {
		cmdSlice = append(getShell(runConfig, req.builder.platform), cmdSlice...)
	}

	runConfig.Cmd = strslice.StrSlice(cmdSlice)
	// set config as already being escaped, this prevents double escaping on windows
	runConfig.ArgsEscaped = true

	if err := req.builder.commit(req.state, fmt.Sprintf("CMD %q", cmdSlice)); err != nil {
		return err
	}

	if len(req.args) != 0 {
		req.state.cmdSet = true
	}

	return nil
}

// parseOptInterval(flag) is the duration of flag.Value, or 0 if
// empty. An error is reported if the value is given and less than minimum duration.
func parseOptInterval(f *Flag) (time.Duration, error) {
	s := f.Value
	if s == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, err
	}
	if d < time.Duration(container.MinimumDuration) {
		return 0, fmt.Errorf("Interval %#v cannot be less than %s", f.name, container.MinimumDuration)
	}
	return d, nil
}

// HEALTHCHECK foo
//
// Set the default healthcheck command to run in the container (which may be empty).
// Argument handling is the same as RUN.
//
func healthcheck(req dispatchRequest) error {
	if len(req.args) == 0 {
		return errAtLeastOneArgument("HEALTHCHECK")
	}
	runConfig := req.state.runConfig
	typ := strings.ToUpper(req.args[0])
	args := req.args[1:]
	if typ == "NONE" {
		if len(args) != 0 {
			return errors.New("HEALTHCHECK NONE takes no arguments")
		}
		test := strslice.StrSlice{typ}
		runConfig.Healthcheck = &container.HealthConfig{
			Test: test,
		}
	} else {
		if runConfig.Healthcheck != nil {
			oldCmd := runConfig.Healthcheck.Test
			if len(oldCmd) > 0 && oldCmd[0] != "NONE" {
				fmt.Fprintf(req.builder.Stdout, "Note: overriding previous HEALTHCHECK: %v\n", oldCmd)
			}
		}

		healthcheck := container.HealthConfig{}

		flInterval := req.flags.AddString("interval", "")
		flTimeout := req.flags.AddString("timeout", "")
		flStartPeriod := req.flags.AddString("start-period", "")
		flRetries := req.flags.AddString("retries", "")

		if err := req.flags.Parse(); err != nil {
			return err
		}

		switch typ {
		case "CMD":
			cmdSlice := handleJSONArgs(args, req.attributes)
			if len(cmdSlice) == 0 {
				return errors.New("Missing command after HEALTHCHECK CMD")
			}

			if !req.attributes["json"] {
				typ = "CMD-SHELL"
			}

			healthcheck.Test = strslice.StrSlice(append([]string{typ}, cmdSlice...))
		default:
			return fmt.Errorf("Unknown type %#v in HEALTHCHECK (try CMD)", typ)
		}

		interval, err := parseOptInterval(flInterval)
		if err != nil {
			return err
		}
		healthcheck.Interval = interval

		timeout, err := parseOptInterval(flTimeout)
		if err != nil {
			return err
		}
		healthcheck.Timeout = timeout

		startPeriod, err := parseOptInterval(flStartPeriod)
		if err != nil {
			return err
		}
		healthcheck.StartPeriod = startPeriod

		if flRetries.Value != "" {
			retries, err := strconv.ParseInt(flRetries.Value, 10, 32)
			if err != nil {
				return err
			}
			if retries < 1 {
				return fmt.Errorf("--retries must be at least 1 (not %d)", retries)
			}
			healthcheck.Retries = int(retries)
		} else {
			healthcheck.Retries = 0
		}

		runConfig.Healthcheck = &healthcheck
	}

	return req.builder.commit(req.state, fmt.Sprintf("HEALTHCHECK %q", runConfig.Healthcheck))
}

// ENTRYPOINT /usr/sbin/nginx
//
// Set the entrypoint to /usr/sbin/nginx. Will accept the CMD as the arguments
// to /usr/sbin/nginx. Uses the default shell if not in JSON format.
//
// Handles command processing similar to CMD and RUN, only req.runConfig.Entrypoint
// is initialized at newBuilder time instead of through argument parsing.
//
func entrypoint(req dispatchRequest) error {
	if err := req.flags.Parse(); err != nil {
		return err
	}

	runConfig := req.state.runConfig
	parsed := handleJSONArgs(req.args, req.attributes)

	switch {
	case req.attributes["json"]:
		// ENTRYPOINT ["echo", "hi"]
		runConfig.Entrypoint = strslice.StrSlice(parsed)
	case len(parsed) == 0:
		// ENTRYPOINT []
		runConfig.Entrypoint = nil
	default:
		// ENTRYPOINT echo hi
		runConfig.Entrypoint = strslice.StrSlice(append(getShell(runConfig, req.builder.platform), parsed[0]))
	}

	// when setting the entrypoint if a CMD was not explicitly set then
	// set the command to nil
	if !req.state.cmdSet {
		runConfig.Cmd = nil
	}

	return req.builder.commit(req.state, fmt.Sprintf("ENTRYPOINT %q", runConfig.Entrypoint))
}

// EXPOSE 6667/tcp 7000/tcp
//
// Expose ports for links and port mappings. This all ends up in
// req.runConfig.ExposedPorts for runconfig.
//
func expose(req dispatchRequest) error {
	portsTab := req.args

	if len(req.args) == 0 {
		return errAtLeastOneArgument("EXPOSE")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	runConfig := req.state.runConfig
	if runConfig.ExposedPorts == nil {
		runConfig.ExposedPorts = make(nat.PortSet)
	}

	ports, _, err := nat.ParsePortSpecs(portsTab)
	if err != nil {
		return err
	}

	// instead of using ports directly, we build a list of ports and sort it so
	// the order is consistent. This prevents cache burst where map ordering
	// changes between builds
	portList := make([]string, len(ports))
	var i int
	for port := range ports {
		if _, exists := runConfig.ExposedPorts[port]; !exists {
			runConfig.ExposedPorts[port] = struct{}{}
		}
		portList[i] = string(port)
		i++
	}
	sort.Strings(portList)
	return req.builder.commit(req.state, "EXPOSE "+strings.Join(portList, " "))
}

// USER foo
//
// Set the user to 'foo' for future commands and when running the
// ENTRYPOINT/CMD at container run time.
//
func user(req dispatchRequest) error {
	if len(req.args) != 1 {
		return errExactlyOneArgument("USER")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	req.state.runConfig.User = req.args[0]
	return req.builder.commit(req.state, fmt.Sprintf("USER %v", req.args))
}

// VOLUME /foo
//
// Expose the volume /foo for use. Will also accept the JSON array form.
//
func volume(req dispatchRequest) error {
	if len(req.args) == 0 {
		return errAtLeastOneArgument("VOLUME")
	}

	if err := req.flags.Parse(); err != nil {
		return err
	}

	runConfig := req.state.runConfig
	if runConfig.Volumes == nil {
		runConfig.Volumes = map[string]struct{}{}
	}
	for _, v := range req.args {
		v = strings.TrimSpace(v)
		if v == "" {
			return errors.New("VOLUME specified can not be an empty string")
		}
		runConfig.Volumes[v] = struct{}{}
	}
	return req.builder.commit(req.state, fmt.Sprintf("VOLUME %v", req.args))
}

// STOPSIGNAL signal
//
// Set the signal that will be used to kill the container.
func stopSignal(req dispatchRequest) error {
	if len(req.args) != 1 {
		return errExactlyOneArgument("STOPSIGNAL")
	}

	sig := req.args[0]
	_, err := signal.ParseSignal(sig)
	if err != nil {
		return err
	}

	req.state.runConfig.StopSignal = sig
	return req.builder.commit(req.state, fmt.Sprintf("STOPSIGNAL %v", req.args))
}

// ARG name[=value]
//
// Adds the variable foo to the trusted list of variables that can be passed
// to builder using the --build-arg flag for expansion/substitution or passing to 'run'.
// Dockerfile author may optionally set a default value of this variable.
func arg(req dispatchRequest) error {
	if len(req.args) != 1 {
		return errExactlyOneArgument("ARG")
	}

	var (
		name       string
		newValue   string
		hasDefault bool
	)

	arg := req.args[0]
	// 'arg' can just be a name or name-value pair. Note that this is different
	// from 'env' that handles the split of name and value at the parser level.
	// The reason for doing it differently for 'arg' is that we support just
	// defining an arg and not assign it a value (while 'env' always expects a
	// name-value pair). If possible, it will be good to harmonize the two.
	if strings.Contains(arg, "=") {
		parts := strings.SplitN(arg, "=", 2)
		if len(parts[0]) == 0 {
			return errBlankCommandNames("ARG")
		}

		name = parts[0]
		newValue = parts[1]
		hasDefault = true
	} else {
		name = arg
		hasDefault = false
	}

	var value *string
	if hasDefault {
		value = &newValue
	}
	req.builder.buildArgs.AddArg(name, value)

	// Arg before FROM doesn't add a layer
	if !req.state.hasFromImage() {
		req.builder.buildArgs.AddMetaArg(name, value)
		return nil
	}
	return req.builder.commit(req.state, "ARG "+arg)
}

// SHELL powershell -command
//
// Set the non-default shell to use.
func shell(req dispatchRequest) error {
	if err := req.flags.Parse(); err != nil {
		return err
	}
	shellSlice := handleJSONArgs(req.args, req.attributes)
	switch {
	case len(shellSlice) == 0:
		// SHELL []
		return errAtLeastOneArgument("SHELL")
	case req.attributes["json"]:
		// SHELL ["powershell", "-command"]
		req.state.runConfig.Shell = strslice.StrSlice(shellSlice)
	default:
		// SHELL powershell -command - not JSON
		return errNotJSON("SHELL", req.original)
	}
	return req.builder.commit(req.state, fmt.Sprintf("SHELL %v", shellSlice))
}

func errAtLeastOneArgument(command string) error {
	return fmt.Errorf("%s requires at least one argument", command)
}

func errExactlyOneArgument(command string) error {
	return fmt.Errorf("%s requires exactly one argument", command)
}

func errAtLeastTwoArguments(command string) error {
	return fmt.Errorf("%s requires at least two arguments", command)
}

func errBlankCommandNames(command string) error {
	return fmt.Errorf("%s names can not be blank", command)
}

func errTooManyArguments(command string) error {
	return fmt.Errorf("Bad input to %s, too many arguments", command)
}
