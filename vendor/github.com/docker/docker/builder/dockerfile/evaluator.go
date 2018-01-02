// Package dockerfile is the evaluation step in the Dockerfile parse/evaluate pipeline.
//
// It incorporates a dispatch table based on the parser.Node values (see the
// parser package for more information) that are yielded from the parser itself.
// Calling newBuilder with the BuildOpts struct can be used to customize the
// experience for execution purposes only. Parsing is controlled in the parser
// package, and this division of responsibility should be respected.
//
// Please see the jump table targets for the actual invocations, most of which
// will call out to the functions in internals.go to deal with their tasks.
//
// ONBUILD is a special case, which is covered in the onbuild() func in
// dispatchers.go.
//
// The evaluator uses the concept of "steps", which are usually each processable
// line in the Dockerfile. Each step is numbered and certain actions are taken
// before and after each step, such as creating an image ID and removing temporary
// containers and images. Note that ONBUILD creates a kinda-sorta "sub run" which
// includes its own set of steps (usually only one of them).
package dockerfile

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/dockerfile/command"
	"github.com/docker/docker/builder/dockerfile/parser"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/runconfig/opts"
	"github.com/pkg/errors"
)

// Environment variable interpolation will happen on these statements only.
var replaceEnvAllowed = map[string]bool{
	command.Env:        true,
	command.Label:      true,
	command.Add:        true,
	command.Copy:       true,
	command.Workdir:    true,
	command.Expose:     true,
	command.Volume:     true,
	command.User:       true,
	command.StopSignal: true,
	command.Arg:        true,
}

// Certain commands are allowed to have their args split into more
// words after env var replacements. Meaning:
//   ENV foo="123 456"
//   EXPOSE $foo
// should result in the same thing as:
//   EXPOSE 123 456
// and not treat "123 456" as a single word.
// Note that: EXPOSE "$foo" and EXPOSE $foo are not the same thing.
// Quotes will cause it to still be treated as single word.
var allowWordExpansion = map[string]bool{
	command.Expose: true,
}

type dispatchRequest struct {
	builder    *Builder // TODO: replace this with a smaller interface
	args       []string
	attributes map[string]bool
	flags      *BFlags
	original   string
	shlex      *ShellLex
	state      *dispatchState
	source     builder.Source
}

func newDispatchRequestFromOptions(options dispatchOptions, builder *Builder, args []string) dispatchRequest {
	return dispatchRequest{
		builder:    builder,
		args:       args,
		attributes: options.node.Attributes,
		original:   options.node.Original,
		flags:      NewBFlagsWithArgs(options.node.Flags),
		shlex:      options.shlex,
		state:      options.state,
		source:     options.source,
	}
}

type dispatcher func(dispatchRequest) error

var evaluateTable map[string]dispatcher

func init() {
	evaluateTable = map[string]dispatcher{
		command.Add:         add,
		command.Arg:         arg,
		command.Cmd:         cmd,
		command.Copy:        dispatchCopy, // copy() is a go builtin
		command.Entrypoint:  entrypoint,
		command.Env:         env,
		command.Expose:      expose,
		command.From:        from,
		command.Healthcheck: healthcheck,
		command.Label:       label,
		command.Maintainer:  maintainer,
		command.Onbuild:     onbuild,
		command.Run:         run,
		command.Shell:       shell,
		command.StopSignal:  stopSignal,
		command.User:        user,
		command.Volume:      volume,
		command.Workdir:     workdir,
	}
}

func formatStep(stepN int, stepTotal int) string {
	return fmt.Sprintf("%d/%d", stepN+1, stepTotal)
}

// This method is the entrypoint to all statement handling routines.
//
// Almost all nodes will have this structure:
// Child[Node, Node, Node] where Child is from parser.Node.Children and each
// node comes from parser.Node.Next. This forms a "line" with a statement and
// arguments and we process them in this normalized form by hitting
// evaluateTable with the leaf nodes of the command and the Builder object.
//
// ONBUILD is a special case; in this case the parser will emit:
// Child[Node, Child[Node, Node...]] where the first node is the literal
// "onbuild" and the child entrypoint is the command of the ONBUILD statement,
// such as `RUN` in ONBUILD RUN foo. There is special case logic in here to
// deal with that, at least until it becomes more of a general concern with new
// features.
func (b *Builder) dispatch(options dispatchOptions) (*dispatchState, error) {
	node := options.node
	cmd := node.Value
	upperCasedCmd := strings.ToUpper(cmd)

	// To ensure the user is given a decent error message if the platform
	// on which the daemon is running does not support a builder command.
	if err := platformSupports(strings.ToLower(cmd)); err != nil {
		buildsFailed.WithValues(metricsCommandNotSupportedError).Inc()
		return nil, err
	}

	msg := bytes.NewBufferString(fmt.Sprintf("Step %s : %s%s",
		options.stepMsg, upperCasedCmd, formatFlags(node.Flags)))

	args := []string{}
	ast := node
	if cmd == command.Onbuild {
		var err error
		ast, args, err = handleOnBuildNode(node, msg)
		if err != nil {
			return nil, err
		}
	}

	runConfigEnv := options.state.runConfig.Env
	envs := append(runConfigEnv, b.buildArgs.FilterAllowed(runConfigEnv)...)
	processFunc := createProcessWordFunc(options.shlex, cmd, envs)
	words, err := getDispatchArgsFromNode(ast, processFunc, msg)
	if err != nil {
		buildsFailed.WithValues(metricsErrorProcessingCommandsError).Inc()
		return nil, err
	}
	args = append(args, words...)

	fmt.Fprintln(b.Stdout, msg.String())

	f, ok := evaluateTable[cmd]
	if !ok {
		buildsFailed.WithValues(metricsUnknownInstructionError).Inc()
		return nil, fmt.Errorf("unknown instruction: %s", upperCasedCmd)
	}
	options.state.updateRunConfig()
	err = f(newDispatchRequestFromOptions(options, b, args))
	return options.state, err
}

type dispatchOptions struct {
	state   *dispatchState
	stepMsg string
	node    *parser.Node
	shlex   *ShellLex
	source  builder.Source
}

// dispatchState is a data object which is modified by dispatchers
type dispatchState struct {
	runConfig  *container.Config
	maintainer string
	cmdSet     bool
	imageID    string
	baseImage  builder.Image
	stageName  string
}

func newDispatchState() *dispatchState {
	return &dispatchState{runConfig: &container.Config{}}
}

func (s *dispatchState) updateRunConfig() {
	s.runConfig.Image = s.imageID
}

// hasFromImage returns true if the builder has processed a `FROM <image>` line
func (s *dispatchState) hasFromImage() bool {
	return s.imageID != "" || (s.baseImage != nil && s.baseImage.ImageID() == "")
}

func (s *dispatchState) isCurrentStage(target string) bool {
	if target == "" {
		return false
	}
	return strings.EqualFold(s.stageName, target)
}

func (s *dispatchState) beginStage(stageName string, image builder.Image) {
	s.stageName = stageName
	s.imageID = image.ImageID()

	if image.RunConfig() != nil {
		s.runConfig = image.RunConfig()
	} else {
		s.runConfig = &container.Config{}
	}
	s.baseImage = image
	s.setDefaultPath()
}

// Add the default PATH to runConfig.ENV if one exists for the platform and there
// is no PATH set. Note that Windows containers on Windows won't have one as it's set by HCS
func (s *dispatchState) setDefaultPath() {
	// TODO @jhowardmsft LCOW Support - This will need revisiting later
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}
	if system.DefaultPathEnv(platform) == "" {
		return
	}
	envMap := opts.ConvertKVStringsToMap(s.runConfig.Env)
	if _, ok := envMap["PATH"]; !ok {
		s.runConfig.Env = append(s.runConfig.Env, "PATH="+system.DefaultPathEnv(platform))
	}
}

func handleOnBuildNode(ast *parser.Node, msg *bytes.Buffer) (*parser.Node, []string, error) {
	if ast.Next == nil {
		return nil, nil, errors.New("ONBUILD requires at least one argument")
	}
	ast = ast.Next.Children[0]
	msg.WriteString(" " + ast.Value + formatFlags(ast.Flags))
	return ast, []string{ast.Value}, nil
}

func formatFlags(flags []string) string {
	if len(flags) > 0 {
		return " " + strings.Join(flags, " ")
	}
	return ""
}

func getDispatchArgsFromNode(ast *parser.Node, processFunc processWordFunc, msg *bytes.Buffer) ([]string, error) {
	args := []string{}
	for i := 0; ast.Next != nil; i++ {
		ast = ast.Next
		words, err := processFunc(ast.Value)
		if err != nil {
			return nil, err
		}
		args = append(args, words...)
		msg.WriteString(" " + ast.Value)
	}
	return args, nil
}

type processWordFunc func(string) ([]string, error)

func createProcessWordFunc(shlex *ShellLex, cmd string, envs []string) processWordFunc {
	switch {
	case !replaceEnvAllowed[cmd]:
		return func(word string) ([]string, error) {
			return []string{word}, nil
		}
	case allowWordExpansion[cmd]:
		return func(word string) ([]string, error) {
			return shlex.ProcessWords(word, envs)
		}
	default:
		return func(word string) ([]string, error) {
			word, err := shlex.ProcessWord(word, envs)
			return []string{word}, err
		}
	}
}

// checkDispatch does a simple check for syntax errors of the Dockerfile.
// Because some of the instructions can only be validated through runtime,
// arg, env, etc., this syntax check will not be complete and could not replace
// the runtime check. Instead, this function is only a helper that allows
// user to find out the obvious error in Dockerfile earlier on.
func checkDispatch(ast *parser.Node) error {
	cmd := ast.Value
	upperCasedCmd := strings.ToUpper(cmd)

	// To ensure the user is given a decent error message if the platform
	// on which the daemon is running does not support a builder command.
	if err := platformSupports(strings.ToLower(cmd)); err != nil {
		return err
	}

	// The instruction itself is ONBUILD, we will make sure it follows with at
	// least one argument
	if upperCasedCmd == "ONBUILD" {
		if ast.Next == nil {
			buildsFailed.WithValues(metricsMissingOnbuildArgumentsError).Inc()
			return errors.New("ONBUILD requires at least one argument")
		}
	}

	if _, ok := evaluateTable[cmd]; ok {
		return nil
	}
	buildsFailed.WithValues(metricsUnknownInstructionError).Inc()
	return errors.Errorf("unknown instruction: %s", upperCasedCmd)
}
