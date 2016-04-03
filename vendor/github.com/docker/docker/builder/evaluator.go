// Package builder is the evaluation step in the Dockerfile parse/evaluate pipeline.
//
// It incorporates a dispatch table based on the parser.Node values (see the
// parser package for more information) that are yielded from the parser itself.
// Calling NewBuilder with the BuildOpts struct can be used to customize the
// experience for execution purposes only. Parsing is controlled in the parser
// package, and this division of resposibility should be respected.
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
package builder

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api"
	"github.com/docker/docker/builder/command"
	"github.com/docker/docker/builder/parser"
	"github.com/docker/docker/cliconfig"
	"github.com/docker/docker/daemon"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/pkg/tarsum"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/utils"
)

// Environment variable interpolation will happen on these statements only.
var replaceEnvAllowed = map[string]struct{}{
	command.Env:     {},
	command.Label:   {},
	command.Add:     {},
	command.Copy:    {},
	command.Workdir: {},
	command.Expose:  {},
	command.Volume:  {},
	command.User:    {},
}

var evaluateTable map[string]func(*Builder, []string, map[string]bool, string) error

func init() {
	evaluateTable = map[string]func(*Builder, []string, map[string]bool, string) error{
		command.Env:        env,
		command.Label:      label,
		command.Maintainer: maintainer,
		command.Add:        add,
		command.Copy:       dispatchCopy, // copy() is a go builtin
		command.From:       from,
		command.Onbuild:    onbuild,
		command.Workdir:    workdir,
		command.Run:        run,
		command.Cmd:        cmd,
		command.Entrypoint: entrypoint,
		command.Expose:     expose,
		command.Volume:     volume,
		command.User:       user,
	}
}

// internal struct, used to maintain configuration of the Dockerfile's
// processing as it evaluates the parsing result.
type Builder struct {
	Daemon *daemon.Daemon

	// effectively stdio for the run. Because it is not stdio, I said
	// "Effectively". Do not use stdio anywhere in this package for any reason.
	OutStream io.Writer
	ErrStream io.Writer

	Verbose      bool
	UtilizeCache bool
	cacheBusted  bool

	// controls how images and containers are handled between steps.
	Remove      bool
	ForceRemove bool
	Pull        bool

	// set this to true if we want the builder to not commit between steps.
	// This is useful when we only want to use the evaluator table to generate
	// the final configs of the Dockerfile but dont want the layers
	disableCommit bool

	// Registry server auth configs used to pull images when handling `FROM`.
	AuthConfigs map[string]cliconfig.AuthConfig

	// Deprecated, original writer used for ImagePull. To be removed.
	OutOld          io.Writer
	StreamFormatter *streamformatter.StreamFormatter

	Config *runconfig.Config // runconfig for cmd, run, entrypoint etc.

	// both of these are controlled by the Remove and ForceRemove options in BuildOpts
	TmpContainers map[string]struct{} // a map of containers used for removes

	dockerfileName string        // name of Dockerfile
	dockerfile     *parser.Node  // the syntax tree of the dockerfile
	image          string        // image name for commit processing
	maintainer     string        // maintainer name. could probably be removed.
	cmdSet         bool          // indicates is CMD was set in current Dockerfile
	BuilderFlags   *BuilderFlags // current cmd's BuilderFlags - temporary
	context        tarsum.TarSum // the context is a tarball that is uploaded by the client
	contextPath    string        // the path of the temporary directory the local context is unpacked to (server side)
	noBaseImage    bool          // indicates that this build does not start from any base image, but is being built from an empty file system.

	// Set resource restrictions for build containers
	cpuSetCpus   string
	cpuSetMems   string
	cpuShares    int64
	cpuPeriod    int64
	cpuQuota     int64
	cgroupParent string
	memory       int64
	memorySwap   int64

	cancelled <-chan struct{} // When closed, job was cancelled.

	activeImages []string
	id           string // Used to hold reference images
}

// Run the builder with the context. This is the lynchpin of this package. This
// will (barring errors):
//
// * call readContext() which will set up the temporary directory and unpack
//   the context into it.
// * read the dockerfile
// * parse the dockerfile
// * walk the parse tree and execute it by dispatching to handlers. If Remove
//   or ForceRemove is set, additional cleanup around containers happens after
//   processing.
// * Print a happy message and return the image ID.
//
func (b *Builder) Run(context io.Reader) (string, error) {
	if err := b.readContext(context); err != nil {
		return "", err
	}

	defer func() {
		if err := os.RemoveAll(b.contextPath); err != nil {
			logrus.Debugf("[BUILDER] failed to remove temporary context: %s", err)
		}
	}()

	if err := b.readDockerfile(); err != nil {
		return "", err
	}

	// some initializations that would not have been supplied by the caller.
	b.Config = &runconfig.Config{}

	b.TmpContainers = map[string]struct{}{}

	for i, n := range b.dockerfile.Children {
		select {
		case <-b.cancelled:
			logrus.Debug("Builder: build cancelled!")
			fmt.Fprintf(b.OutStream, "Build cancelled")
			return "", fmt.Errorf("Build cancelled")
		default:
			// Not cancelled yet, keep going...
		}
		if err := b.dispatch(i, n); err != nil {
			if b.ForceRemove {
				b.clearTmp()
			}
			return "", err
		}
		fmt.Fprintf(b.OutStream, " ---> %s\n", stringid.TruncateID(b.image))
		if b.Remove {
			b.clearTmp()
		}
	}

	if b.image == "" {
		return "", fmt.Errorf("No image was generated. Is your Dockerfile empty?")
	}

	fmt.Fprintf(b.OutStream, "Successfully built %s\n", stringid.TruncateID(b.image))
	return b.image, nil
}

// Reads a Dockerfile from the current context. It assumes that the
// 'filename' is a relative path from the root of the context
func (b *Builder) readDockerfile() error {
	// If no -f was specified then look for 'Dockerfile'. If we can't find
	// that then look for 'dockerfile'.  If neither are found then default
	// back to 'Dockerfile' and use that in the error message.
	if b.dockerfileName == "" {
		b.dockerfileName = api.DefaultDockerfileName
		tmpFN := filepath.Join(b.contextPath, api.DefaultDockerfileName)
		if _, err := os.Lstat(tmpFN); err != nil {
			tmpFN = filepath.Join(b.contextPath, strings.ToLower(api.DefaultDockerfileName))
			if _, err := os.Lstat(tmpFN); err == nil {
				b.dockerfileName = strings.ToLower(api.DefaultDockerfileName)
			}
		}
	}

	origFile := b.dockerfileName

	filename, err := symlink.FollowSymlinkInScope(filepath.Join(b.contextPath, origFile), b.contextPath)
	if err != nil {
		return fmt.Errorf("The Dockerfile (%s) must be within the build context", origFile)
	}

	fi, err := os.Lstat(filename)
	if os.IsNotExist(err) {
		return fmt.Errorf("Cannot locate specified Dockerfile: %s", origFile)
	}
	if fi.Size() == 0 {
		return fmt.Errorf("The Dockerfile (%s) cannot be empty", origFile)
	}

	f, err := os.Open(filename)
	if err != nil {
		return err
	}

	b.dockerfile, err = parser.Parse(f)
	f.Close()

	if err != nil {
		return err
	}

	// After the Dockerfile has been parsed, we need to check the .dockerignore
	// file for either "Dockerfile" or ".dockerignore", and if either are
	// present then erase them from the build context. These files should never
	// have been sent from the client but we did send them to make sure that
	// we had the Dockerfile to actually parse, and then we also need the
	// .dockerignore file to know whether either file should be removed.
	// Note that this assumes the Dockerfile has been read into memory and
	// is now safe to be removed.

	excludes, _ := utils.ReadDockerIgnore(filepath.Join(b.contextPath, ".dockerignore"))
	if rm, _ := fileutils.Matches(".dockerignore", excludes); rm == true {
		os.Remove(filepath.Join(b.contextPath, ".dockerignore"))
		b.context.(tarsum.BuilderContext).Remove(".dockerignore")
	}
	if rm, _ := fileutils.Matches(b.dockerfileName, excludes); rm == true {
		os.Remove(filepath.Join(b.contextPath, b.dockerfileName))
		b.context.(tarsum.BuilderContext).Remove(b.dockerfileName)
	}

	return nil
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
// "onbuild" and the child entrypoint is the command of the ONBUILD statmeent,
// such as `RUN` in ONBUILD RUN foo. There is special case logic in here to
// deal with that, at least until it becomes more of a general concern with new
// features.
func (b *Builder) dispatch(stepN int, ast *parser.Node) error {
	cmd := ast.Value
	attrs := ast.Attributes
	original := ast.Original
	flags := ast.Flags
	strs := []string{}
	msg := fmt.Sprintf("Step %d : %s", stepN, strings.ToUpper(cmd))

	if len(ast.Flags) > 0 {
		msg += " " + strings.Join(ast.Flags, " ")
	}

	if cmd == "onbuild" {
		if ast.Next == nil {
			return fmt.Errorf("ONBUILD requires at least one argument")
		}
		ast = ast.Next.Children[0]
		strs = append(strs, ast.Value)
		msg += " " + ast.Value

		if len(ast.Flags) > 0 {
			msg += " " + strings.Join(ast.Flags, " ")
		}

	}

	// count the number of nodes that we are going to traverse first
	// so we can pre-create the argument and message array. This speeds up the
	// allocation of those list a lot when they have a lot of arguments
	cursor := ast
	var n int
	for cursor.Next != nil {
		cursor = cursor.Next
		n++
	}
	l := len(strs)
	strList := make([]string, n+l)
	copy(strList, strs)
	msgList := make([]string, n)

	var i int
	for ast.Next != nil {
		ast = ast.Next
		var str string
		str = ast.Value
		if _, ok := replaceEnvAllowed[cmd]; ok {
			var err error
			str, err = ProcessWord(ast.Value, b.Config.Env)
			if err != nil {
				return err
			}
		}
		strList[i+l] = str
		msgList[i] = ast.Value
		i++
	}

	msg += " " + strings.Join(msgList, " ")
	fmt.Fprintln(b.OutStream, msg)

	// XXX yes, we skip any cmds that are not valid; the parser should have
	// picked these out already.
	if f, ok := evaluateTable[cmd]; ok {
		b.BuilderFlags = NewBuilderFlags()
		b.BuilderFlags.Args = flags
		return f(b, strList, attrs, original)
	}

	return fmt.Errorf("Unknown instruction: %s", strings.ToUpper(cmd))
}
