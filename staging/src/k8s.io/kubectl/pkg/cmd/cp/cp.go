/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cp

import (
	"archive/tar"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubectl/pkg/cmd/exec"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	cpExample = templates.Examples(i18n.T(`
		# !!!Important Note!!!
		# Requires that the 'tar' binary is present in your container
		# image.  If 'tar' is not present, 'kubectl cp' will fail.
		#
		# For advanced use cases, such as symlinks, wildcard expansion or
		# file mode preservation, consider using 'kubectl exec'.

		# Copy /tmp/foo local file to /tmp/bar in a remote pod in namespace <some-namespace>
		tar cf - /tmp/foo | kubectl exec -i -n <some-namespace> <some-pod> -- tar xf - -C /tmp/bar

		# Copy /tmp/foo from a remote pod to /tmp/bar locally
		kubectl exec -n <some-namespace> <some-pod> -- tar cf - /tmp/foo | tar xf - -C /tmp/bar

		# Copy /tmp/foo_dir local directory to /tmp/bar_dir in a remote pod in the default namespace
		kubectl cp /tmp/foo_dir <some-pod>:/tmp/bar_dir

		# Copy /tmp/foo local file to /tmp/bar in a remote pod in a specific container
		kubectl cp /tmp/foo <some-pod>:/tmp/bar -c <specific-container>

		# Copy /tmp/foo local file to /tmp/bar in a remote pod in namespace <some-namespace>
		kubectl cp /tmp/foo <some-namespace>/<some-pod>:/tmp/bar

		# Copy /tmp/foo from a remote pod to /tmp/bar locally
		kubectl cp <some-namespace>/<some-pod>:/tmp/foo /tmp/bar`))
)

// CopyOptions have the data required to perform the copy operation
type CopyOptions struct {
	Container  string
	Namespace  string
	NoPreserve bool
	MaxTries   int

	ClientConfig      *restclient.Config
	Clientset         kubernetes.Interface
	ExecParentCmdName string
	Executor          exec.RemoteExecutor

	args []string

	genericiooptions.IOStreams
}

// NewCopyOptions creates the options for copy
func NewCopyOptions(ioStreams genericiooptions.IOStreams) *CopyOptions {
	return &CopyOptions{
		IOStreams: ioStreams,
	}
}

// NewCmdCp creates a new Copy command.
func NewCmdCp(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewCopyOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "cp <file-spec-src> <file-spec-dest>",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Copy files and directories to and from containers"),
		Long:                  i18n.T("Copy files and directories to and from containers."),
		Example:               cpExample,
		ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			var comps []string
			if len(args) == 0 {
				if strings.IndexAny(toComplete, "/.~") == 0 {
					// Looks like a path, do nothing
				} else if strings.Contains(toComplete, ":") {
					// TODO: complete remote files in the pod
				} else if idx := strings.Index(toComplete, "/"); idx > 0 {
					// complete <namespace>/<pod>
					namespace := toComplete[:idx]
					template := "{{ range .items }}{{ .metadata.namespace }}/{{ .metadata.name }}: {{ end }}"
					comps = completion.CompGetFromTemplate(&template, f, namespace, []string{"pod"}, toComplete)
				} else {
					// Complete namespaces followed by a /
					for _, ns := range completion.CompGetResource(f, "namespace", toComplete) {
						comps = append(comps, fmt.Sprintf("%s/", ns))
					}
					// Complete pod names followed by a :
					for _, pod := range completion.CompGetResource(f, "pod", toComplete) {
						comps = append(comps, fmt.Sprintf("%s:", pod))
					}

					// Finally, provide file completion if we need to.
					// We only do this if:
					// 1- There are other completions found (if there are no completions,
					//    the shell will do file completion itself)
					// 2- If there is some input from the user (or else we will end up
					//    listing the entire content of the current directory which could
					//    be too many choices for the user)
					if len(comps) > 0 && len(toComplete) > 0 {
						if files, err := os.ReadDir("."); err == nil {
							for _, file := range files {
								filename := file.Name()
								if strings.HasPrefix(filename, toComplete) {
									if file.IsDir() {
										filename = fmt.Sprintf("%s/", filename)
									}
									// We are completing a file prefix
									comps = append(comps, filename)
								}
							}
						}
					} else if len(toComplete) == 0 {
						// If the user didn't provide any input to complete,
						// we provide a hint that a path can also be used
						comps = append(comps, "./", "/")
					}
				}
			}
			return comps, cobra.ShellCompDirectiveNoSpace
		},
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunWithContext(cmd.Context()))
		},
	}
	cmdutil.AddContainerVarFlags(cmd, &o.Container, o.Container)
	cmd.Flags().BoolVarP(&o.NoPreserve, "no-preserve", "", false, "The copied file/directory's ownership and permissions will not be preserved in the container")
	cmd.Flags().IntVarP(&o.MaxTries, "retries", "", 0, "Set number of retries to complete a copy operation from a container. Specify 0 to disable or any negative value for infinite retrying. The default is 0 (no retry).")

	return cmd
}

var (
	errFileSpecDoesntMatchFormat = errors.New("filespec must match the canonical format: [[namespace/]pod:]file/path")
)

func extractFileSpec(arg string) (fileSpec, error) {
	i := strings.Index(arg, ":")

	// filespec starting with a semicolon is invalid
	if i == 0 {
		return fileSpec{}, errFileSpecDoesntMatchFormat
	}
	if i == -1 {
		return fileSpec{
			File: newLocalPath(arg),
		}, nil
	}

	pod, file := arg[:i], arg[i+1:]
	pieces := strings.Split(pod, "/")
	switch len(pieces) {
	case 1:
		return fileSpec{
			PodName: pieces[0],
			File:    newRemotePath(file),
		}, nil
	case 2:
		return fileSpec{
			PodNamespace: pieces[0],
			PodName:      pieces[1],
			File:         newRemotePath(file),
		}, nil
	default:
		return fileSpec{}, errFileSpecDoesntMatchFormat
	}
}

// Complete completes all the required options
func (o *CopyOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if cmd.Parent() != nil {
		o.ExecParentCmdName = cmd.Parent().CommandPath()
	}
	o.Executor = &exec.DefaultRemoteExecutor{}

	var err error
	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.Clientset, err = f.KubernetesClientSet()
	if err != nil {
		return err
	}

	o.ClientConfig, err = f.ToRESTConfig()
	if err != nil {
		return err
	}

	o.args = args
	return nil
}

// Validate makes sure provided values for CopyOptions are valid
func (o *CopyOptions) Validate() error {
	if len(o.args) != 2 {
		return fmt.Errorf("source and destination are required")
	}
	return nil
}

// Run performs the execution.
func (o *CopyOptions) Run() error {
	return o.RunWithContext(context.Background())
}

// RunWithContext performs the execution using the command context.
func (o *CopyOptions) RunWithContext(ctx context.Context) error {
	srcSpec, err := extractFileSpec(o.args[0])
	if err != nil {
		return err
	}
	destSpec, err := extractFileSpec(o.args[1])
	if err != nil {
		return err
	}

	if len(srcSpec.PodName) != 0 && len(destSpec.PodName) != 0 {
		return fmt.Errorf("one of src or dest must be a local file specification")
	}
	if len(srcSpec.File.String()) == 0 || len(destSpec.File.String()) == 0 {
		return errors.New("filepath can not be empty")
	}

	if len(srcSpec.PodName) != 0 {
		return o.copyFromPod(ctx, srcSpec, destSpec)
	}
	if len(destSpec.PodName) != 0 {
		return o.copyToPod(ctx, srcSpec, destSpec, &exec.ExecOptions{})
	}
	return fmt.Errorf("one of src or dest must be a remote file specification")
}

// checkDestinationIsDir receives a destination fileSpec and
// determines if the provided destination path exists on the
// pod. If the destination path does not exist or is _not_ a
// directory, an error is returned with the exit code received.
func (o *CopyOptions) checkDestinationIsDir(ctx context.Context, dest fileSpec) error {
	options := &exec.ExecOptions{
		StreamOptions: exec.StreamOptions{
			IOStreams: genericiooptions.IOStreams{
				Out:    io.Discard,
				ErrOut: io.Discard,
			},

			Namespace: dest.PodNamespace,
			PodName:   dest.PodName,
		},

		Command:  []string{"test", "-d", dest.File.String()},
		Executor: o.Executor,
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	done := make(chan error)

	go func() {
		done <- o.execute(ctx, options)
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

func (o *CopyOptions) copyToPod(ctx context.Context, src, dest fileSpec, options *exec.ExecOptions) error {
	if _, err := os.Stat(src.File.String()); err != nil {
		return fmt.Errorf("%s doesn't exist in local filesystem", src.File)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	reader, writer := io.Pipe()

	srcFile := src.File.(localPath)
	destFile := dest.File.(remotePath)

	if err := o.checkDestinationIsDir(ctx, dest); err == nil {
		// If no error, dest.File was found to be a directory.
		// Copy specified src into it
		destFile = destFile.Join(srcFile.Base())
	} else if errors.Is(err, context.DeadlineExceeded) {
		// we haven't decided destination is directory or not because context timeout is exceeded.
		// That's why, we should shortcut the process in here.
		return err
	}

	var cmdArr []string

	if o.NoPreserve {
		cmdArr = []string{"tar", "--no-same-permissions", "--no-same-owner", "-xmf", "-"}
	} else {
		cmdArr = []string{"tar", "-xmf", "-"}
	}
	destFileDir := destFile.Dir().String()
	if len(destFileDir) > 0 {
		cmdArr = append(cmdArr, "-C", destFileDir)
	}

	options.StreamOptions = exec.StreamOptions{
		IOStreams: genericiooptions.IOStreams{
			In:     &contextReader{ctx: ctx, r: reader},
			Out:    &contextWriter{ctx: ctx, w: o.Out},
			ErrOut: &contextWriter{ctx: ctx, w: o.ErrOut},
		},
		Stdin: true,

		Namespace: dest.PodNamespace,
		PodName:   dest.PodName,
	}

	options.Command = cmdArr
	options.Executor = o.Executor

	execDone := make(chan error, 1)
	go func() {
		execDone <- o.execute(ctx, options)
	}()

	tarDone := make(chan error, 1)
	go func() {
		tarDone <- makeTar(srcFile, destFile, writer)
	}()

	var execErr, tarErr error
	select {
	case execErr = <-execDone:
		// Exec finished first (remote tar exited or failed); stop local tar.
		_ = writer.Close()
		_ = reader.Close()
		select {
		case tarErr = <-tarDone:
		case <-time.After(30 * time.Second):
		}
		if execErr != nil {
			return execErr
		}
		return wrapCopyWriteError(tarErr)

	case tarErr = <-tarDone:
		// Local tar finished; send EOF to remote tar and wait for it to exit.
		_ = writer.Close()
		if tarErr != nil {
			// Local failure: kill remote tar immediately instead of waiting for EOF.
			cancel()
			_ = reader.Close()
		}
		select {
		case execErr = <-execDone:
		case <-time.After(30 * time.Second):
			cancel()
			_ = reader.Close()
			return fmt.Errorf("timed out waiting for remote tar to complete")
		}
		if tarErr != nil {
			return wrapCopyWriteError(tarErr)
		}
		return execErr
	}
}

// contextReader stops supplying stdin to the remote tar when the copy is cancelled.
type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (cr *contextReader) Read(p []byte) (int, error) {
	if err := cr.ctx.Err(); err != nil {
		return 0, err
	}
	return cr.r.Read(p)
}

// contextWriter stops forwarding remote output when the copy is cancelled.
type contextWriter struct {
	ctx context.Context
	w   io.Writer
}

func (cw *contextWriter) Write(p []byte) (int, error) {
	if err := cw.ctx.Err(); err != nil {
		return 0, err
	}
	return cw.w.Write(p)
}

// podToLocalTarCommand starts tar as the exec session process (no shell wrapper).
// A shell pipeline leaves tar/cat as children of sh; when the stream closes, sh exits
// and those children are reparented to the container init and keep running.
func podToLocalTarCommand(path string, resumeFrom uint64) []string {
	if resumeFrom > 0 {
		escaped := strings.ReplaceAll(path, "'", `'\''`)
		script := fmt.Sprintf("trap 'kill 0' TERM; tar cf - '%s' | tail -c+%d", escaped, resumeFrom)
		return []string{"sh", "-c", script}
	}
	return []string{"tar", "cf", "-", path}
}

func podToLocalTarKillPattern(path string) string {
	return fmt.Sprintf("tar cf - %s", path)
}

// killPodToLocalTarProcess stops orphaned tar from a failed pod-to-local copy.
// Busybox tar may keep running after the exec stream closes; pkill is the exec
// process itself (no shell) so it does not accumulate sh/cat/pkill zombies.
func (o *CopyOptions) killPodToLocalTarProcess(src fileSpec, path string) {
	pattern := podToLocalTarKillPattern(path)
	for _, sig := range []string{"TERM", "KILL"} {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		options := &exec.ExecOptions{
			StreamOptions: exec.StreamOptions{
				IOStreams: genericiooptions.IOStreams{
					Out:    io.Discard,
					ErrOut: io.Discard,
				},
				Namespace: src.PodNamespace,
				PodName:   src.PodName,
			},
			Command:  []string{"pkill", "-" + sig, "-f", pattern},
			Executor: o.Executor,
		}
		_ = o.execute(ctx, options)
		cancel()
	}
}

func (o *CopyOptions) finishPodToLocalCopy(src fileSpec, path string, err error) error {
	if err != nil {
		o.killPodToLocalTarProcess(src, path)
	}
	return err
}

func (o *CopyOptions) copyFromPod(ctx context.Context, src, dest fileSpec) error {
	if o.MaxTries != 0 {
		return o.copyFromPodWithRetry(ctx, src, dest)
	}
	return o.copyFromPodOnce(ctx, src, dest)
}

// copyFromPodOnce copies from a pod without --retries. It mirrors copyToPod teardown:
// cancel the exec stream and wait for the remote tar to exit instead of only closing
// the local pipe (which can return "closed pipe" while tar keeps running in the pod).
func (o *CopyOptions) copyFromPodOnce(ctx context.Context, src, dest fileSpec) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	reader, writer := io.Pipe()

	srcFile := src.File.(remotePath)
	destFile := dest.File.(localPath)
	prefix := stripPathShortcuts(srcFile.StripSlashes().Clean().String())

	options := &exec.ExecOptions{
		StreamOptions: exec.StreamOptions{
			IOStreams: genericiooptions.IOStreams{
				In:     nil,
				Out:    &contextWriter{ctx: ctx, w: writer},
				ErrOut: &contextWriter{ctx: ctx, w: o.ErrOut},
			},
			Namespace: src.PodNamespace,
			PodName:   src.PodName,
		},
		Command:  podToLocalTarCommand(srcFile.String(), 0),
		Executor: o.Executor,
	}

	execDone := make(chan error, 1)
	go func() {
		execDone <- o.execute(ctx, options)
	}()

	untarDone := make(chan error, 1)
	go func() {
		untarDone <- o.untarAll(src.PodNamespace, src.PodName, prefix, srcFile, destFile, reader)
	}()

	teardownStreams := func() {
		cancel()
		_ = writer.Close()
		_ = reader.Close()
	}

	var execErr, untarErr error
	select {
	case execErr = <-execDone:
		select {
		case untarErr = <-untarDone:
		case <-time.After(30 * time.Second):
		}
		teardownStreams()
		if execErr != nil {
			return o.finishPodToLocalCopy(src, srcFile.String(), execErr)
		}
		return o.finishPodToLocalCopy(src, srcFile.String(), wrapCopyWriteError(untarErr))

	case untarErr = <-untarDone:
		teardownStreams()
		select {
		case execErr = <-execDone:
		case <-time.After(30 * time.Second):
			return o.finishPodToLocalCopy(src, srcFile.String(), fmt.Errorf("timed out waiting for remote tar to complete"))
		}
		if untarErr != nil {
			return o.finishPodToLocalCopy(src, srcFile.String(), wrapCopyWriteError(untarErr))
		}
		return o.finishPodToLocalCopy(src, srcFile.String(), execErr)
	}
}

func (o *CopyOptions) copyFromPodWithRetry(ctx context.Context, src, dest fileSpec) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	tp := newTarPipe(src, o, ctx)
	defer tp.Close()

	srcFile := src.File.(remotePath)
	destFile := dest.File.(localPath)
	prefix := stripPathShortcuts(srcFile.StripSlashes().Clean().String())
	err := o.untarAll(src.PodNamespace, src.PodName, prefix, srcFile, destFile, tp)
	if err != nil {
		cancel()
		tp.waitExec(30 * time.Second)
	}
	return o.finishPodToLocalCopy(src, srcFile.String(), wrapCopyWriteError(err))
}

type TarPipe struct {
	src       fileSpec
	o         *CopyOptions
	ctx       context.Context
	reader    *io.PipeReader
	outStream *io.PipeWriter
	bytesRead uint64
	retries   int

	mu            sync.Mutex
	sessionCancel context.CancelFunc
	execDone      chan error
	wg            sync.WaitGroup
	closed        bool
}

func newTarPipe(src fileSpec, o *CopyOptions, ctx context.Context) *TarPipe {
	t := &TarPipe{
		src: src,
		o:   o,
		ctx: ctx,
	}
	t.initReadFrom(0)
	return t
}

// Close stops the remote tar process and waits for the exec session to finish.
func (t *TarPipe) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	t.endSessionLocked()
	return nil
}

func (t *TarPipe) endSessionLocked() {
	if t.sessionCancel != nil {
		t.sessionCancel()
		t.sessionCancel = nil
	}
	// Close the write half first so a blocked remote-command copy unblocks when the
	// local untar stops reading (e.g. no space left on device on the host).
	if t.outStream != nil {
		_ = t.outStream.Close()
		t.outStream = nil
	}
	if t.reader != nil {
		_ = t.reader.Close()
		t.reader = nil
	}
	t.wg.Wait()
}

func (t *TarPipe) initReadFrom(n uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	t.endSessionLocked()

	t.reader, t.outStream = io.Pipe()
	sessionCtx, sessionCancel := context.WithCancel(t.ctx)
	t.sessionCancel = sessionCancel

	options := &exec.ExecOptions{
		StreamOptions: exec.StreamOptions{
			IOStreams: genericiooptions.IOStreams{
				In:     nil,
				Out:    &contextWriter{ctx: sessionCtx, w: t.outStream},
				ErrOut: &contextWriter{ctx: sessionCtx, w: t.o.ErrOut},
			},

			Namespace: t.src.PodNamespace,
			PodName:   t.src.PodName,
		},

		Command:  podToLocalTarCommand(t.src.File.String(), n),
		Executor: t.o.Executor,
	}

	writer := t.outStream
	t.execDone = make(chan error, 1)
	t.wg.Add(1)
	go func() {
		defer t.wg.Done()
		defer writer.Close()
		t.execDone <- t.o.execute(sessionCtx, options)
	}()
}

// waitExec blocks until the remote tar exec finishes or times out.
func (t *TarPipe) waitExec(timeout time.Duration) {
	t.mu.Lock()
	done := t.execDone
	t.mu.Unlock()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(timeout):
	}
}

func (t *TarPipe) Read(p []byte) (n int, err error) {
	n, err = t.reader.Read(p)
	if err != nil {
		if t.o.MaxTries < 0 || t.retries < t.o.MaxTries {
			t.retries++
			fmt.Fprintf(t.o.ErrOut, "Resuming copy at %d bytes, retry %d/%d\n", t.bytesRead, t.retries, t.o.MaxTries)
			t.initReadFrom(t.bytesRead + 1)
			err = nil
		} else {
			fmt.Fprintf(t.o.ErrOut, "Dropping out copy after %d retries\n", t.retries)
		}
	} else {
		t.bytesRead += uint64(n)
	}
	return
}

func makeTar(src localPath, dest remotePath, writer io.Writer) error {
	// TODO: use compression here?
	tarWriter := tar.NewWriter(writer)
	defer tarWriter.Close()

	srcPath := src.Clean()
	destPath := dest.Clean()
	return recursiveTar(srcPath.Dir(), srcPath.Base(), destPath.Dir(), destPath.Base(), tarWriter)
}

func recursiveTar(srcDir, srcFile localPath, destDir, destFile remotePath, tw *tar.Writer) error {
	matchedPaths, err := srcDir.Join(srcFile).Glob()
	if err != nil {
		return err
	}
	for _, fpath := range matchedPaths {
		stat, err := os.Lstat(fpath)
		if err != nil {
			return err
		}
		if stat.IsDir() {
			files, err := os.ReadDir(fpath)
			if err != nil {
				return err
			}
			if len(files) == 0 {
				//case empty directory
				hdr, _ := tar.FileInfoHeader(stat, fpath)
				hdr.Name = destFile.String()
				if err := tw.WriteHeader(hdr); err != nil {
					return err
				}
			}
			for _, f := range files {
				if err := recursiveTar(srcDir, srcFile.Join(newLocalPath(f.Name())),
					destDir, destFile.Join(newRemotePath(f.Name())), tw); err != nil {
					return err
				}
			}
			return nil
		} else if stat.Mode()&os.ModeSymlink != 0 {
			//case soft link
			hdr, _ := tar.FileInfoHeader(stat, fpath)
			target, err := os.Readlink(fpath)
			if err != nil {
				return err
			}

			hdr.Linkname = target
			hdr.Name = destFile.String()
			if err := tw.WriteHeader(hdr); err != nil {
				return err
			}
		} else {
			//case regular file or other file type like pipe
			hdr, err := tar.FileInfoHeader(stat, fpath)
			if err != nil {
				return err
			}
			hdr.Name = destFile.String()

			if err := tw.WriteHeader(hdr); err != nil {
				return err
			}

			f, err := os.Open(fpath)
			if err != nil {
				return err
			}
			defer f.Close()

			if _, err := io.Copy(tw, f); err != nil {
				return err
			}
			return f.Close()
		}
	}
	return nil
}

func (o *CopyOptions) untarAll(ns, pod string, prefix string, src remotePath, dest localPath, reader io.Reader) error {
	symlinkWarningPrinted := false
	// TODO: use compression here?
	tarReader := tar.NewReader(reader)
	for {
		header, err := tarReader.Next()
		if err != nil {
			if err != io.EOF {
				return err
			}
			break
		}

		// All the files will start with the prefix, which is the directory where
		// they were located on the pod, we need to strip down that prefix, but
		// if the prefix is missing it means the tar was tempered with.
		// For the case where prefix is empty we need to ensure that the path
		// is not absolute, which also indicates the tar file was tempered with.
		if !strings.HasPrefix(header.Name, prefix) {
			return fmt.Errorf("tar contents corrupted")
		}

		// basic file information
		mode := header.FileInfo().Mode()
		// header.Name is a name of the REMOTE file, so we need to create
		// a remotePath so that it goes through appropriate processing related
		// with cleaning remote paths
		destFileName := dest.Join(newRemotePath(header.Name[len(prefix):]))

		if !isRelative(dest, destFileName) {
			fmt.Fprintf(o.IOStreams.ErrOut, "warning: file %q is outside target destination, skipping\n", destFileName)
			continue
		}

		if err := os.MkdirAll(destFileName.Dir().String(), 0755); err != nil {
			return err
		}
		if header.FileInfo().IsDir() {
			if err := os.MkdirAll(destFileName.String(), 0755); err != nil {
				return err
			}
			continue
		}

		if mode&os.ModeSymlink != 0 {
			if !symlinkWarningPrinted && len(o.ExecParentCmdName) > 0 {
				fmt.Fprintf(o.IOStreams.ErrOut,
					"warning: skipping symlink: %q -> %q (consider using \"%s exec -n %q %q -- tar cf - %q | tar xf -\")\n",
					destFileName, header.Linkname, o.ExecParentCmdName, ns, pod, src)
				symlinkWarningPrinted = true
				continue
			}
			fmt.Fprintf(o.IOStreams.ErrOut, "warning: skipping symlink: %q -> %q\n", destFileName, header.Linkname)
			continue
		}
		outFile, err := os.Create(destFileName.String())
		if err != nil {
			return err
		}
		if _, err := io.Copy(outFile, tarReader); err != nil {
			outFile.Close()
			return err
		}
		if err := outFile.Close(); err != nil {
			return err
		}
	}

	return nil
}

func wrapCopyWriteError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, syscall.ENOSPC) {
		return fmt.Errorf("copy failed: no space left on device: %w", err)
	}
	var pathErr *os.PathError
	if errors.As(err, &pathErr) && errors.Is(pathErr.Err, syscall.ENOSPC) {
		return fmt.Errorf("copy failed: no space left on device while writing %q: %w", pathErr.Path, err)
	}
	return err
}

func (o *CopyOptions) execute(ctx context.Context, options *exec.ExecOptions) error {
	if len(options.Namespace) == 0 {
		options.Namespace = o.Namespace
	}

	if len(o.Container) > 0 {
		options.ContainerName = o.Container
	}

	options.Config = o.ClientConfig
	options.PodClient = o.Clientset.CoreV1()

	if err := options.Validate(); err != nil {
		return err
	}

	if options.Executor == nil {
		options.Executor = o.Executor
	}

	return options.RunWithContext(ctx)
}
