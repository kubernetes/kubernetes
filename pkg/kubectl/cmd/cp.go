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

package cmd

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
)

var (
	cpExample = templates.Examples(i18n.T(`
		# !!!Important Note!!!
		# Requires that the 'tar' binary is present in your container
		# image.  If 'tar' is not present, 'kubectl cp' will fail.

		# Copy /tmp/foo_dir local directory to /tmp/bar_dir in a remote pod in the default namespace
		kubectl cp /tmp/foo_dir <some-pod>:/tmp/bar_dir

		# Copy /tmp/foo local file to /tmp/bar in a remote pod in a specific container
		kubectl cp /tmp/foo <some-pod>:/tmp/bar -c <specific-container>

		# Copy /tmp/foo local file to /tmp/bar in a remote pod in namespace <some-namespace>
		kubectl cp /tmp/foo <some-namespace>/<some-pod>:/tmp/bar

		# Copy /tmp/foo from a remote pod to /tmp/bar locally
		kubectl cp <some-namespace>/<some-pod>:/tmp/foo /tmp/bar`))

	cpUsageStr = dedent.Dedent(`
		expected 'cp <file-spec-src> <file-spec-dest> [-c container]'.
		<file-spec> is:
		[namespace/]pod-name:/file/path for a remote file
		/file/path for a local file`)
)

// NewCmdCp creates a new Copy command.
func NewCmdCp(f cmdutil.Factory, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "cp <file-spec-src> <file-spec-dest>",
		Short:   i18n.T("Copy files and directories to and from containers."),
		Long:    "Copy files and directories to and from containers.",
		Example: cpExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(runCopy(f, cmd, cmdOut, cmdErr, args))
		},
	}
	cmd.Flags().StringP("container", "c", "", "Container name. If omitted, the first container in the pod will be chosen")

	return cmd
}

type fileSpec struct {
	PodNamespace string
	PodName      string
	File         string
}

var (
	errFileSpecDoesntMatchFormat = errors.New("Filespec must match the canonical format: [[namespace/]pod:]file/path")
	errFileCannotBeEmpty         = errors.New("Filepath can not be empty")
)

func extractFileSpec(arg string) (fileSpec, error) {
	pieces := strings.Split(arg, ":")
	if len(pieces) == 1 {
		return fileSpec{File: arg}, nil
	}
	if len(pieces) != 2 {
		// FIXME Kubernetes can't copy files that contain a ':'
		// character.
		return fileSpec{}, errFileSpecDoesntMatchFormat
	}
	file := pieces[1]

	pieces = strings.Split(pieces[0], "/")
	if len(pieces) == 1 {
		return fileSpec{
			PodName: pieces[0],
			File:    file,
		}, nil
	}
	if len(pieces) == 2 {
		return fileSpec{
			PodNamespace: pieces[0],
			PodName:      pieces[1],
			File:         file,
		}, nil
	}

	return fileSpec{}, errFileSpecDoesntMatchFormat
}

func runCopy(f cmdutil.Factory, cmd *cobra.Command, out, cmderr io.Writer, args []string) error {
	if len(args) != 2 {
		return cmdutil.UsageErrorf(cmd, cpUsageStr)
	}
	srcSpec, err := extractFileSpec(args[0])
	if err != nil {
		return err
	}
	destSpec, err := extractFileSpec(args[1])
	if err != nil {
		return err
	}
	if len(srcSpec.PodName) != 0 {
		return copyFromPod(f, cmd, cmderr, srcSpec, destSpec)
	}
	if len(destSpec.PodName) != 0 {
		return copyToPod(f, cmd, out, cmderr, srcSpec, destSpec)
	}
	return cmdutil.UsageErrorf(cmd, "One of src or dest must be a remote file specification")
}

// checkDestinationIsDir receives a destination fileSpec and
// determines if the provided destination path exists on the
// pod. If the destination path does not exist or is _not_ a
// directory, an error is returned with the exit code received.
func checkDestinationIsDir(dest fileSpec, f cmdutil.Factory, cmd *cobra.Command) error {
	options := &ExecOptions{
		StreamOptions: StreamOptions{
			Out: bytes.NewBuffer([]byte{}),
			Err: bytes.NewBuffer([]byte{}),

			Namespace: dest.PodNamespace,
			PodName:   dest.PodName,
		},

		Command:  []string{"test", "-d", dest.File},
		Executor: &DefaultRemoteExecutor{},
	}

	return execute(f, cmd, options)
}

func copyToPod(f cmdutil.Factory, cmd *cobra.Command, stdout, stderr io.Writer, src, dest fileSpec) error {
	if len(src.File) == 0 {
		return errFileCannotBeEmpty
	}
	reader, writer := io.Pipe()

	// strip trailing slash (if any)
	if strings.HasSuffix(string(dest.File[len(dest.File)-1]), "/") {
		dest.File = dest.File[:len(dest.File)-1]
	}

	if err := checkDestinationIsDir(dest, f, cmd); err == nil {
		// If no error, dest.File was found to be a directory.
		// Copy specified src into it
		dest.File = dest.File + "/" + path.Base(src.File)
	}

	go func() {
		defer writer.Close()
		err := makeTar(src.File, dest.File, writer)
		cmdutil.CheckErr(err)
	}()

	// TODO: Improve error messages by first testing if 'tar' is present in the container?
	cmdArr := []string{"tar", "xf", "-"}
	destDir := path.Dir(dest.File)
	if len(destDir) > 0 {
		cmdArr = append(cmdArr, "-C", destDir)
	}

	options := &ExecOptions{
		StreamOptions: StreamOptions{
			In:    reader,
			Out:   stdout,
			Err:   stderr,
			Stdin: true,

			Namespace: dest.PodNamespace,
			PodName:   dest.PodName,
		},

		Command:  cmdArr,
		Executor: &DefaultRemoteExecutor{},
	}
	return execute(f, cmd, options)
}

func copyFromPod(f cmdutil.Factory, cmd *cobra.Command, cmderr io.Writer, src, dest fileSpec) error {
	if len(src.File) == 0 {
		return errFileCannotBeEmpty
	}

	reader, outStream := io.Pipe()
	options := &ExecOptions{
		StreamOptions: StreamOptions{
			In:  nil,
			Out: outStream,
			Err: cmderr,

			Namespace: src.PodNamespace,
			PodName:   src.PodName,
		},

		// TODO: Improve error messages by first testing if 'tar' is present in the container?
		Command:  []string{"tar", "cf", "-", src.File},
		Executor: &DefaultRemoteExecutor{},
	}

	go func() {
		defer outStream.Close()
		execute(f, cmd, options)
	}()
	prefix := getPrefix(src.File)
	prefix = path.Clean(prefix)
	return untarAll(reader, dest.File, prefix)
}

func makeTar(srcPath, destPath string, writer io.Writer) error {
	// TODO: use compression here?
	tarWriter := tar.NewWriter(writer)
	defer tarWriter.Close()

	srcPath = path.Clean(srcPath)
	destPath = path.Clean(destPath)
	return recursiveTar(path.Dir(srcPath), path.Base(srcPath), path.Dir(destPath), path.Base(destPath), tarWriter)
}

func recursiveTar(srcBase, srcFile, destBase, destFile string, tw *tar.Writer) error {
	filepath := path.Join(srcBase, srcFile)
	stat, err := os.Lstat(filepath)
	if err != nil {
		return err
	}
	if stat.IsDir() {
		files, err := ioutil.ReadDir(filepath)
		if err != nil {
			return err
		}
		if len(files) == 0 {
			//case empty directory
			hdr, _ := tar.FileInfoHeader(stat, filepath)
			hdr.Name = destFile
			if err := tw.WriteHeader(hdr); err != nil {
				return err
			}
		}
		for _, f := range files {
			if err := recursiveTar(srcBase, path.Join(srcFile, f.Name()), destBase, path.Join(destFile, f.Name()), tw); err != nil {
				return err
			}
		}
		return nil
	} else if stat.Mode()&os.ModeSymlink != 0 {
		//case soft link
		hdr, _ := tar.FileInfoHeader(stat, filepath)
		target, err := os.Readlink(filepath)
		if err != nil {
			return err
		}

		hdr.Linkname = target
		hdr.Name = destFile
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
	} else {
		//case regular file or other file type like pipe
		hdr, err := tar.FileInfoHeader(stat, filepath)
		if err != nil {
			return err
		}
		hdr.Name = destFile

		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}

		f, err := os.Open(filepath)
		if err != nil {
			return err
		}
		defer f.Close()

		if _, err := io.Copy(tw, f); err != nil {
			return err
		}
		return f.Close()
	}
	return nil
}

// clean prevents path traversals by stripping them out.
// This is adapted from https://golang.org/src/net/http/fs.go#L74
func clean(fileName string) string {
	return path.Clean(string(os.PathSeparator) + fileName)
}

func untarAll(reader io.Reader, destFile, prefix string) error {
	entrySeq := -1

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
		entrySeq++
		mode := header.FileInfo().Mode()
		outFileName := path.Join(destFile, clean(header.Name[len(prefix):]))
		baseName := path.Dir(outFileName)
		if err := os.MkdirAll(baseName, 0755); err != nil {
			return err
		}
		if header.FileInfo().IsDir() {
			if err := os.MkdirAll(outFileName, 0755); err != nil {
				return err
			}
			continue
		}

		// handle coping remote file into local directory
		if entrySeq == 0 && !header.FileInfo().IsDir() {
			exists, err := dirExists(outFileName)
			if err != nil {
				return err
			}
			if exists {
				outFileName = filepath.Join(outFileName, path.Base(clean(header.Name)))
			}
		}

		if mode&os.ModeSymlink != 0 {
			err := os.Symlink(header.Linkname, outFileName)
			if err != nil {
				return err
			}
		} else {
			outFile, err := os.Create(outFileName)
			if err != nil {
				return err
			}
			defer outFile.Close()
			if _, err := io.Copy(outFile, tarReader); err != nil {
				return err
			}
			if err := outFile.Close(); err != nil {
				return err
			}
		}
	}

	if entrySeq == -1 {
		//if no file was copied
		errInfo := fmt.Sprintf("error: %s no such file or directory", prefix)
		return errors.New(errInfo)
	}
	return nil
}

func getPrefix(file string) string {
	if file[0] == '/' {
		// tar strips the leading '/' if it's there, so we will too
		return file[1:]
	}
	return file
}

func execute(f cmdutil.Factory, cmd *cobra.Command, options *ExecOptions) error {
	if len(options.Namespace) == 0 {
		namespace, _, err := f.DefaultNamespace()
		if err != nil {
			return err
		}
		options.Namespace = namespace
	}

	container := cmdutil.GetFlagString(cmd, "container")
	if len(container) > 0 {
		options.ContainerName = container
	}

	config, err := f.ClientConfig()
	if err != nil {
		return err
	}
	options.Config = config

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	options.PodClient = clientset.Core()

	if err := options.Validate(); err != nil {
		return err
	}

	if err := options.Run(); err != nil {
		return err
	}
	return nil
}

// dirExists checks if a path exists and is a directory.
func dirExists(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err == nil && fi.IsDir() {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
