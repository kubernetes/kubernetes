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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
)

var (
	cp_example = templates.Examples(`
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
		kubectl cp <some-namespace>/<some-pod>:/tmp/foo /tmp/bar`)

	cpUsageStr = dedent.Dedent(`
	    expected 'cp <file-spec-src> <file-spec-dest> [-c container]'.
	    <file-spec> is:
		[namespace/]pod-name:/file/path for a remote file
		/file/path for a local file`)
)

// NewCmdCp creates a new Copy command.
func NewCmdCp(f cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "cp <file-spec-src> <file-spec-dest>",
		Short:   "Copy files and directories to and from containers.",
		Long:    "Copy files and directories to and from containers.",
		Example: cp_example,
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

func extractFileSpec(arg string) (fileSpec, error) {
	pieces := strings.Split(arg, ":")
	if len(pieces) == 1 {
		return fileSpec{File: arg}, nil
	}
	if len(pieces) != 2 {
		return fileSpec{}, fmt.Errorf("Unexpected fileSpec: %s, expected [[namespace/]pod:]file/path", arg)
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

	return fileSpec{}, fmt.Errorf("Unexpected file spec: %s, expected [[namespace/]pod:]file/path", arg)
}

func runCopy(f cmdutil.Factory, cmd *cobra.Command, out, cmderr io.Writer, args []string) error {
	if len(args) != 2 {
		return cmdutil.UsageError(cmd, cpUsageStr)
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
		return copyFromPod(f, cmd, out, cmderr, srcSpec, destSpec)
	}
	if len(destSpec.PodName) != 0 {
		return copyToPod(f, cmd, out, cmderr, srcSpec, destSpec)
	}
	return cmdutil.UsageError(cmd, "One of src or dest must be a remote file specification")
}

func copyToPod(f cmdutil.Factory, cmd *cobra.Command, stdout, stderr io.Writer, src, dest fileSpec) error {
	reader, writer := io.Pipe()
	go func() {
		defer writer.Close()
		err := makeTar(src.File, writer)
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

func copyFromPod(f cmdutil.Factory, cmd *cobra.Command, out, cmderr io.Writer, src, dest fileSpec) error {
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

	return untarAll(reader, dest.File, prefix)
}

func makeTar(filepath string, writer io.Writer) error {
	// TODO: use compression here?
	tarWriter := tar.NewWriter(writer)
	defer tarWriter.Close()
	return recursiveTar(path.Dir(filepath), path.Base(filepath), tarWriter)
}

func recursiveTar(base, file string, tw *tar.Writer) error {
	filepath := path.Join(base, file)
	stat, err := os.Stat(filepath)
	if err != nil {
		return err
	}
	if stat.IsDir() {
		files, err := ioutil.ReadDir(filepath)
		if err != nil {
			return err
		}
		for _, f := range files {
			if err := recursiveTar(base, path.Join(file, f.Name()), tw); err != nil {
				return err
			}
		}
		return nil
	}
	hdr, err := tar.FileInfoHeader(stat, filepath)
	if err != nil {
		return err
	}
	hdr.Name = file
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	f, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(tw, f)
	return err
}

func untarAll(reader io.Reader, destFile, prefix string) error {
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
		outFileName := path.Join(destFile, header.Name[len(prefix):])
		baseName := path.Dir(outFileName)
		if err := os.MkdirAll(baseName, 0755); err != nil {
			return err
		}
		if header.FileInfo().IsDir() {
			os.MkdirAll(outFileName, 0755)
			continue
		}
		outFile, err := os.Create(outFileName)
		if err != nil {
			return err
		}
		defer outFile.Close()
		io.Copy(outFile, tarReader)
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
