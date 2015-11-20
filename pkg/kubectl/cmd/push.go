/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	push_long = `Gather local files and push them up to storage.`

	push_example = `# push a directory up to cloud storage in the bucket my-files/archive.tgz
$ kubectl push --cloud-provider=gce ./some/directory  my-files

# push a directory up to cloud storage in the bucket my-files/some-other-name.tgz
$ kubectl push --cloud-provider=gce ./some/directory  my-files/some-other-name.tgz

# push a file up to cloud storage in the bucket my-files/file.html
$ kubectl push --cloud-provider=gce ./some/file.html my-files

# push a file up to cloud storage and rename the file
$ kubectl push --cloud-provider=gce ./some/file.html my-files/other-file.html
`
)

func NewCmdPush(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "push <directory or file> <bucket>[/<file>]",
		Short:   "Push a fileset up to storage and image push",
		Long:    push_long,
		Example: push_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(ValidatePushArgs(cmd, args))
			cmdutil.CheckErr(RunPush(f, cmd, args, out))
		},
	}
	cmdutil.AddCloudProviderFlags(cmd)
	return cmd
}

func ValidatePushArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("expected <directory or file> <bucket[/file]>")
	}
	return nil
}

func extractBucketAndFile(spec string) (bucket, file string) {
	if strings.Index(spec, "/") != -1 {
		parts := strings.SplitN(spec, "/", 2)
		return parts[0], parts[1]
	}
	return spec, ""
}

func RunPush(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	path := args[0]
	bucket, file := extractBucketAndFile(args[1])

	err := cmdutil.UploadDirectoryOrFile(cmd, path, bucket, file)
	if err != nil {
		return err
	}
	out.Write([]byte(fmt.Sprintf("%s/%s\n", bucket, file)))

	return nil
}
