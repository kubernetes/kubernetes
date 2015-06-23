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

package main

import (
	"io"
	"io/ioutil"
	"os"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	kubectlcmd "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	kruntime "github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilyaml "github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
)

// TODO marekbiskup 2015-06-23: This is a temporary version, not meant for users.
// Many things are missing:
// * bash completion for files (json/yaml) - it partially exists but doesn't work
// * handling files with multiple resources.
// * reading a resource from stdin
// * reading entire directory, also via http (like kubectl create)
// * better error handling, better messages
// * (?) handling of object kind and version (there is a version in the file,
//   so we want to query it too.
// * other template languages/output formats (like kubectl get)
// * flags uhnified with kubectl get/create
// * takes several files in one command - maybe it should take just one

func tryDecodeSingleObj(data []byte) (object kruntime.Object, err error) {
	// JSON is valid YAML, so this should work for everything.
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	obj, err := api.Scheme.Decode(json)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

func getFileData(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return ioutil.ReadAll(file)
}

func extractFromFile(filename string) (object kruntime.Object, err error) {
	data, err := getFileData(filename)
	if err != nil {
		return object, err
	}

	return tryDecodeSingleObj(data)
}

func NewResourceQueryCommand(in io.Reader, out, err io.Writer) *cobra.Command {
	var filenames util.StringList
	var template *string
	cmd := &cobra.Command{
		Use:   "resourcequery",
		Short: "resourcequery is used to print a resource using a template.",
		Long: `Print the resource defined in a file using a template.

JSON and YAML input formats are accepted.
This is an early alpha version of this command.
Expect it to change or disappear.`,
		Example: `// Print the resource name from the file.
$ resourcequery -t {{.metadata.name}} -f pod.json`,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(kubectlcmd.ValidateArgs(cmd, args))

			p, err := kubectl.NewTemplatePrinter([]byte(*template))
			cmdutil.CheckErr(err)
			for _, filename := range filenames {
				obj, _ := extractFromFile(filename)
				cmdutil.CheckErr(err)
				_, objKind, err := api.Scheme.ObjectVersionAndKind(obj)
				cmdutil.CheckErr(err)
				// kind is erased when the object is parsed.
				// but we need to be able to query it
				// note that we don't set the version
				err = conversion.NewScheme().SetVersionAndKind("", objKind, obj)
				cmdutil.CheckErr(err)

				cmdutil.CheckErr(p.PrintObj(obj, out))
			}
			return
		},
	}

	usage := "Filename to format using the given template"
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	template = cmd.Flags().StringP("template", "t", "", "Template string. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]")
	cmd.MarkFlagRequired("filename")
	cmd.MarkFlagRequired("template")

	return cmd
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	cmd := NewResourceQueryCommand(os.Stdin, os.Stdout, os.Stderr)
	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
