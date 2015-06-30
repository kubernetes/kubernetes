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

// We use the github.com/spf13/cobra builtin markdown generation capabilities
// to generate markdown docs for the kube* daemons and kubectl here. Because
// of the nasty use of flags bound to the global FlagSet, flag.CommandLine,
// we can't actually do this in one binary because certain flags will cross
// pollinate. Instead we compile and run a gencobradoc main per command with
// 'go run'. This is slower and messier but necessary as long as we use/import
// packages that bind flags to flag.CommandLine. To add a cobra command to gendocs,
// make a cmd.go file in a subdirectory of ${KUBE_ROOT}/cmd/gendocs/cobra that
// has a method 'func Cmd() *cobra.Command' then add the name of the subdirectory
// to the commands var in this file.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

var commands = []string{"kubeapiserver", "kubecontrollermanager", "kubectl", "kubelet", "kubeproxy", "kubescheduler"}

const (
	generatedFileName = "cmd_generated.go"
	cobraGenFolder    = "cmd/gendocs/cobra"
	cobraGenMain      = "gencobradocs.go"
)

func main() {
	path := "docs/"
	kubeRoot := os.Getenv("KUBE_ROOT")
	if len(os.Args) == 2 {
		path = os.Args[1]
	} else if len(os.Args) == 3 {
		path = os.Args[1]
		kubeRoot = os.Args[2]
	} else if len(os.Args) > 3 {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [kube-root directory]\n", os.Args[0])
		os.Exit(1)
	}

	kubeRoot, err := filepath.Abs(kubeRoot)
	if err != nil {
		fmt.Printf("error: %v", err)
		os.Exit(1)
	}

	for _, command := range commands {
		err := genCobraDocs(command, path, kubeRoot)
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
	}
}

const template = `
package main

import (
        "github.com/spf13/cobra"

        cmd "github.com/GoogleCloudPlatform/kubernetes/%s/%s"
)

func getCmd() *cobra.Command {
        return cmd.Cmd()
}
`

func genCobraDocs(command, outputPath, kubeRoot string) error {
	f, err := os.Create(filepath.Join(kubeRoot, cobraGenFolder, generatedFileName))
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	fmt.Fprintf(f, template, cobraGenFolder, command)
	f.Close()

	b, err := exec.Command(
		"go",
		"run",
		filepath.Join(kubeRoot, cobraGenFolder, cobraGenMain),
		filepath.Join(kubeRoot, cobraGenFolder, generatedFileName),
		outputPath,
	).CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed with err %q and output: %s", err, string(b))
	}
	return nil
}
