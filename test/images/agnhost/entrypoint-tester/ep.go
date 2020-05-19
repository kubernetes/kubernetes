/*
Copyright 2015 The Kubernetes Authors.

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

package entrypoint

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// CmdEntrypointTester is used by agnhost Cobra.
var CmdEntrypointTester = &cobra.Command{
	Use:   "entrypoint-tester",
	Short: "Prints the args it's passed and exits",
	Long:  "Prints the args it's passed and exits.",
	Run:   main,
}

// This program prints all the executable's arguments and exits.
func main(cmd *cobra.Command, args []string) {
	// Some of the entrypoint-tester related tests overrides agnhost's default entrypoint
	// with agnhost-2, and this function's args will only contain the subcommand's
	// args (./agnhost entrypoint-tester these args), but we need to print *all* the
	// args, which is why os.Args should be printed instead.
	fmt.Printf("%v\n", os.Args)
	os.Exit(0)
}
