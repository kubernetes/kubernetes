/*
Copyright 2017 The Kubernetes Authors.

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

package nonewprivs

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// CmdNoneNewPrivs is used by agnhost Cobra.
var CmdNoNewPrivs = &cobra.Command{
	Use:   "nonewprivs",
	Short: "Prints the UID of the running process",
	Long:  `A go app that prints the UID of the process running to test security context features`,
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

func main(cmd *cobra.Command, args []string) {
	fmt.Printf("Effective uid: %d\n", os.Geteuid())
}
