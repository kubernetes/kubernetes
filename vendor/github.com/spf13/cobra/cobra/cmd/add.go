// Copyright © 2015 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	RootCmd.AddCommand(addCmd)
}

var pName string

// initialize Command
var addCmd = &cobra.Command{
	Use:     "add [command name]",
	Aliases: []string{"command"},
	Short:   "Add a command to a Cobra Application",
	Long: `Add (cobra add) will create a new command, with a license and
the appropriate structure for a Cobra-based CLI application,
and register it to its parent (default RootCmd).

If you want your command to be public, pass in the command name
with an initial uppercase letter.

Example: cobra add server  -> resulting in a new cmd/server.go
  `,

	Run: func(cmd *cobra.Command, args []string) {
		if len(args) != 1 {
			er("add needs a name for the command")
		}
		guessProjectPath()
		createCmdFile(args[0])
	},
}

func init() {
	addCmd.Flags().StringVarP(&pName, "parent", "p", "RootCmd", "name of parent command for this command")
}

func parentName() string {
	if !strings.HasSuffix(strings.ToLower(pName), "cmd") {
		return pName + "Cmd"
	}

	return pName
}

func createCmdFile(cmdName string) {
	lic := getLicense()

	template := `{{ comment .copyright }}
{{ comment .license }}

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// {{.cmdName}}Cmd represents the {{.cmdName}} command
var {{ .cmdName }}Cmd = &cobra.Command{
	Use:   "{{ .cmdName }}",
	Short: "A brief description of your command",
	Long: ` + "`" + `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.` + "`" + `,
	Run: func(cmd *cobra.Command, args []string) {
		// TODO: Work your own magic here
		fmt.Println("{{ .cmdName }} called")
	},
}

func init() {
	{{ .parentName }}.AddCommand({{ .cmdName }}Cmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// {{.cmdName}}Cmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// {{.cmdName}}Cmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")

}
`

	var data map[string]interface{}
	data = make(map[string]interface{})

	data["copyright"] = copyrightLine()
	data["license"] = lic.Header
	data["appName"] = projectName()
	data["viper"] = viper.GetBool("useViper")
	data["parentName"] = parentName()
	data["cmdName"] = cmdName

	err := writeTemplateToFile(filepath.Join(ProjectPath(), guessCmdDir()), cmdName+".go", template, data)
	if err != nil {
		er(err)
	}
	fmt.Println(cmdName, "created at", filepath.Join(ProjectPath(), guessCmdDir(), cmdName+".go"))
}
