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
	"bytes"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	RootCmd.AddCommand(initCmd)
}

// initialize Command
var initCmd = &cobra.Command{
	Use:     "init [name]",
	Aliases: []string{"initialize", "initialise", "create"},
	Short:   "Initialize a Cobra Application",
	Long: `Initialize (cobra init) will create a new application, with a license
and the appropriate structure for a Cobra-based CLI application.

  * If a name is provided, it will be created in the current directory;
  * If no name is provided, the current directory will be assumed;
  * If a relative path is provided, it will be created inside $GOPATH
    (e.g. github.com/spf13/hugo);
  * If an absolute path is provided, it will be created;
  * If the directory already exists but is empty, it will be used.

Init will not use an existing directory with contents.`,

	Run: func(cmd *cobra.Command, args []string) {
		switch len(args) {
		case 0:
			inputPath = ""

		case 1:
			inputPath = args[0]

		default:
			er("init doesn't support more than 1 parameter")
		}
		guessProjectPath()
		initializePath(projectPath)
	},
}

func initializePath(path string) {
	b, err := exists(path)
	if err != nil {
		er(err)
	}

	if !b { // If path doesn't yet exist, create it
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			er(err)
		}
	} else { // If path exists and is not empty don't use it
		empty, err := exists(path)
		if err != nil {
			er(err)
		}
		if !empty {
			er("Cobra will not create a new project in a non empty directory")
		}
	}
	// We have a directory and it's empty.. Time to initialize it.

	createLicenseFile()
	createMainFile()
	createRootCmdFile()
}

func createLicenseFile() {
	lic := getLicense()

	// Don't bother writing a LICENSE file if there is no text.
	if lic.Text != "" {
		data := make(map[string]interface{})

		// Try to remove the email address, if any
		data["copyright"] = strings.Split(copyrightLine(), " <")[0]

		data["appName"] = projectName()

		// Generate license template from text and data.
		r, _ := templateToReader(lic.Text, data)
		buf := new(bytes.Buffer)
		buf.ReadFrom(r)

		err := writeTemplateToFile(ProjectPath(), "LICENSE", buf.String(), data)
		_ = err
		// if err != nil {
		// 	er(err)
		// }
	}
}

func createMainFile() {
	lic := getLicense()

	template := `{{ comment .copyright }}
{{if .license}}{{ comment .license }}
{{end}}
package main

import "{{ .importpath }}"

func main() {
	cmd.Execute()
}
`
	data := make(map[string]interface{})

	data["copyright"] = copyrightLine()
	data["appName"] = projectName()

	// Generate license template from header and data.
	r, _ := templateToReader(lic.Header, data)
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	data["license"] = buf.String()

	data["importpath"] = guessImportPath() + "/" + guessCmdDir()

	err := writeTemplateToFile(ProjectPath(), "main.go", template, data)
	_ = err
	// if err != nil {
	// 	er(err)
	// }
}

func createRootCmdFile() {
	lic := getLicense()

	template := `{{ comment .copyright }}
{{if .license}}{{ comment .license }}
{{end}}
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
{{ if .viper }}	"github.com/spf13/viper"
{{ end }})
{{if .viper}}
var cfgFile string
{{ end }}
// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "{{ .appName }}",
	Short: "A brief description of your application",
	Long: ` + "`" + `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.` + "`" + `,
// Uncomment the following line if your bare application
// has an action associated with it:
//	Run: func(cmd *cobra.Command, args []string) { },
}

// Execute adds all child commands to the root command sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}

func init() {
{{ if .viper }}	cobra.OnInitialize(initConfig)

{{ end }}	// Here you will define your flags and configuration settings.
	// Cobra supports Persistent Flags, which, if defined here,
	// will be global for your application.
{{ if .viper }}
	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.{{ .appName }}.yaml)")
{{ else }}
	// RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.{{ .appName }}.yaml)")
{{ end }}	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	RootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
{{ if .viper }}
// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if cfgFile != "" { // enable ability to specify config file via flag
		viper.SetConfigFile(cfgFile)
	}

	viper.SetConfigName(".{{ .appName }}") // name of config file (without extension)
	viper.AddConfigPath("$HOME")  // adding home directory as first search path
	viper.AutomaticEnv()          // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}
{{ end }}`

	data := make(map[string]interface{})

	data["copyright"] = copyrightLine()
	data["appName"] = projectName()

	// Generate license template from header and data.
	r, _ := templateToReader(lic.Header, data)
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	data["license"] = buf.String()

	data["viper"] = viper.GetBool("useViper")

	err := writeTemplateToFile(ProjectPath()+string(os.PathSeparator)+guessCmdDir(), "root.go", template, data)
	if err != nil {
		er(err)
	}

	fmt.Println("Your Cobra application is ready at")
	fmt.Println(ProjectPath())
	fmt.Println("Give it a try by going there and running `go run main.go`")
	fmt.Println("Add commands to it by running `cobra add [cmdname]`")
}
