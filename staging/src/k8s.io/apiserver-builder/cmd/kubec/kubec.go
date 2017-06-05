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

package main

// kc init --repo-name github.com/pwittrock/test

import (
	"fmt"
	"os"
	"runtime"

	"bufio"
	"bytes"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"io"
	"io/ioutil"
	"k8s.io/apimachinery/pkg/util/sets"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"text/template"
	"time"
)

var repoPath string
var repoName string
var copyFrom string
var domain string
var types []string
var skipOpenApi bool
var skipRegister bool
var skipConversion bool
var skipDeepCopy bool

func main() {
	if len(os.Getenv("GOMAXPROCS")) == 0 {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}

	cmd.Flags().StringVar(&repoPath, "repo-path", "/out", "path to repo")
	cmd.Flags().StringVar(&copyFrom, "from-path", "/go/src/github.com/pwittrock/apiserver-helloworld/", "path to repo to copy from")
	cmd.Flags().StringVar(&repoName, "repo-name", "", "full name of repo")
	cmd.AddCommand(initCmd, addTypesCmd, genCmd, genDocs)

	initCmd.Flags().StringVar(&domain, "domain", "k8s.io", "domain group lives in")
	initCmd.Flags().StringVar(&repoName, "repo-name", "", "full name of repo")

	addTypesCmd.Flags().StringSliceVar(&types, "types", []string{}, "list of group/version/kind")
	addTypesCmd.Flags().StringVar(&domain, "domain", "k8s.io", "domain group lives in")
	addTypesCmd.Flags().StringVar(&repoName, "repo-name", "", "full name of repo")

	genCmd.Flags().BoolVar(&skipRegister, "skip-register", false, "")
	genCmd.Flags().BoolVar(&skipDeepCopy, "skip-deepcopy", false, "")
	genCmd.Flags().BoolVar(&skipConversion, "skip-conversion", false, "")
	genCmd.Flags().BoolVar(&skipOpenApi, "skip-openapi", false, "")
	genCmd.Flags().StringVar(&repoName, "repo-name", "", "full name of repo")

	genDocs.Flags().BoolVar(&skipOpenApi, "skip-openapi", false, "If true, don't generate swagger.json")
	genDocs.Flags().StringVar(&repoName, "repo-name", "", "full name of repo")

	if err := cmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}

func GetRepoName() {
	return
}

func RunMain(cmd *cobra.Command, args []string) {
	cmd.Help()
}

var cmd = &cobra.Command{
	Use:   "kubec",
	Short: "kubec builds Kubernetes extensions",
	Long:  `kubec is a set of commands for building Kubernetes extensions`,
	Run:   RunMain,
}

var genDocs = &cobra.Command{
	Use:   "generate-docs",
	Short: "Create generated docs files",
	Long:  `Create generated docs files`,
	Run:   RunGenDocsCmd,
}

var bearerToken = regexp.MustCompile(`Local Authorization Token: (.+)$`)

func RunGenDocsCmd(cmd *cobra.Command, args []string) {
	dir := filepath.Join(repoPath, "src", repoName)

	if !skipOpenApi {
		config := fmt.Sprintf("%s/cmd/kubec/empty_config", copyFrom)
		cert := "/var/run/kubernetes/apiserver.crt"
		c := exec.Command("go", "run", filepath.Join("./main.go"),
			"--authentication-kubeconfig", config,
			"--authorization-kubeconfig", config,
			"--client-ca-file", cert,
			"--requestheader-client-ca-file", cert,
			"--requestheader-username-headers", "X-Remote-User",
			"--requestheader-group-headers", "X-Remote-Group",
			"--requestheader-extra-headers-prefix", "X-Remote-Extra-",
			"--etcd-servers", "http://localhost:2379",
			"--secure-port", "9443",
			"--tls-ca-file", cert,
			"--print-bearer-token")
		c.Env = append(c.Env, fmt.Sprintf("REPO=%s", repoName))
		c.Env = append(c.Env, fmt.Sprintf("GOPATH=%s", repoPath))
		c.Dir = dir

		token := make(chan string)
		go func() {
			// Obtain a pipe to receive the stdout of the command
			out, err := c.StdoutPipe()
			if err != nil {
				panic(err)
			}
			c.Stderr = c.Stdout
			r := bufio.NewReader(out)

			// Start the child process
			err = c.Start()
			if err != nil {
				panic(err)
			}

			t := ""
			for {
				line, _, err := r.ReadLine()
				if err == io.EOF && len(line) == 0 {
					break
				}

				fmt.Printf("%s\n", line)
				if bearerToken.Match(line) {
					t = bearerToken.FindStringSubmatch(string(line))[1]
				}
				// Wait to send the message until the server is running
				if strings.Contains(string(line), "Serving securely on 0.0.0.0:9443") {
					token <- t
				}

				if err == io.EOF {
					err := fmt.Errorf("Last line not terminated: %q", line)
					panic(err)
				}
				if err != nil {
					panic(err)
				}
			}
		}()

		var stdOut bytes.Buffer
		var stdErr bytes.Buffer

		fmt.Printf("Waiting for server to start...\n")
		select {
		case <-time.After(5 * time.Minute):
			if err := c.Process.Kill(); err != nil {
				glog.Fatal("failed to kill: ", err)
			}
			glog.Infof("process killed as timeout reached")
		case bearer := <-token:
			time.Sleep(1 * time.Second)
			a := []string{"-k",
				"-H", fmt.Sprintf("Authorization: Bearer %s", bearer),
				"https://localhost:9443/swagger.json",
			}
			c2 := exec.Command("curl", a...)
			c2.Stdout = &stdOut
			c2.Stderr = &stdErr

			fmt.Printf("Using curl to retrieve swagger: curl %v\n", a)
			err := c2.Run()
			if err != nil {
				panic(fmt.Errorf("Failed to get swagger spec %v: %s %s", err, stdErr.Bytes(), stdOut.Bytes()))
			}
			c.Process.Kill()
		}

		openapiFile := filepath.Join(dir, "docs", "openapi-spec", "swagger.json")
		_, err := os.Stat(openapiFile)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat file %s %v", openapiFile, err))
			}
			fmt.Printf("Creating %s file\n", openapiFile)
			f, err := os.Create(openapiFile)
			if err != nil {
				panic(err)
			}
			f.Close()
		}

		// Write the swagger file
		fmt.Printf("Writing %s file\n", openapiFile)
		err = ioutil.WriteFile(openapiFile, stdOut.Bytes(), 0644)
		if err != nil {
			panic(err)
		}
	}

	configDir := filepath.Join(dir, "docs")

	// Remove the old includes directory
	rmFiles := exec.Command("rm", "-rf", filepath.Join(configDir, "includes"))
	out, err := rmFiles.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("Failed to remove includes %s %v", out, err))

	}
	fmt.Print(string(out))

	templateDir := filepath.Join(copyFrom, "vendor", "github.com", "kubernetes-incubator", "reference-docs", "gen_open_api")
	refDocsCmd := filepath.Join(copyFrom, "vendor", "github.com", "kubernetes-incubator", "reference-docs", "main.go")
	a := []string{"run", refDocsCmd,
		"--doc-type", "open-api",
		"--allow-errors",
		"--use-tags",
		"--gen-open-api-dir", templateDir,
		"--config-dir", configDir,
	}
	runGo := exec.Command("go", a...)
	out, err = runGo.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("Failed to run go %v %s %v", a, out, err))

	}
	fmt.Print(string(out))

	broDocs := "/go/src/github.com/Birdrock/brodocs"

	// Execute in a shell to interpret *
	cleanDocs := exec.Command("bash", "-c", fmt.Sprintf("rm -rf %s/documents/*", broDocs))
	cleanDocs.Dir = broDocs
	out, err = cleanDocs.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("%s %v", out, err))
	}

	// Execute in a shell to interpret *
	copyIncludes := exec.Command("bash", "-c", fmt.Sprintf("cp -r %s/includes/* %s/documents/", configDir, broDocs))
	copyIncludes.Dir = broDocs
	out, err = copyIncludes.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("%s %v", out, err))
	}

	copyManifest := exec.Command("bash", "-c", fmt.Sprintf("cp -r %s/manifest.json %s", configDir, broDocs))
	copyManifest.Dir = broDocs
	out, err = copyManifest.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("%s %v", out, err))
	}

	runBrodocs := exec.Command("node", "brodoc.js")
	runBrodocs.Dir = broDocs
	out, err = runBrodocs.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("%s %v", out, err))
	}

	docsBuildDir := filepath.Join(configDir, "build")
	out, _ = exec.Command("mkdir", "-p", docsBuildDir).CombinedOutput()

	copyOutput := exec.Command("bash", "-c", fmt.Sprintf("cp -r %s/* %s/", broDocs, docsBuildDir))
	copyOutput.Dir = broDocs
	out, err = copyOutput.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("%s %v", out, err))
	}

	//go run main.go --doc-type open-api --config-dir ~/sample-apiserver/src/k8s.io/sample-apiserver/docs/gen_open_api/ --allow-errors

	// Write the file
	// docker run -t -i -v $(pwd)/docs/includes:/source -v $(pwd)/docs/build:/build -v $(pwd)/docs:/manifest pwittrock/brodocs
}

var genCmd = &cobra.Command{
	Use:   "generate",
	Short: "Create generated files",
	Long:  `Create generated files`,
	Run:   RunGenCmd,
}

func RunGenCmd(cmd *cobra.Command, args []string) {
	c := exec.Command("./run.sh")
	c.Env = append(c.Env, fmt.Sprintf("REPO=%s", repoName))
	c.Env = append(c.Env, fmt.Sprintf("GOPATH=%s", repoPath))
	c.Env = append(c.Env, fmt.Sprintf("DO_WIRING=%v", !skipRegister))
	c.Env = append(c.Env, fmt.Sprintf("DO_CONVERSIONS=%v", !skipConversion))
	c.Env = append(c.Env, fmt.Sprintf("DO_DEEPCOPY=%v", !skipDeepCopy))
	c.Env = append(c.Env, fmt.Sprintf("DO_OPENAPI=%s", !skipOpenApi))
	out, err := c.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("Error generating files: %v %s", err, out))
	}
	fmt.Printf("%s", out)
}

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize a directory",
	Long:  `Initialize a directory`,
	Run:   RunInit,
}

func RunInit(cmd *cobra.Command, args []string) {
	repoPath = filepath.Join(repoPath, "src", repoName)
	out, err := exec.Command("cp", "-r", filepath.Join(copyFrom, "vendor"), repoPath).CombinedOutput()
	fmt.Printf("%s", out)
	for i := 0; i < 3 && err != nil; i++ {
		out, err = exec.Command("cp", "-r", filepath.Join(copyFrom, "vendor"), repoPath).CombinedOutput()
	}
	out, err = exec.Command("cp", "-r", filepath.Join(copyFrom, "Godeps"), repoPath).CombinedOutput()
	fmt.Printf("%s", out)
	for i := 0; i < 3 && err != nil; i++ {
		out, err = exec.Command("cp", "-r", filepath.Join(copyFrom, "Godeps"), repoPath).CombinedOutput()
	}
	mainIn := filepath.Join(repoPath, "main.go")
	out, err = exec.Command("cp", "-r", filepath.Join(copyFrom, "main.go"), mainIn).CombinedOutput()
	for i := 0; i < 3 && err != nil; i++ {
		out, err = exec.Command("cp", "-r", filepath.Join(copyFrom, "Godeps"), repoPath).CombinedOutput()
	}
	fmt.Printf("%s", out)
	out, _ = exec.Command("sed",
		"-i''", // Empty suffix
		fmt.Sprintf("s$github.com/pwittrock/apiserver-helloworld$%s$g", repoName),
		mainIn,
	).CombinedOutput()

	fmt.Printf("%s", out)
	out, _ = exec.Command("mkdir", "-p", filepath.Join(repoPath, "apis")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("mkdir", "-p", filepath.Join(repoPath, "docs", "openapi-spec")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("mkdir", "-p", filepath.Join(repoPath, "docs", "static_includes")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("mkdir", "-p", filepath.Join(repoPath, "docs", "examples")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("mkdir", "-p", filepath.Join(repoPath, "pkg/openapi")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("cp", "-r", filepath.Join(copyFrom, "pkg/openapi/openapi_generated.go"), filepath.Join(repoPath, "pkg/openapi")).CombinedOutput()
	fmt.Printf("%s", out)
	out, _ = exec.Command("cp", "-r", filepath.Join(copyFrom, "pkg/openapi/doc.go"), filepath.Join(repoPath, "pkg/openapi")).CombinedOutput()
	fmt.Printf("%s", out)
}

var addTypesCmd = &cobra.Command{
	Use:   "add-types",
	Short: "Create new entries for group/version/kind types",
	Long:  `Specify types using group/version/kind`,
	Run:   RunAddTypes,
}

func RunAddTypes(cmd *cobra.Command, args []string) {
	repoPath = filepath.Join(repoPath, "src", repoName)

	groups := sets.String{}
	groupVersions := sets.String{}
	kindsToGroupVersion := map[string]string{}
	for _, tuple := range types {
		groupVersionKind := strings.Split(tuple, "/")
		groups.Insert(groupVersionKind[0])
		gv := filepath.Join(groupVersionKind[0], groupVersionKind[1])
		kindsToGroupVersion[groupVersionKind[2]] = gv
		groupVersions.Insert(gv)
	}

	for _, gv := range groupVersions.List() {
		split := strings.Split(gv, "/")
		group := split[0]
		version := split[1]

		path := filepath.Join(repoPath, "apis", gv)
		_, err := os.Stat(path)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat directory %s %v", path, err))
			}
			fmt.Printf("Creating directory %s\n", path)
			out, err := exec.Command("mkdir", "-p", path).CombinedOutput()
			if err != nil {
				fmt.Printf("Failed to create directory %s %v %s", path, err, out)
			}
		}

		apisDocPath := filepath.Join(repoPath, "apis/doc.go")
		_, err = os.Stat(apisDocPath)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat file %s %v", apisDocPath, err))
			}
			fmt.Printf("Creating file %s\n", apisDocPath)
			t := template.Must(template.New("apis-doc-template").Parse(apisDocTemplate))
			f, err := os.Create(apisDocPath)
			if err != nil {
				fmt.Println(err)
				return
			}
			f.Close()

			f, err = os.OpenFile(apisDocPath, os.O_WRONLY, 0)
			err = t.Execute(f, ApisDocTemplateArguments{
				Domain: domain,
			})
			if err != nil {
				fmt.Println(err)
			}
			f.Close()
		}

		typesgo := filepath.Join(path, "types.go")
		_, err = os.Stat(typesgo)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat file %s %v", typesgo, err))
			}
			t := template.Must(template.New("new-types-template").Parse(newTypesTemplate))
			f, err := os.Create(typesgo)
			if err != nil {
				fmt.Println(err)
				return
			}
			f.Close()

			f, err = os.OpenFile(typesgo, os.O_WRONLY, 0)
			err = t.Execute(f, NewTypesGoArguments{
				Package: version,
			})
			if err != nil {
				fmt.Println(err)
			}
			f.Close()
		}

		docgo := filepath.Join(path, "doc.go")
		_, err = os.Stat(docgo)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat file %s %v", docgo, err))
			}

			t := template.Must(template.New("new-doc-template").Parse(newVersionDocTemplate))
			f, err := os.Create(docgo)
			if err != nil {
				fmt.Println(err)
				return
			}
			f.Close()

			f, err = os.OpenFile(docgo, os.O_WRONLY, 0)
			err = t.Execute(f, NewDocTemplateArguments{version, filepath.Join(repoName, "apis", group), group})
			if err != nil {
				fmt.Println(err)
			}
			f.Close()
		}

		groupdocgo := filepath.Join(repoPath, "apis", group, "doc.go")
		_, err = os.Stat(groupdocgo)
		if err != nil {
			if !os.IsNotExist(err) {
				panic(fmt.Sprintf("Could not stat file %s %v", groupdocgo, err))
			}

			t := template.Must(template.New("new-group-doc-template").Parse(newGroupDocTemplate))
			f, err := os.Create(groupdocgo)
			if err != nil {
				fmt.Println(err)
				return
			}
			f.Close()

			f, err = os.OpenFile(groupdocgo, os.O_WRONLY, 0)
			err = t.Execute(f, NewDocTemplateArguments{version, filepath.Join(repoName, "apis", group), group})
			if err != nil {
				fmt.Println(err)
			}
			f.Close()
		}
	}

	for k, gv := range kindsToGroupVersion {
		t := template.Must(template.New("add-types-template").Parse(addTypesTemplate))
		path := filepath.Join(repoPath, "apis", gv)

		typesgo := filepath.Join(path, "types.go")
		f, err := os.Open(typesgo)
		if err != nil {
			panic(err)
			return
		}

		contents, err := ioutil.ReadAll(f)
		if err != nil {
			panic(err)
			return
		}
		if strings.Contains(string(contents), fmt.Sprintf("type %s struct {", k)) {
			fmt.Printf("Skipping kind %s\n", k)
			f.Close()
			continue
		}
		f.Close()

		f, err = os.OpenFile(typesgo, os.O_WRONLY|os.O_APPEND, 0)
		err = t.Execute(f, AddTypeArguments{
			Kind:     k,
			Resource: fmt.Sprintf("%ss", strings.ToLower(k)),
		})
		if err != nil {
			fmt.Println(err)
		}
		f.Close()
	}
}

type AddTypeArguments struct {
	Resource string
	Kind     string
}

var addTypesTemplate = (`
// +genclient=true
// +genapi=true
// +resource={{.Resource}}
// +k8s:openapi-gen=true
type {{.Kind}} struct {
	metav1.TypeMeta   ` + "`json:\",inline\"`" + `
	metav1.ObjectMeta ` + "`json:\"metadata,omitempty\"`" + `

	Spec   {{.Kind}}Spec   ` + "`json:\"spec,omitempty\"`" + `
	Status {{.Kind}}Status ` + "`json:\"status,omitempty\"`" + `
}

type {{.Kind}}Spec struct {
}

type {{.Kind}}Status struct {
}
`)

type NewTypesGoArguments struct {
	Package string
}

var newTypesTemplate = (`

package {{.Package}}

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

`)

type NewDocTemplateArguments struct {
	Version string
	Package string
	Group   string
}

var newVersionDocTemplate = `
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

// +k8s:openapi-gen=true
// +k8s:deepcopy-gen=package,register
// +k8s:conversion-gen={{.Package}}

package {{.Version}}
`

var newGroupDocTemplate = `
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

// +k8s:deepcopy-gen=package,register

// Package api is the internal version of the API.
package {{.Group}}

`

type ApisDocTemplateArguments struct {
	Domain string
}

var apisDocTemplate = `
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

//
// +domain={{.Domain}}

package apis

`

var openApiDoc = `/*
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

// Package openapi exists to hold generated openapi code
package openapi

`
