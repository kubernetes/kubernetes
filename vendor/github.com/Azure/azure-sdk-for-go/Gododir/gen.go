package main

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// To run this package...
// go run gen.go -- --sdk 3.14.16

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	do "gopkg.in/godo.v2"
)

type service struct {
	Name      string
	Fullname  string
	Namespace string
	TaskName  string
	Tag       string
	Input     string
	Output    string
}

const (
	testsSubDir = "tests"
)

type mapping struct {
	PlaneInput  string
	PlaneOutput string
	Services    []service
}

type failList []string

type failLocker struct {
	sync.Mutex
	failList
}

var (
	start           time.Time
	gopath          = os.Getenv("GOPATH")
	sdkVersion      string
	autorestDir     string
	swaggersDir     string
	testGen         bool
	deps            do.P
	services        = []*service{}
	servicesMapping = []mapping{
		{
			PlaneOutput: "arm",
			PlaneInput:  "resource-manager",
			Services: []service{
				{Name: "advisor"},
				{Name: "analysisservices"},
				// {
				// Autorest Bug, duplicate files
				// 	Name: "apimanagement",
				// },
				{Name: "appinsights"},
				{Name: "authorization"},
				{Name: "automation"},
				// {
				// 	Name:   "commerce",
				// 	Input:  "azsadmin/resource-manager/commerce",
				// 	Output: "azsadmin/commerce",
				// },
				// {
				// 	Name:   "fabric",
				// 	Input:  "azsadmin/resource-manager/fabric",
				// 	Output: "azsadmin/fabric",
				// },
				// {
				// 	Name:   "infrastructureinsights",
				// 	Input:  "azsadmin/resource-manager/InfrastructureInsights",
				// 	Output: "azsadmin/infrastructureinsights",
				// },
				{Name: "batch"},
				{Name: "billing"},
				{Name: "cdn"},
				{Name: "cognitiveservices"},
				{Name: "commerce"},
				{Name: "compute"},
				{
					Name:  "containerservice",
					Input: "compute/resource-manager",
					Tag:   "package-container-service-2017-01",
				},
				{Name: "consumption"},
				{Name: "containerinstance"},
				{Name: "containerregistry"},
				{Name: "cosmos-db"},
				{Name: "customer-insights"},
				{
					Name:   "account",
					Input:  "datalake-analytics/resource-manager",
					Output: "datalake-analytics/account",
				},
				{
					Name:   "account",
					Input:  "datalake-store/resource-manager",
					Output: "datalake-store/account",
				},
				{Name: "devtestlabs"},
				{Name: "dns"},
				{Name: "eventgrid"},
				{Name: "eventhub"},
				{Name: "hdinsight"},
				{Name: "intune"},
				{Name: "iothub"},
				{Name: "keyvault"},
				{Name: "logic"},
				{
					Name:   "commitmentplans",
					Input:  "machinelearning/resource-manager",
					Output: "machinelearning/commitmentplans",
					Tag:    "package-commitmentPlans-2016-05-preview",
				},
				{
					Name:   "webservices",
					Input:  "machinelearning/resource-manager",
					Output: "machinelearning/webservices",
					Tag:    "package-webservices-2017-01",
				},
				{Name: "marketplaceordering"},
				{Name: "mediaservices"},
				{Name: "mobileengagement"},
				{Name: "monitor"},
				{Name: "mysql"},
				{Name: "network"},
				{Name: "notificationhubs"},
				{Name: "operationalinsights"},
				{Name: "operationsmanagement"},
				{Name: "postgresql"},
				{Name: "powerbiembedded"},
				{Name: "recoveryservices"},
				{Name: "recoveryservicesbackup"},
				{Name: "recoveryservicessiterecovery"},
				{
					Name: "redis",
					Tag:  "package-2016-04",
				},
				{Name: "relay"},
				{Name: "resourcehealth"},
				{
					Name:   "features",
					Input:  "resources/resource-manager",
					Output: "resources/features",
					Tag:    "package-features-2015-12",
				},
				{
					Name:   "links",
					Input:  "resources/resource-manager",
					Output: "resources/links",
					Tag:    "package-links-2016-09",
				},
				{
					Name:   "locks",
					Input:  "resources/resource-manager",
					Output: "resources/locks",
					Tag:    "package-locks-2016-09",
				},
				{
					Name:   "managedapplications",
					Input:  "resources/resource-manager",
					Output: "resources/managedapplications",
					Tag:    "package-managedapplications-2016-09",
				},
				{
					Name:   "policy",
					Input:  "resources/resource-manager",
					Output: "resources/policy",
					Tag:    "package-policy-2016-12",
				},
				{
					Name:   "resources",
					Input:  "resources/resource-manager",
					Output: "resources/resources",
					Tag:    "package-resources-2017-05",
				},
				{
					Name:   "subscriptions",
					Input:  "resources/resource-manager",
					Output: "resources/subscriptions",
					Tag:    "package-subscriptions-2016-06",
				},
				{Name: "scheduler"},
				{Name: "search"},
				{Name: "servermanagement"},
				{Name: "service-map"},
				{Name: "servicebus"},
				{Name: "servicefabric"},
				{Name: "sql"},
				{Name: "storage"},
				{Name: "storageimportexport"},
				{Name: "storsimple8000series"},
				{Name: "streamanalytics"},
				// {
				// error in the modeler
				// https://github.com/Azure/autorest/issues/2579
				// Name: "timeseriesinsights",
				// },
				{Name: "trafficmanager"},
				{Name: "visualstudio"},
				{Name: "web"},
			},
		},
		{
			PlaneOutput: "dataplane",
			PlaneInput:  "data-plane",
			Services: []service{
				{Name: "keyvault"},
				{
					Name:   "face",
					Input:  "cognitiveservices/data-plane/Face",
					Output: "cognitiveservices/face",
				},
				{
					Name:   "textanalytics",
					Input:  "cognitiveservices/data-plane/TextAnalytics",
					Output: "cognitiveservices/textanalytics",
				},
			},
		},
		{
			PlaneInput: "data-plane",
			Services: []service{
				{
					Name:   "filesystem",
					Input:  "datalake-store/data-plane",
					Output: "datalake-store/filesystem",
				},
			},
		},
		{
			PlaneOutput: "arm",
			PlaneInput:  "data-plane",
			Services: []service{
				{Name: "graphrbac"},
			},
		},
	}
	fails = failLocker{}
)

func init() {
	start = time.Now()
	for _, swaggerGroup := range servicesMapping {
		swg := swaggerGroup
		for _, service := range swg.Services {
			s := service
			initAndAddService(&s, swg.PlaneInput, swg.PlaneOutput)
		}
	}
}

func main() {
	do.Godo(tasks)
}

func initAndAddService(service *service, planeInput, planeOutput string) {
	if service.Input == "" {
		service.Input = service.Name
	}
	path := []string{service.Input}
	if service.Input == service.Name {
		path = append(path, planeInput)
	}
	path = append(path, "readme.md")
	service.Input = filepath.Join(path...)

	if service.Output == "" {
		service.Output = service.Name
	}
	service.TaskName = fmt.Sprintf("%s>%s", planeOutput, strings.Join(strings.Split(service.Output, "/"), ">"))
	service.Fullname = filepath.Join(planeOutput, service.Output)
	service.Namespace = filepath.Join("github.com", "Azure", "azure-sdk-for-go", service.Fullname)
	service.Output = filepath.Join(gopath, "src", service.Namespace)

	services = append(services, service)
	deps = append(deps, service.TaskName)
}

func tasks(p *do.Project) {
	p.Task("default", do.S{"setvars", "generate:all", "management", "report"}, nil)
	p.Task("setvars", nil, setVars)
	p.Use("generate", generateTasks)
	p.Use("gofmt", formatTasks)
	p.Use("gobuild", buildTasks)
	p.Use("golint", lintTasks)
	p.Use("govet", vetTasks)
	p.Task("management", do.S{"setvars"}, managementVersion)
	p.Task("addVersion", nil, addVersion)
	p.Task("report", nil, report)
}

func setVars(c *do.Context) {
	if gopath == "" {
		panic("Gopath not set\n")
	}

	sdkVersion = c.Args.MustString("s", "sdk", "version")
	autorestDir = c.Args.MayString("", "a", "ar", "autorest")
	swaggersDir = c.Args.MayString("", "w", "sw", "swagger")
	testGen = c.Args.MayBool(false, "t", "testgen")
}

func generateTasks(p *do.Project) {
	addTasks(generate, p)
}

func generate(service *service) {
	codegen := "--go"
	if testGen {
		codegen = "--go.testgen"
		service.Fullname = strings.Join([]string{service.Fullname, testsSubDir}, string(os.PathSeparator))
		service.Output = filepath.Join(service.Output, testsSubDir)
	}

	fmt.Printf("Generating %s...\n\n", service.Fullname)

	fullInput := ""
	if swaggersDir == "" {
		fullInput = fmt.Sprintf("https://raw.githubusercontent.com/Azure/azure-rest-api-specs/current/specification/%s", service.Input)
	} else {
		fullInput = filepath.Join(swaggersDir, "azure-rest-api-specs", "specification", service.Input)
	}

	execCommand := "autorest"
	commandArgs := []string{
		fullInput,
		codegen,
		"--license-header=MICROSOFT_APACHE_NO_VERSION",
		fmt.Sprintf("--namespace=%s", service.Name),
		fmt.Sprintf("--output-folder=%s", service.Output),
		fmt.Sprintf("--package-version=%s", sdkVersion),
		"--clear-output-folder",
		"--can-clear-output-folder",
	}
	if service.Tag != "" {
		commandArgs = append(commandArgs, fmt.Sprintf("--tag=%s", service.Tag))
	}
	if testGen {
		commandArgs = append([]string{"-LEGACY"}, commandArgs...)
	}

	if autorestDir != "" {
		// if an AutoRest directory was specified then assume
		// the caller wants to use a locally-built version.
		commandArgs = append(commandArgs, fmt.Sprintf("--use=%s", autorestDir))
	}

	autorest := exec.Command(execCommand, commandArgs...)

	fmt.Println(commandArgs)

	if _, stderr, err := runner(autorest); err != nil {
		fails.Add(fmt.Sprintf("%s: autorest error: %s: %s", service.Fullname, err, stderr))
	}

	format(service)
	build(service)
	lint(service)
	vet(service)
}

func formatTasks(p *do.Project) {
	addTasks(format, p)
}

func format(service *service) {
	fmt.Printf("Formatting %s...\n\n", service.Fullname)
	gofmt := exec.Command("gofmt", "-w", service.Output)
	_, stderr, err := runner(gofmt)
	if err != nil {
		fails.Add(fmt.Sprintf("%s: gofmt error:%s: %s", service.Fullname, err, stderr))
	}
}

func buildTasks(p *do.Project) {
	addTasks(build, p)
}

func build(service *service) {
	fmt.Printf("Building %s...\n\n", service.Fullname)
	gobuild := exec.Command("go", "build", service.Namespace)
	_, stderr, err := runner(gobuild)
	if err != nil {
		fails.Add(fmt.Sprintf("%s: build error: %s: %s", service.Fullname, err, stderr))
	}
}

func lintTasks(p *do.Project) {
	addTasks(lint, p)
}

func lint(service *service) {
	fmt.Printf("Linting %s...\n\n", service.Fullname)
	golint := exec.Command(filepath.Join(gopath, "bin", "golint"), service.Namespace)
	_, stderr, err := runner(golint)
	if err != nil {
		fails.Add(fmt.Sprintf("%s: golint error: %s: %s", service.Fullname, err, stderr))
	}
}

func vetTasks(p *do.Project) {
	addTasks(vet, p)
}

func vet(service *service) {
	fmt.Printf("Vetting %s...\n\n", service.Fullname)
	govet := exec.Command("go", "vet", service.Namespace)
	_, stderr, err := runner(govet)
	if err != nil {
		fails.Add(fmt.Sprintf("%s: go vet error: %s: %s", service.Fullname, err, stderr))
	}
}

func addVersion(c *do.Context) {
	gitStatus := exec.Command("git", "status", "-s")
	out, _, err := runner(gitStatus)
	if err != nil {
		panic(fmt.Errorf("Git error: %s", err))
	}
	files := strings.Split(out, "\n")

	for _, f := range files {
		if strings.HasPrefix(f, " M ") && strings.HasSuffix(f, "version.go") {
			gitAdd := exec.Command("git", "add", f[3:])
			_, _, err := runner(gitAdd)
			if err != nil {
				panic(fmt.Errorf("Git error: %s", err))
			}
		}
	}
}

func managementVersion(c *do.Context) {
	version("management")
}

func version(packageName string) {
	versionFile := filepath.Join(packageName, "version.go")
	os.Remove(versionFile)
	template := `// +build go1.7

package %s

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

var (
	sdkVersion = "%s"
)
`
	data := []byte(fmt.Sprintf(template, packageName, sdkVersion))
	ioutil.WriteFile(versionFile, data, 0644)
}

func addTasks(fn func(*service), p *do.Project) {
	for _, service := range services {
		s := service
		p.Task(s.TaskName, nil, func(c *do.Context) {
			fn(s)
		})
	}
	p.Task("all", deps, nil)
}

func runner(cmd *exec.Cmd) (string, string, error) {
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if stdout.Len() > 0 {
		fmt.Println(stdout.String())
	}
	if stderr.Len() > 0 {
		fmt.Println(stderr.String())
	}
	return stdout.String(), stderr.String(), err
}

func (fl *failLocker) Add(fail string) {
	fl.Lock()
	defer fl.Unlock()
	fl.failList = append(fl.failList, fail)
}

func report(c *do.Context) {
	fmt.Printf("Script ran for %s\n", time.Since(start))
	for _, f := range fails.failList {
		fmt.Println(f)
		fmt.Println("==========")
	}
}
