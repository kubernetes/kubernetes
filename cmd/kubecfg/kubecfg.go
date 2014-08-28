/*
Copyright 2014 Google Inc. All rights reserved.

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
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubecfg"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	verflag "github.com/GoogleCloudPlatform/kubernetes/pkg/version/flag"
	"github.com/golang/glog"
)

var (
	serverVersion = flag.Bool("server_version", false, "Print the server's version number.")
	preventSkew   = flag.Bool("expect_version_match", false, "Fail if server's version doesn't match own version.")
	httpServer    = flag.String("h", "", "The host to connect to.")
	config        = flag.String("c", "", "Path to the config file.")
	selector      = flag.String("l", "", "Selector (label query) to use for listing")
	updatePeriod  = flag.Duration("u", 60*time.Second, "Update interval period")
	portSpec      = flag.String("p", "", "The port spec, comma-separated list of <external>:<internal>,...")
	servicePort   = flag.Int("s", -1, "If positive, create and run a corresponding service on this port, only used with 'run'")
	authConfig    = flag.String("auth", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.  If missing, prompt the user.  Only used if doing https.")
	json          = flag.Bool("json", false, "If true, print raw JSON for responses")
	yaml          = flag.Bool("yaml", false, "If true, print raw YAML for responses")
	verbose       = flag.Bool("verbose", false, "If true, print extra information")
	proxy         = flag.Bool("proxy", false, "If true, run a proxy to the api server")
	www           = flag.String("www", "", "If -proxy is true, use this directory to serve static files")
	templateFile  = flag.String("template_file", "", "If present, load this file as a golang template and use it for output printing")
	templateStr   = flag.String("template", "", "If present, parse this string as a golang template and use it for output printing")
)

var parser = kubecfg.NewParser(map[string]interface{}{
	"pods":                   api.Pod{},
	"services":               api.Service{},
	"replicationControllers": api.ReplicationController{},
	"minions":                api.Minion{},
})

func usage() {
	fmt.Fprintf(os.Stderr, `usage: kubecfg -h [-c config/file.json] [-p :,..., :] <method>

  Kubernetes REST API:
  kubecfg [OPTIONS] get|list|create|delete|update <%s>[/<id>]

  Manage replication controllers:
  kubecfg [OPTIONS] stop|rm|rollingupdate <controller>
  kubecfg [OPTIONS] run <image> <replicas> <controller>
  kubecfg [OPTIONS] resize <controller> <replicas>

  Options:
`, prettyWireStorage())
	flag.PrintDefaults()

}

func prettyWireStorage() string {
	types := parser.SupportedWireStorage()
	sort.Strings(types)
	return strings.Join(types, "|")
}

// readConfig reads and parses pod, replicationController, and service
// configuration files. If any errors log and exit non-zero.
func readConfig(storage string) []byte {
	if len(*config) == 0 {
		glog.Fatal("Need config file (-c)")
	}
	data, err := ioutil.ReadFile(*config)
	if err != nil {
		glog.Fatalf("Unable to read %v: %v\n", *config, err)
	}
	data, err = parser.ToWireFormat(data, storage)
	if err != nil {
		glog.Fatalf("Error parsing %v as an object for %v: %v\n", *config, storage, err)
	}
	if *verbose {
		glog.Infof("Parsed config file successfully; sending:\n%v\n", string(data))
	}
	return data
}

func main() {
	flag.Usage = func() {
		usage()
	}

	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	var masterServer string
	if len(*httpServer) > 0 {
		masterServer = *httpServer
	} else if len(os.Getenv("KUBERNETES_MASTER")) > 0 {
		masterServer = os.Getenv("KUBERNETES_MASTER")
	} else {
		masterServer = "http://localhost:8080"
	}
	kubeClient, err := client.New(masterServer, nil)
	if err != nil {
		glog.Fatalf("Unable to parse %s as a URL: %v", masterServer, err)
	}

	// TODO: this won't work if TLS is enabled with client cert auth, but no
	// passwords are required. Refactor when we address client auth abstraction.
	if kubeClient.Secure() {
		auth, err := kubecfg.LoadAuthInfo(*authConfig, os.Stdin)
		if err != nil {
			glog.Fatalf("Error loading auth: %v", err)
		}
		kubeClient, err = client.New(masterServer, auth)
		if err != nil {
			glog.Fatalf("Unable to parse %s as a URL: %v", masterServer, err)
		}
	}

	if *serverVersion {
		got, err := kubeClient.ServerVersion()
		if err != nil {
			fmt.Printf("Couldn't read version from server: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Server Version: %#v\n", got)
		os.Exit(0)
	}

	if *preventSkew {
		got, err := kubeClient.ServerVersion()
		if err != nil {
			fmt.Printf("Couldn't read version from server: %v\n", err)
			os.Exit(1)
		}
		if c, s := version.Get(), *got; !reflect.DeepEqual(c, s) {
			fmt.Printf("Server version (%#v) differs from client version (%#v)!\n", s, c)
			os.Exit(1)
		}
	}

	if *proxy {
		glog.Info("Starting to serve on localhost:8001")
		server := kubecfg.NewProxyServer(*www, kubeClient)
		glog.Fatal(server.Serve())
	}

	if len(flag.Args()) < 1 {
		usage()
		os.Exit(1)
	}
	method := flag.Arg(0)

	matchFound := executeAPIRequest(method, kubeClient) || executeControllerRequest(method, kubeClient)
	if matchFound == false {
		glog.Fatalf("Unknown command %s", method)
	}
}

// storagePathFromArg normalizes a path and breaks out the first segment if available
func storagePathFromArg(arg string) (storage, path string, hasSuffix bool) {
	path = strings.Trim(arg, "/")
	segments := strings.SplitN(path, "/", 2)
	storage = segments[0]
	if len(segments) > 1 && segments[1] != "" {
		hasSuffix = true
	}
	return storage, path, hasSuffix
}

//checkStorage returns true if the provided storage is valid
func checkStorage(storage string) bool {
	for _, allowed := range parser.SupportedWireStorage() {
		if allowed == storage {
			return true
		}
	}
	return false
}

func executeAPIRequest(method string, c *client.Client) bool {
	storage, path, hasSuffix := storagePathFromArg(flag.Arg(1))
	validStorage := checkStorage(storage)
	verb := ""
	setBody := false
	var version uint64
	switch method {
	case "get":
		verb = "GET"
		if !validStorage || !hasSuffix {
			glog.Fatalf("usage: kubecfg [OPTIONS] %s <%s>[/<id>]", method, prettyWireStorage())
		}
	case "list":
		verb = "GET"
		if !validStorage || hasSuffix {
			glog.Fatalf("usage: kubecfg [OPTIONS] %s <%s>", method, prettyWireStorage())
		}
	case "delete":
		verb = "DELETE"
		if !validStorage || !hasSuffix {
			glog.Fatalf("usage: kubecfg [OPTIONS] %s <%s>/<id>", method, prettyWireStorage())
		}
	case "create":
		verb = "POST"
		setBody = true
		if !validStorage || hasSuffix {
			glog.Fatalf("usage: kubecfg [OPTIONS] %s <%s>", method, prettyWireStorage())
		}
	case "update":
		obj, err := c.Verb("GET").Path(path).Do().Get()
		if err != nil {
			glog.Fatalf("error obtaining resource version for update: %v", err)
		}
		jsonBase, err := api.FindJSONBase(obj)
		if err != nil {
			glog.Fatalf("error finding json base for update: %v", err)
		}
		version = jsonBase.ResourceVersion()
		verb = "PUT"
		setBody = true
		if !validStorage || !hasSuffix {
			glog.Fatalf("usage: kubecfg [OPTIONS] %s <%s>/<id>", method, prettyWireStorage())
		}
	default:
		return false
	}

	r := c.Verb(verb).
		Path(path).
		ParseSelectorParam("labels", *selector)
	if setBody {
		if version != 0 {
			data := readConfig(storage)
			obj, err := api.Decode(data)
			if err != nil {
				glog.Fatalf("error setting resource version: %v", err)
			}
			jsonBase, err := api.FindJSONBase(obj)
			if err != nil {
				glog.Fatalf("error setting resource version: %v", err)
			}
			jsonBase.SetResourceVersion(version)
			data, err = api.Encode(obj)
			if err != nil {
				glog.Fatalf("error setting resource version: %v", err)
			}
			r.Body(data)
		} else {
			r.Body(readConfig(storage))
		}
	}
	result := r.Do()
	obj, err := result.Get()
	if err != nil {
		glog.Fatalf("Got request error: %v\n", err)
		return false
	}

	var printer kubecfg.ResourcePrinter
	switch {
	case *json:
		printer = &kubecfg.IdentityPrinter{}
	case *yaml:
		printer = &kubecfg.YAMLPrinter{}
	case len(*templateFile) > 0 || len(*templateStr) > 0:
		var data []byte
		if len(*templateFile) > 0 {
			var err error
			data, err = ioutil.ReadFile(*templateFile)
			if err != nil {
				glog.Fatalf("Error reading template %s, %v\n", *templateFile, err)
				return false
			}
		} else {
			data = []byte(*templateStr)
		}
		tmpl, err := template.New("output").Parse(string(data))
		if err != nil {
			glog.Fatalf("Error parsing template %s, %v\n", string(data), err)
			return false
		}
		printer = &kubecfg.TemplatePrinter{
			Template: tmpl,
		}
	default:
		printer = humanReadablePrinter()
	}

	if err = printer.PrintObj(obj, os.Stdout); err != nil {
		body, _ := result.Raw()
		glog.Fatalf("Failed to print: %v\nRaw received object:\n%#v\n\nBody received: %v", err, obj, string(body))
	}
	fmt.Print("\n")

	return true
}

func executeControllerRequest(method string, c *client.Client) bool {
	parseController := func() string {
		if len(flag.Args()) != 2 {
			glog.Fatal("usage: kubecfg [OPTIONS] stop|rm|rollingupdate <controller>")
		}
		return flag.Arg(1)
	}

	var err error
	switch method {
	case "stop":
		err = kubecfg.StopController(parseController(), c)
	case "rm":
		err = kubecfg.DeleteController(parseController(), c)
	case "rollingupdate":
		err = kubecfg.Update(parseController(), c, *updatePeriod)
	case "run":
		if len(flag.Args()) != 4 {
			glog.Fatal("usage: kubecfg [OPTIONS] run <image> <replicas> <controller>")
		}
		image := flag.Arg(1)
		replicas, err2 := strconv.Atoi(flag.Arg(2))
		if err2 != nil {
			glog.Fatalf("Error parsing replicas: %v", err2)
		}
		name := flag.Arg(3)
		err = kubecfg.RunController(image, name, replicas, c, *portSpec, *servicePort)
	case "resize":
		args := flag.Args()
		if len(args) < 3 {
			glog.Fatal("usage: kubecfg resize <controller> <replicas>")
		}
		name := args[1]
		replicas, err2 := strconv.Atoi(args[2])
		if err2 != nil {
			glog.Fatalf("Error parsing replicas: %v", err2)
		}
		err = kubecfg.ResizeController(name, replicas, c)
	default:
		return false
	}
	if err != nil {
		glog.Fatalf("Error: %v", err)
	}
	return true
}

func humanReadablePrinter() *kubecfg.HumanReadablePrinter {
	printer := kubecfg.NewHumanReadablePrinter()
	// Add Handler calls here to support additional types
	return printer
}
