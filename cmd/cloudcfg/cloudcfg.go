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
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	kube_client "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudcfg"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const APP_VERSION = "0.1"

// The flag package provides a default help printer via -h switch
var (
	versionFlag  = flag.Bool("v", false, "Print the version number.")
	httpServer   = flag.String("h", "", "The host to connect to.")
	config       = flag.String("c", "", "Path to the config file.")
	selector     = flag.String("l", "", "Selector (label query) to use for listing")
	updatePeriod = flag.Duration("u", 60*time.Second, "Update interarrival period")
	portSpec     = flag.String("p", "", "The port spec, comma-separated list of <external>:<internal>,...")
	servicePort  = flag.Int("s", -1, "If positive, create and run a corresponding service on this port, only used with 'run'")
	authConfig   = flag.String("auth", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.  If missing, prompt the user.  Only used if doing https.")
	json         = flag.Bool("json", false, "If true, print raw JSON for responses")
	yaml         = flag.Bool("yaml", false, "If true, print raw YAML for responses")
	verbose      = flag.Bool("verbose", false, "If true, print extra information")
	proxy        = flag.Bool("proxy", false, "If true, run a proxy to the api server")
	www          = flag.String("www", "", "If -proxy is true, use this directory to serve static files")
)

func usage() {
	fmt.Fprint(os.Stderr, `usage: cloudcfg -h [-c config/file.json] [-p :,..., :] <method>

  Kubernetes REST API:
  cloudcfg [OPTIONS] get|list|create|delete|update <url>

  Manage replication controllers:
  cloudcfg [OPTIONS] stop|rm|rollingupdate <controller>
  cloudcfg [OPTIONS] run <image> <replicas> <controller>
  cloudcfg [OPTIONS] resize <controller> <replicas>

  Options:
`)
	flag.PrintDefaults()
}

// Reads & parses config file. On error, calls glog.Fatal().
func readConfig(storage string) []byte {
	if len(*config) == 0 {
		glog.Fatal("Need config file (-c)")
	}
	data, err := ioutil.ReadFile(*config)
	if err != nil {
		glog.Fatalf("Unable to read %v: %#v\n", *config, err)
	}
	data, err = cloudcfg.ToWireFormat(data, storage)
	if err != nil {
		glog.Fatalf("Error parsing %v as an object for %v: %#v\n", *config, storage, err)
	}
	if *verbose {
		glog.Infof("Parsed config file successfully; sending:\n%v\n", string(data))
	}
	return data
}

// CloudCfg command line tool.
func main() {
	flag.Usage = func() {
		usage()
	}

	flag.Parse() // Scan the arguments list
	util.InitLogs()
	defer util.FlushLogs()

	if *versionFlag {
		fmt.Println("Version:", APP_VERSION)
		os.Exit(0)
	}

	secure := true
	parsedUrl, err := url.Parse(*httpServer)
	if err != nil {
		glog.Fatalf("Unable to parse %v as a URL\n", err)
	}
	if parsedUrl.Scheme != "" && parsedUrl.Scheme != "https" {
		secure = false
	}

	var auth *kube_client.AuthInfo
	if secure {
		auth, err = cloudcfg.LoadAuthInfo(*authConfig)
		if err != nil {
			glog.Fatalf("Error loading auth: %#v", err)
		}
	}

	if *proxy {
		glog.Info("Starting to serve on localhost:8001")
		server := cloudcfg.NewProxyServer(*www, *httpServer, auth)
		glog.Fatal(server.Serve())
	}

	if len(flag.Args()) < 1 {
		usage()
		os.Exit(1)
	}
	method := flag.Arg(0)

	matchFound := executeAPIRequest(method, auth) || executeControllerRequest(method, auth)
	if matchFound == false {
		glog.Fatalf("Unknown command %s", method)
	}
}

// Attempts to execute an API request
func executeAPIRequest(method string, auth *kube_client.AuthInfo) bool {
	parseStorage := func() string {
		if len(flag.Args()) != 2 {
			glog.Fatal("usage: cloudcfg [OPTIONS] get|list|create|update|delete <url>")
		}
		return strings.Trim(flag.Arg(1), "/")
	}

	verb := ""
	switch method {
	case "get", "list":
		verb = "GET"
	case "delete":
		verb = "DELETE"
	case "create":
		verb = "POST"
	case "update":
		verb = "PUT"
	default:
		return false
	}

	s := kube_client.New(*httpServer, auth)
	r := s.Verb(verb).
		Path(parseStorage()).
		ParseSelector(*selector)
	if method == "create" || method == "update" {
		r.Body(readConfig(parseStorage()))
	}
	obj, err := r.Do().Get()
	if err != nil {
		glog.Fatalf("Got request error: %v\n", err)
		return false
	}

	var printer cloudcfg.ResourcePrinter
	if *json {
		printer = &cloudcfg.IdentityPrinter{}
	} else if *yaml {
		printer = &cloudcfg.YAMLPrinter{}
	} else {
		printer = &cloudcfg.HumanReadablePrinter{}
	}

	if err = printer.PrintObj(obj, os.Stdout); err != nil {
		glog.Fatalf("Failed to print: %#v\nRaw received object:\n%#v\n", err, obj)
	}
	fmt.Print("\n")

	return true
}

// Attempts to execute a replicationController request
func executeControllerRequest(method string, auth *kube_client.AuthInfo) bool {
	parseController := func() string {
		if len(flag.Args()) != 2 {
			glog.Fatal("usage: cloudcfg [OPTIONS] stop|rm|rollingupdate <controller>")
		}
		return flag.Arg(1)
	}

	c := kube_client.New(*httpServer, auth)

	var err error
	switch method {
	case "stop":
		err = cloudcfg.StopController(parseController(), c)
	case "rm":
		err = cloudcfg.DeleteController(parseController(), c)
	case "rollingupdate":
		err = cloudcfg.Update(parseController(), c, *updatePeriod)
	case "run":
		if len(flag.Args()) != 4 {
			glog.Fatal("usage: cloudcfg [OPTIONS] run <image> <replicas> <controller>")
		}
		image := flag.Arg(1)
		replicas, err := strconv.Atoi(flag.Arg(2))
		name := flag.Arg(3)
		if err != nil {
			glog.Fatalf("Error parsing replicas: %#v", err)
		}
		err = cloudcfg.RunController(image, name, replicas, c, *portSpec, *servicePort)
	case "resize":
		args := flag.Args()
		if len(args) < 3 {
			glog.Fatal("usage: cloudcfg resize <controller> <replicas>")
		}
		name := args[1]
		replicas, err := strconv.Atoi(args[2])
		if err != nil {
			glog.Fatalf("Error parsing replicas: %#v", err)
		}
		err = cloudcfg.ResizeController(name, replicas, c)
	default:
		return false
	}
	if err != nil {
		glog.Fatalf("Error: %#v", err)
	}
	return true
}
