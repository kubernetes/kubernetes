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
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strconv"
	"time"

	kube_client "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudcfg"
)

const APP_VERSION = "0.1"

// The flag package provides a default help printer via -h switch
var (
	versionFlag  = flag.Bool("v", false, "Print the version number.")
	httpServer   = flag.String("h", "", "The host to connect to.")
	config       = flag.String("c", "", "Path to the config file.")
	labelQuery   = flag.String("l", "", "Label query to use for listing")
	updatePeriod = flag.Duration("u", 60*time.Second, "Update interarrival period")
	portSpec     = flag.String("p", "", "The port spec, comma-separated list of <external>:<internal>,...")
	servicePort  = flag.Int("s", -1, "If positive, create and run a corresponding service on this port, only used with 'run'")
	authConfig   = flag.String("auth", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.  If missing, prompt the user.  Only used if doing https.")
	json         = flag.Bool("json", false, "If true, print raw JSON for responses")
	yaml         = flag.Bool("yaml", false, "If true, print raw YAML for responses")
)

func usage() {
	log.Fatal("Usage: cloudcfg -h <host> [-c config/file.json] [-p <hostPort>:<containerPort>,..., <hostPort-n>:<containerPort-n> <method> <path>")
}

// CloudCfg command line tool.
func main() {
	flag.Parse() // Scan the arguments list

	if *versionFlag {
		fmt.Println("Version:", APP_VERSION)
		os.Exit(0)
	}

	if len(flag.Args()) < 2 {
		usage()
	}
	method := flag.Arg(0)
	secure := true
	parsedUrl, err := url.Parse(*httpServer)
	if err != nil {
		log.Fatalf("Unable to parse %v as a URL\n", err)
	}
	if parsedUrl.Scheme != "" && parsedUrl.Scheme != "https" {
		secure = false
	}
	url := *httpServer + path.Join("/api/v1beta1", flag.Arg(1))
	var request *http.Request

	var printer cloudcfg.ResourcePrinter
	if *json {
		printer = &cloudcfg.IdentityPrinter{}
	} else if *yaml {
		printer = &cloudcfg.YAMLPrinter{}
	} else {
		printer = &cloudcfg.HumanReadablePrinter{}
	}

	var auth *kube_client.AuthInfo
	if secure {
		auth, err = cloudcfg.LoadAuthInfo(*authConfig)
		if err != nil {
			log.Fatalf("Error loading auth: %#v", err)
		}
	}

	switch method {
	case "get", "list":
		if len(*labelQuery) > 0 && method == "list" {
			url = url + "?labels=" + *labelQuery
		}
		request, err = http.NewRequest("GET", url, nil)
	case "delete":
		request, err = http.NewRequest("DELETE", url, nil)
	case "create":
		request, err = cloudcfg.RequestWithBody(*config, url, "POST")
	case "update":
		request, err = cloudcfg.RequestWithBody(*config, url, "PUT")
	case "rollingupdate":
		client := &kube_client.Client{
			Host: *httpServer,
			Auth: auth,
		}
		cloudcfg.Update(flag.Arg(1), client, *updatePeriod)
	case "run":
		args := flag.Args()
		if len(args) < 4 {
			log.Fatal("usage: cloudcfg -h <host> run <image> <replicas> <name>")
		}
		image := args[1]
		replicas, err := strconv.Atoi(args[2])
		name := args[3]
		if err != nil {
			log.Fatalf("Error parsing replicas: %#v", err)
		}
		err = cloudcfg.RunController(image, name, replicas, kube_client.Client{Host: *httpServer, Auth: auth}, *portSpec, *servicePort)
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	case "stop":
		err = cloudcfg.StopController(flag.Arg(1), kube_client.Client{Host: *httpServer, Auth: auth})
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	case "rm":
		err = cloudcfg.DeleteController(flag.Arg(1), kube_client.Client{Host: *httpServer, Auth: auth})
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	default:
		log.Fatalf("Unknown command: %s", method)
	}
	if err != nil {
		log.Fatalf("Error: %#v", err)
	}
	body, err := cloudcfg.DoRequest(request, auth)
	if err != nil {
		log.Fatalf("Error: %#v", err)
	}
	err = printer.Print(body, os.Stdout)
	if err != nil {
		log.Fatalf("Failed to print: %#v", err)
	}
	fmt.Print("\n")
}
