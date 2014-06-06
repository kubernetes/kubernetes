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
	"os"
	"strconv"
	"time"

	kube_client "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudcfg"
)

const APP_VERSION = "0.1"

// The flag package provides a default help printer via -h switch
var versionFlag *bool = flag.Bool("v", false, "Print the version number.")
var httpServer *string = flag.String("h", "", "The host to connect to.")
var config *string = flag.String("c", "", "Path to the config file.")
var labelQuery *string = flag.String("l", "", "Label query to use for listing")
var updatePeriod *time.Duration = flag.Duration("u", 60*time.Second, "Update interarrival in seconds")
var portSpec *string = flag.String("p", "", "The port spec, comma-separated list of <external>:<internal>,...")
var servicePort *int = flag.Int("s", -1, "If positive, create and run a corresponding service on this port, only used with 'run'")
var authConfig *string = flag.String("auth", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.  If missing, prompt the user")

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
	url := *httpServer + "/api/v1beta1" + flag.Arg(1)
	var request *http.Request
	var err error

	auth, err := cloudcfg.LoadAuthInfo(*authConfig)
	if err != nil {
		log.Fatalf("Error loading auth: %#v", err)
	}

	if method == "get" || method == "list" {
		if len(*labelQuery) > 0 && method == "list" {
			url = url + "?labels=" + *labelQuery
		}
		request, err = http.NewRequest("GET", url, nil)
	} else if method == "delete" {
		request, err = http.NewRequest("DELETE", url, nil)
	} else if method == "create" {
		request, err = cloudcfg.RequestWithBody(*config, url, "POST")
	} else if method == "update" {
		request, err = cloudcfg.RequestWithBody(*config, url, "PUT")
	} else if method == "rollingupdate" {
		client := &kube_client.Client{
			Host: *httpServer,
			Auth: &auth,
		}
		cloudcfg.Update(flag.Arg(1), client, *updatePeriod)
	} else if method == "run" {
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
		err = cloudcfg.RunController(image, name, replicas, kube_client.Client{Host: *httpServer, Auth: &auth}, *portSpec, *servicePort)
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	} else if method == "stop" {
		err = cloudcfg.StopController(flag.Arg(1), kube_client.Client{Host: *httpServer, Auth: &auth})
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	} else if method == "rm" {
		err = cloudcfg.DeleteController(flag.Arg(1), kube_client.Client{Host: *httpServer, Auth: &auth})
		if err != nil {
			log.Fatalf("Error: %#v", err)
		}
		return
	} else {
		log.Fatalf("Unknown command: %s", method)
	}
	if err != nil {
		log.Fatalf("Error: %#v", err)
	}
	var body string
	body, err = cloudcfg.DoRequest(request, auth.User, auth.Password)
	if err != nil {
		log.Fatalf("Error: %#v", err)
	}
	fmt.Println(body)
}
