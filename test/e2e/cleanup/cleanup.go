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

package main

import (
	"log"
	"os"
	"strings"

	flag "github.com/spf13/pflag"

	"k8s.io/kubernetes/test/e2e"
)

// This is a script to help cleanup external resources. Currently it only
// supports Ingress.
var (
	flags    = flag.NewFlagSet("A script to purge resources from a live cluster/project etc", flag.ContinueOnError)
	resource = flags.String("resource", "", "name of the resource to cleanup, eg: ingress")
	project  = flags.String("project", "", "name of the project, eg: argument of gcloud --project")
)

func main() {
	flags.Parse(os.Args)
	if *resource == "" || *project == "" {
		log.Fatalf("Please specify a resource and project to cleanup.")
	}
	if strings.ToLower(*resource) == "ingress" {
		ingController := e2e.GCEIngressController{UID: ".*", Project: *project}
		log.Printf("%v", ingController.Cleanup(true))
		return
	}
	log.Fatalf("Unknown resource %v", *resource)
}
