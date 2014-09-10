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
	"os/exec"
	"regexp"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/golang/glog"
)

var (
	numMinions = flag.Int("numMinions", 4, "Number of minions in the cluster.")
	project    = flag.String("project", defaultProject(), "Default GCE Project.")
	zone       = flag.String("zone", "us-central1-b", "Zone where cluster will be launched.")
)

func usage() {
	fmt.Fprint(os.Stderr, "Usage: kube-deploy up|down|push\n")
}

func main() {
	flag.Set("stderrthreshold", "INFO")

	flag.Parse()

	if len(flag.Args()) != 1 {
		usage()
		os.Exit(1)
	}

	switch flag.Arg(0) {
	case "up":
		up()
	case "down":
		down()
	case "push":
		push()
	default:
		usage()
	}
}

func up() {
	cloud, err := gce_cloud.CreateGCECloud(*project, *zone)
	if err != nil {
		glog.Fatalf("failed to create cloud: %v", err)
	}
	deployer, success := cloud.Deployer()
	if !success {
		glog.Fatalf("cloud provider does not support deployment")
	}
	glog.Info("Starting VMs and configuring firewalls\n")
	if err := deployer.CreateCluster(*numMinions); err != nil {
		glog.Fatalf("failed to bring up cluster: %v", err)
	}
	glog.Info("Kubernetes cluster is running\n")
	glog.Info("Security note: The server above uses a self signed certificate.  This is\n")
	glog.Info("    subject to \"Man in the middle\" type attacks.\n")
}

func down() {
	cloud, err := gce_cloud.CreateGCECloud(*project, *zone)
	if err != nil {
		glog.Fatalf("failed to create cloud: %v", err)
	}
	deployer, success := cloud.Deployer()
	if !success {
		glog.Fatalf("cloud provider does not support deployment")
	}
	if err := deployer.DeleteCluster(*numMinions); err != nil {
		glog.Fatalf("failed to down cluster: %v", err)
	}
	glog.Info("Kubernetes cluster successfully taken down.\n")
}

func push() {
	fmt.Println("Not yet implemented!")
}

func defaultProject() string {
	if proj := projectFromEnv(); proj != "" {
		return proj
	}
	if proj := projectFromGCloud(); proj != "" {
		return proj
	}
	return ""
}

func projectFromEnv() string {
	return os.Getenv("PROJECT")
}

func projectFromGCloud() string {
	cmd := exec.Command("gcloud", "config", "list", "project")
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return ""
	}
	if err := cmd.Start(); err != nil {
		return ""
	}
	contents, err := ioutil.ReadAll(outPipe)
	if err != nil {
		return ""
	}
	if err := cmd.Wait(); err != nil {
		return ""
	}
	prjReg, err := regexp.Compile(`project = (.*)\n`)
	if err != nil {
		return ""
	}
	project := prjReg.FindStringSubmatch(string(contents))
	// element 1 is the submatch
	if project == nil || len(project) < 2 {
		return ""
	}
	return project[1]
}
