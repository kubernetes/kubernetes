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
	"crypto/tls"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	compute "code.google.com/p/google-api-go-client/compute/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kube_updown"
	"github.com/golang/glog"
)

const (
	zone       = "us-central1-b"
	numMinions = 4
)

var (
	project = flag.String("project", defaultProject(), "Default GCE Project.")
)

func defaultProject() string {
	if proj := projectFromEnv(); proj != "" {
		return proj
	}
	if proj := projectFromGcloud(); proj != "" {
		return proj
	}
	return ""
}

func projectFromEnv() string {
	return os.Getenv("PROJECT")
}

func projectFromGcloud() string {
	cmd := exec.Command("gcloud", "config", "list", "project")
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return ""
	}
	if err = cmd.Start(); err != nil {
		return ""
	}
	contents, err := ioutil.ReadAll(outPipe)
	if err != nil {
		return ""
	}
	if err = cmd.Wait(); err != nil {
		return ""
	}
	prjReg, err := regexp.Compile(`project = (.*)\n`)
	if err != nil {
		return ""
	}
	project := prjReg.FindStringSubmatch(string(contents))
	// element 1 is the submatch
	if project == nil {
		return ""
	}
	return project[1]
}

func main() {
	flag.Parse()
	fmt.Printf("Project: %v\n", *project)

	c := kube_updown.GetOAuthClient()
	svc, err := compute.New(c)
	if err != nil {
		glog.Fatalf("couldn't create compute api client: %v", err)
	}

	fmt.Printf("Starting VMs and configuring firewalls\n")
	zoneOps := make([]*compute.Operation, 1+numMinions)
	globalOps := make([]*compute.Operation, 1+2*numMinions)

	fmt.Printf("Creating firewall kubernetes-master-https\n")
	globalOps[0], err = kube_updown.AddMasterFirewall(svc, *project)
	if err != nil {
		glog.Fatalf("couldn't start operation: %v", err)
	}
	fmt.Printf("Creating instance kubernetes-master\n")
	zoneOps[0], err = kube_updown.AddMaster(svc, *project, zone)
	if err != nil {
		glog.Fatalf("couldn't start operation: %v", err)
	}

	for i := 1; i <= numMinions; i++ {
		fmt.Printf("Creating firewall kubernetes-minion-%v-all\n", i)
		globalOps[i], err = kube_updown.AddMinionFirewall(svc, *project, i)
		if err != nil {
			glog.Fatalf("couldn't create firewall-rule insert operation: %v", err)
		}
		fmt.Printf("Creating instance kubernetes-minion-%v\n", i)
		zoneOps[i], err = kube_updown.AddMinion(svc, *project, zone, i)
		if err != nil {
			glog.Fatalf("couldn't create instance insert operation: %v", err)
		}
		fmt.Printf("Creating route kubernetes-minion-%v\n", i)
		globalOps[numMinions+i], err = kube_updown.AddMinionRoute(svc, *project, zone, i)
		if err != nil {
			glog.Fatalf("couldn't create route insert operation: %v", err)
		}
	}
	// Wait for all operations to complete
	for _, op := range zoneOps {
		target, resource := targetInfo(op.TargetLink)
		op, err = svc.ZoneOperations.Get(*project, zone, op.Name).Do()
		if err != nil {
			glog.Fatalf("error getting operation: %v\n", err)
		}
		for op.Status != "DONE" {
			fmt.Printf("Waiting 2s for %v of %v %v\n", op.OperationType, resource, target)
			time.Sleep(2 * time.Second)
			op, err = svc.ZoneOperations.Get(*project, zone, op.Name).Do()
			if err != nil {
				glog.Fatalf("error getting operation: %v\n", err)
			}
		}
		if op.Error != nil {
			glog.Errorf("errors in operation %v:\n", op.Name)
			for _, err := range op.Error.Errors {
				glog.Errorf(err.Message)
			}
		}
		fmt.Printf("%v of %v %v has completed\n", op.OperationType, resource, target)
	}
	for _, op := range globalOps {
		target, resource := targetInfo(op.TargetLink)
		op, err = svc.GlobalOperations.Get(*project, op.Name).Do()
		if err != nil {
			glog.Fatalf("error getting operation: %v\n", err)
		}
		for op.Status != "DONE" {
			fmt.Printf("Waiting 2s for %v of %v %v\n", op.OperationType, resource, target)
			time.Sleep(2 * time.Second)
			if err != nil {
				glog.Fatalf("error getting operation: %v\n", err)
			}
		}
		if op.Error != nil {
			glog.Errorf("errors in operation %v:\n", op.Name)
			for _, err := range op.Error.Errors {
				glog.Errorf(err.Message)
			}
		}
		fmt.Printf("%v of %v %v has completed\n", op.OperationType, resource, target)
	}

	// Attempt to contact kube-master until it responds
	kubeMasterIP, err := kube_updown.GetMasterIP(svc, *project, zone)
	if err != nil {
		glog.Fatalf("error getting IP: %v\n", err)
	}
	fmt.Printf("Using master: %v (external IP: %v)\n", kube_updown.MasterName, kubeMasterIP)
	url := fmt.Sprintf("https://%v/api/v1beta1/pods", kubeMasterIP)
	usr, pass := kube_updown.GetCredentials()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		glog.Fatalf("error creating request: %v\n", err)
	}
	req.SetBasicAuth(usr, pass)
	tr := &http.Transport{
		ResponseHeaderTimeout: 5 * time.Second,
		TLSClientConfig:       &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	fmt.Printf("Waiting for cluster initialization.\n\n")
	fmt.Printf("  This will continually check to see if the API for kubernetes is reachable.\n")
	fmt.Printf("  This might loop forever if there was some uncaught error during start up.\n\n")
	for {
		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf(".")
		} else {
			//fmt.Printf("\n\nStatus: %v\n\n", resp.Status)
			if resp.StatusCode != 200 {
				fmt.Printf("Response status was %v. Something might be wrong.\n", resp.StatusCode)
			}
			break
		}
		time.Sleep(2 * time.Second)
	}
	fmt.Printf("\nKubernetes cluster created.\n")
	fmt.Printf("Sanity checking cluster...\n")

	// Check each minion for successful installation of docker
	for i := 1; i <= numMinions; i++ {
		name := fmt.Sprintf("%v-%v", kube_updown.MinionPrefix, i)
		err = exec.Command("gcutil", "ssh", name, "which", "docker").Run()
		if err != nil {
			fmt.Printf("Docker failed to install on %v. You're cluster is unlikely to work correctly.\n", name)
			fmt.Printf("Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)\n")
			os.Exit(1)
		}
	}
	fmt.Print("Kubernetes cluster is running.  Access the master at:\n\n")
	fmt.Printf("  https://%v:%v@%v\n\n", usr, pass, kubeMasterIP)
	fmt.Print("Security note: The server above uses a self signed certificate.  This is\n")
	fmt.Print("    subject to \"Man in the middle\" type attacks.\n")
}

func targetInfo(targetLink string) (target, resourceType string) {
	i := strings.LastIndex(targetLink, "/")
	target = targetLink[i+1:]
	j := strings.LastIndex(targetLink[:i], "/")
	resourceType = targetLink[j+1 : i-1]
	return target, resourceType
}
