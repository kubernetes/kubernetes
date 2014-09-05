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
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kube_updown"
	"github.com/golang/glog"
)

const (
	zone       = "us-central1-b"
	numMinions = 4
)

var project = flag.String("project", defaultProject(), "Default GCE Project.")

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
	if project == nil {
		return ""
	}
	return project[1]
}

func main() {
	flag.Set("stderrthreshold", "INFO")
	flag.Parse()
	cloud, err := gce_cloud.CreateGCECloud(*project, zone)
	if err != nil {
		glog.Fatalf("failed to create cloud: %v", err)
	}
	glog.Info("Starting VMs and configuring firewalls\n")
	ops, err := deployMaster(cloud)
	if err != nil {
		glog.Fatal(err)
	}

	for i := 1; i <= numMinions; i++ {
		newOps, err := deployMinion(cloud, i)
		if err != nil {
			glog.Fatal(err)
		}
		ops = append(ops, newOps...)
	}
	if err := waitForOps(cloud, ops); err != nil {
		glog.Fatal(err)
	}
	if err := checkMaster(cloud); err != nil {
		glog.Fatal(err)
	}
	if err := checkMinions(); err != nil {
		glog.Fatal(err)
	}
	glog.Info("Kubernetes cluster is running\n")
	glog.Info("Security note: The server above uses a self signed certificate.  This is\n")
	glog.Info("    subject to \"Man in the middle\" type attacks.\n")
}

func deployMaster(cloud *gce_cloud.GCECloud) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Info("Creating firewall kubernetes-master-https\n")
	op, err := kube_updown.CreateMasterFirewall(cloud)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	glog.Info("Creating instance kubernetes-master\n")
	op, err = kube_updown.CreateMaster(cloud, *project)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

func deployMinion(cloud *gce_cloud.GCECloud, i int) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Infof("Creating firewall kubernetes-minion-%v-all\n", i)
	op, err := kube_updown.CreateMinionFirewall(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create firewall-rule insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Creating instance kubernetes-minion-%v\n", i)
	op, err = kube_updown.CreateMinion(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create instance insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Creating route kubernetes-minion-%v\n", i)
	op, err = kube_updown.CreateMinionRoute(cloud, *project, zone, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create route insert operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

func waitForOps(cloud *gce_cloud.GCECloud, ops []*gce_cloud.GCEOp) error {
	// Wait for all operations to complete
	for _, op := range ops {
		op, err := cloud.PollOp(op)
		if err != nil {
			return err
		}
		for op.Status() != "DONE" {
			glog.Infof("Waiting 2s for %v of %v %v\n", op.OperationType(), op.Resource(), op.Target())
			time.Sleep(2 * time.Second)
			op, err = cloud.PollOp(op)
			if err != nil {
				return err
			}
		}
		if op.Errors() != nil {
			return errors.New("errors in operation:\n" + strings.Join(op.Errors(), "\n"))
		}
		glog.Infof("%v of %v %v has completed\n", op.OperationType(), op.Resource(), op.Target())
	}
	return nil
}

func checkMaster(cloud *gce_cloud.GCECloud) error {
	// Attempt to contact kube-master until it responds
	kubeMasterIP, err := cloud.IPAddress(kube_updown.MasterName)
	if err != nil {
		return fmt.Errorf("error getting master IP: %v\n", err)
	}
	glog.Infof("Using master: %v (external IP: %v)\n", kube_updown.MasterName, kubeMasterIP)
	url := fmt.Sprintf("https://%v/api/v1beta1/pods", kubeMasterIP)
	usr, pass := kube_updown.GetCredentials()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.SetBasicAuth(usr, pass)
	tr := &http.Transport{
		ResponseHeaderTimeout: 5 * time.Second,
		TLSClientConfig:       &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	glog.Info("Waiting for cluster initialization.\n")
	glog.Info("  This will continually check to see if the API for kubernetes is reachable.\n")
	glog.Info("  This might loop forever if there was some uncaught error during start up.\n")
	t := 0
	for {
		resp, err := client.Do(req)
		if err == nil {
			if resp.StatusCode != 200 {
				glog.Infof("\nResponse status was %v. Something might be wrong.\n", resp.Status)
			}
			break
		}
		time.Sleep(2 * time.Second)
		t = t + 2
		if t%10 == 0 {
			glog.Infof("%v seconds elapsed", t)
		}
	}
	glog.Info("Kubernetes master is running.  Access at:\n")
	glog.Infof("  https://%v:%v@%v\n", usr, pass, kubeMasterIP)
	return nil
}

func checkMinions() error {
	glog.Info("Sanity checking minions...\n")
	// Check each minion for successful installation of docker
	for i := 1; i <= numMinions; i++ {
		name := fmt.Sprintf("%v-%v", kube_updown.MinionPrefix, i)
		if err := exec.Command("gcutil", "ssh", name, "which", "docker").Run(); err != nil {
			return fmt.Errorf("Docker failed to install on %v. You're cluster is unlikely to work correctly.\n"+
				"Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)\n", name)

		}
	}
	return nil
}
