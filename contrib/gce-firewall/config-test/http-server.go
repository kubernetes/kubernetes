/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"flag"
	"fmt"
	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud/compute/metadata"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

type runner struct {
	port                  int
	privateIP             string
	publicIP              string
	forwardedIP           string
	instanceName          string
	projectID             string
	region                string
	firewallName          string
	forwardingRuleName    string
	targetPoolName        string
	firewallCreated       bool
	forwardingRuleCreated bool
	targetPoolCreated     bool
	gCompute              *compute.Service
}

type logger func(log string)

func main() {
	flagPort := flag.Int("port", 8000, "the port to listen on.")
	flag.Parse()

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	randomLabel := r.Int31()
	runnerPtr := &runner{
		port:               *flagPort,
		firewallName:       fmt.Sprintf("kubetest-firewall-%d", randomLabel),
		forwardingRuleName: fmt.Sprintf("kubetest-forward-%d", randomLabel),
		targetPoolName:     fmt.Sprintf("kubetest-targetpool-%d", randomLabel),
	}

	var setupLog bytes.Buffer
	setupErr := runnerPtr.setup(getBufferLogger(setupLog))

	http.HandleFunc("/verify-setup", func(w http.ResponseWriter, r *http.Request) {
		w.Write(setupLog.Bytes())
		if setupErr != nil {
			w.WriteHeader(http.StatusPreconditionFailed)
			w.Write([]byte("FAILURE: setup failed.\n"))
			return
		} else {
			w.Write([]byte("SUCCESS: setup succeeded.\n"))
		}
	})

	http.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("SUCCESS: echo succeeded.\n"))
	})

	http.HandleFunc("/open-firewall", func(w http.ResponseWriter, r *http.Request) {
		err := runnerPtr.openFirewall(getHttpLogger(w))
		if err != nil {
			w.WriteHeader(http.StatusExpectationFailed)
			w.Write([]byte("FAILURE: Firewall could not be set."))
		} else {
			w.Write([]byte("SUCCESS: Firewall was set."))
		}
	})

	http.HandleFunc("/forwarding-ip", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, runnerPtr.forwardedIP)
	})

	http.HandleFunc("/public-ip", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, runnerPtr.publicIP)
	})

	http.HandleFunc("/internal-ip", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, runnerPtr.privateIP)
	})

	http.HandleFunc("/teardown", func(w http.ResponseWriter, r *http.Request) {
		err := runnerPtr.teardown(getHttpLogger(w))
		if err != nil {
			w.WriteHeader(http.StatusExpectationFailed)
			w.Write([]byte("FAILURE: Teardown failed."))
		} else {
			w.Write([]byte("SUCCESS: Teardown successful."))
		}
	})

	glog.V(0).Infof(fmt.Sprintf("Listening on port:%d", runnerPtr.port))
	err := http.ListenAndServe(fmt.Sprintf(":%d", runnerPtr.port), nil)
	if err != nil {
		panic("Error:" + err.Error())
	}
}

func getHttpLogger(w http.ResponseWriter) logger {
	return func(log string) {
		glog.V(4).Infof(log)
		w.Write([]byte(log))
		w.Write([]byte("\n"))
	}
}

func getBufferLogger(buffer bytes.Buffer) logger {
	return func(log string) {
		glog.V(4).Infof(log)
		buffer.Write([]byte(log))
		buffer.Write([]byte("\n"))
	}
}

func (runnerPtr *runner) setup(log logger) error {
	privateIP, err := getPrivateIP()
	if err != nil {
		log(fmt.Sprintf("Error getting PrivateIP: %v", err))
		return err
	}
	log(fmt.Sprintf("Got PrivateIP: %s", privateIP))
	runnerPtr.privateIP = privateIP

	projectID, region, err := getProjectAndRegion()
	if err != nil {
		log(fmt.Sprintf("Error getting ProjectID and region: %v", err))
		return err
	}
	log(fmt.Sprintf("Got ProjectID:%s and region: %s", projectID, region))
	runnerPtr.projectID = projectID
	runnerPtr.region = region

	publicIP, err := getPublicIP()
	if err != nil {
		log(fmt.Sprintf("Error getting PublicIP: %v", err))
		return err
	}
	log(fmt.Sprintf("Got PublicIP: %s", publicIP))
	runnerPtr.publicIP = publicIP

	instanceName, err := getInstanceName()
	if err != nil {
		log(fmt.Sprintf("Error getting instanceName: %v", err))
		return err
	}
	runnerPtr.instanceName = instanceName
	log(fmt.Sprintf("Got instanceName:%s", instanceName))

	tokenSource := google.ComputeTokenSource("")
	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	gCompute, err := compute.New(client)
	if err != nil {
		log(fmt.Sprintf("Error getting Google Compute client: %v", err))
		return err
	}
	runnerPtr.gCompute = gCompute

	fwdIP, err := runnerPtr.createForwardingRule(log)
	if err != nil {
		log(fmt.Sprintf("Error setting up the forwarding rule: %v", err))
		return err
	}
	runnerPtr.forwardedIP = fwdIP
	log(fmt.Sprintf("Created ForwardedIP:%s", fwdIP))
	return nil
}

func (runnerPtr *runner) teardown(log logger) error {
	var err error
	err = nil
	if runnerPtr.firewallCreated {
		_, err1 := runnerPtr.gCompute.Firewalls.Delete(runnerPtr.projectID, runnerPtr.firewallName).Do()
		if err1 != nil {
			log(fmt.Sprintf("teardown: firewall %s could not be deleted. err:%v", runnerPtr.firewallName, err1))
			err = err1
		}
	}
	if runnerPtr.forwardingRuleCreated {
		_, err2 := runnerPtr.gCompute.ForwardingRules.Delete(runnerPtr.projectID, runnerPtr.region, runnerPtr.forwardingRuleName).Do()
		if err2 != nil {
			log(fmt.Sprintf("teardown: forwardingRule %s could not be deleted. err:%v", runnerPtr.forwardingRuleName, err2))
			err = err2
		}
	}
	if runnerPtr.targetPoolCreated {
		_, err3 := runnerPtr.gCompute.TargetPools.Delete(runnerPtr.projectID, runnerPtr.region, runnerPtr.targetPoolName).Do()
		if err3 != nil {
			log(fmt.Sprintf("teardown: targetPool %s could not be deleted. err:%v", runnerPtr.targetPoolName, err3))
			err = err3
		}
	}
	return err
}

func (runner *runner) closeFirewall() error {
	_, err := runner.gCompute.Firewalls.Delete(runner.projectID, runner.firewallName).Do()
	return err
}

func (runner *runner) openFirewall(log logger) error {
	allowed := make([]*compute.FirewallAllowed, 1)
	allowed[0] = &compute.FirewallAllowed{
		IPProtocol: "tcp",
		Ports:      []string{fmt.Sprintf("%d", runner.port)},
	}
	firewall := &compute.Firewall{
		Name:         runner.firewallName,
		SourceRanges: []string{"0.0.0.0/0"},
		Description:  fmt.Sprintf("KubernetesAutoGenerated_OnlyAllowTrafficForDestinationIP_%s", runner.forwardedIP),
		Allowed:      allowed,
	}
	_, err := runner.gCompute.Firewalls.Insert(runner.projectID, firewall).Do()
	if err != nil {
		log(fmt.Sprintf("Could not create firewall rule:%s, Error: %v", runner.firewallName, err))
		return err
	}
	for i := 0; i < 10; i++ {
		_, err := runner.gCompute.Firewalls.Get(runner.projectID, runner.firewallName).Do()
		if err != nil {
			log(fmt.Sprintf("Could not get firewall rule:%s, Error: %v", runner.firewallName, err))
		} else {
			log(fmt.Sprintf("Firewall created:%s", runner.firewallName))
			runner.firewallCreated = true
			return nil
		}
		time.Sleep(time.Second * 2)
	}
	return err
}

func getPublicIP() (string, error) {
	return metadata.Get("instance/network-interfaces/0/access-configs/0/external-ip")
}

func (runner *runner) createForwardingRule(log logger) (string, error) {
	targetPool := &compute.TargetPool{
		Name:      runner.targetPoolName,
		Instances: []string{fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/zones/us-central1-b/instances/%s", runner.instanceName)},
	}

	_, err := runner.gCompute.TargetPools.Insert(runner.projectID, runner.region, targetPool).Do()
	if err != nil {
		log(fmt.Sprintf("Could not create target pool. Error: %v", err))
		return "", err
	}
	runner.targetPoolCreated = true
	log(fmt.Sprintf("Created targetpool: %s", runner.targetPoolName))

	forwardingRule := &compute.ForwardingRule{
		Name:       runner.forwardingRuleName,
		IPProtocol: "TCP",
		PortRange:  fmt.Sprintf("%d", runner.port),
		Region:     runner.region,
		Target:     fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/regions/us-central1/targetPools/%s", runner.targetPoolName),
	}
	_, err = runner.gCompute.ForwardingRules.Insert(runner.projectID, runner.region, forwardingRule).Do()
	if err != nil {
		log(fmt.Sprintf("Could not create forwarding rule. Error: %v", err))
		return "", err
	}
	log(fmt.Sprintf("Creating forwarding rule: %s", runner.forwardingRuleName))
	for i := 0; i < 20; i++ {
		createdForwardingRule, err := runner.gCompute.ForwardingRules.Get(runner.projectID, runner.region, runner.forwardingRuleName).Do()
		if err != nil {
			log(fmt.Sprintf("Could not get forwarding rule:%s, Error: %v", runner.forwardingRuleName, err))
		} else {
			runner.forwardingRuleCreated = true
			return createdForwardingRule.IPAddress, nil
		}
		time.Sleep(time.Second * 2)
	}
	return "", err
}

func getPrivateIP() (string, error) {
	return metadata.Get("instance/network-interfaces/0/ip")
}

func getInstanceName() (string, error) {
	hostname, err := metadata.Get("instance/hostname")
	if err != nil {
		return "", err
	}
	segments := strings.SplitN(hostname, ".", 2)
	return segments[0], nil
}

func getProjectAndRegion() (string, string, error) {
	result, err := metadata.Get("instance/zone")
	if err != nil {
		return "", "", err
	}
	parts := strings.Split(result, "/")
	if len(parts) != 4 {
		return "", "", fmt.Errorf("unexpected response: %s", result)
	}
	zone := parts[3]
	projectID, err := metadata.ProjectID()
	if err != nil {
		return "", "", err
	}
	region := strings.Split(zone, "-")
	if len(region) != 3 {
		return "", "", fmt.Errorf("unexpected response: %s. Unknown region.", result)
	}
	return projectID, strings.Join([]string{region[0], region[1]}, "-"), nil
}
