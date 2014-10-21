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
	"io/ioutil"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubecfg"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	authConfig = flag.String("auth_config", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.")
	host       = flag.String("host", "", "The host to connect to")
)

func waitForPodRunning(c *client.Client, id string) {
	for {
		ctx := api.NewContext()
		time.Sleep(5 * time.Second)
		pod, err := c.GetPod(ctx, id)
		if err != nil {
			glog.Warningf("Get pod failed: %v", err)
			continue
		}
		if pod.CurrentState.Status == api.PodRunning {
			break
		}
		glog.Infof("Waiting for pod status to be running (%s)", pod.CurrentState.Status)
	}
}

func loadObjectOrDie(filePath string) interface{} {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		glog.Fatalf("Failed to read pod: %v", err)
	}
	obj, err := latest.Codec.Decode(data)
	if err != nil {
		glog.Fatalf("Failed to decode pod: %v", err)
	}
	return obj
}

func loadPodOrDie(filePath string) *api.Pod {
	obj := loadObjectOrDie(filePath)
	pod, ok := obj.(*api.Pod)
	if !ok {
		glog.Fatalf("Failed to load pod: %v", obj)
	}
	return pod
}

func loadClientOrDie() *client.Client {
	config := client.Config{
		Host: *host,
	}
	auth, err := kubecfg.LoadAuthInfo(*authConfig, os.Stdin)
	if err != nil {
		glog.Fatalf("Error loading auth: %v", err)
	}
	config.Username = auth.User
	config.Password = auth.Password
	config.CAFile = auth.CAFile
	config.CertFile = auth.CertFile
	config.KeyFile = auth.KeyFile
	config.BearerToken = auth.BearerToken
	if auth.Insecure != nil {
		config.Insecure = *auth.Insecure
	}
	c, err := client.New(&config)
	if err != nil {
		glog.Fatalf("Error creating client")
	}
	return c
}

func TestPodUpdate(c *client.Client) bool {
	ctx := api.NewContext()

	pod := loadPodOrDie("./api/examples/pod.json")
	value := strconv.Itoa(time.Now().Nanosecond())
	pod.Labels["time"] = value

	_, err := c.CreatePod(ctx, pod)
	if err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	defer c.DeletePod(ctx, pod.ID)
	waitForPodRunning(c, pod.ID)
	pods, err := c.ListPods(ctx, labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
	if len(pods.Items) != 1 {
		glog.Errorf("Failed to find the correct pod")
		return false
	}

	podOut, err := c.GetPod(ctx, pod.ID)
	if err != nil {
		glog.Errorf("Failed to get pod: %v", err)
		return false
	}
	value = "time" + value
	pod.Labels["time"] = value
	pod.ResourceVersion = podOut.ResourceVersion
	pod.DesiredState.Manifest.UUID = podOut.DesiredState.Manifest.UUID
	pod, err = c.UpdatePod(ctx, pod)
	if err != nil {
		glog.Errorf("Failed to update pod: %v", err)
		return false
	}
	waitForPodRunning(c, pod.ID)
	pods, err = c.ListPods(ctx, labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
	if len(pods.Items) != 1 {
		glog.Errorf("Failed to find the correct pod after update.")
		return false
	}
	glog.Infof("pod update OK")
	return true
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(3 * time.Minute)
		glog.Fatalf("This test has timed out.")
	}()

	c := loadClientOrDie()

	tests := []func(c *client.Client) bool{
		TestPodUpdate,
	}

	passed := true
	for _, test := range tests {
		testPassed := test(c)
		if !testPassed {
			passed = false
		}
	}
	if !passed {
		glog.Fatalf("Tests failed")
	}
}
