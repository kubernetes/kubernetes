// +build integration,!no-etcd

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

package integration

// This file tests use of the Job API resource.

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestJob tests apiserver-side behavior of creation of Jobs.
func TestJob(t *testing.T) {
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	// TODO: Uncomment when fix #19254
	// defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	framework.DeleteAllEtcdKeys()
	client := client.NewOrDie(&client.Config{Host: s.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	DoTestJobMultipleVersions(t, client)
	// TODO: also test defaulting of selectors and selector validation.
}

func DoTestJobMultipleVersions(t *testing.T, client *client.Client) {
	ns := "ns"

	// Create job via extensions client, and get it back 
	cfg := extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "j",
			Namespace: ns,
		},
		Spec: extensions.JobSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{ "foo": "bar" },
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name: "j",
					Labels: map[string]string{ "foo": "bar" },
				},
				Spec: api.PodSpec{
					RestartPolicy: "OnFailure",
					Containers: []api.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			},
		},
	}

	if _, err := client.Extensions().Jobs(cfg.Namespace).Create(&cfg); err != nil {
		t.Errorf("unable to create test job: %v", err)
	}
	defer deleteJobOrErrorf(t, client, cfg.Namespace, cfg.Name)

	if _, err := client.Extensions().Jobs(cfg.Namespace).Get("j"); err != nil {
		t.Errorf("unable to get test job: %v", err)
	}

	if _, err := client.Extensions().Jobs(cfg.Namespace).Get("j"); err != nil {
		t.Errorf("unable to get test job: %v", err)
	}

/*
	if !api.Semantic.DeepDerivative(job, testJob) {
		t.Errorf("Expected %#v, but got %#v", testJob, job)
	}
*/
	if _, err := client.Batch().Jobs(cfg.Namespace).Get("j"); err != nil {
		t.Errorf("unable to get test job: %v", err)
	}
}

func deleteJobOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.Extensions().Jobs(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete Job %v: %v", name, err)
	}
}
