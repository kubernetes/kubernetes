/*
Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package filter

import (
	"context"
	"fmt"
	"os"
	"testing"

	"golang.org/x/oauth2/google"
	"google.golang.org/api/compute/v1"
)

var (
	projectID = os.Getenv("PROJECT_ID")
	// test requires e2e cluster to be present in the project specified in order to match these filters
	routeFilters   = map[string]interface{}{"name": "e2e-test-.*", "description": "k8s-node-route", "priority": 1000}
	networkFilters = map[string]interface{}{"autoCreateSubnetworks": false}
)

func TestFilterResources(t *testing.T) {
	t.Parallel()
	if projectID == "" {
		t.Skip("Missing projectID... skipping test")
	}

	// See https://cloud.google.com/docs/authentication/.
	// Use GOOGLE_APPLICATION_CREDENTIALS environment variable to specify
	// a service account key file to authenticate to the API.
	hc, err := google.DefaultClient(context.Background(), compute.ComputeScope)
	if err != nil {
		t.Errorf("Could not get authenticated client: %v", err)
	}

	svc, err := compute.New(hc)
	if err != nil {
		t.Errorf("Could not initialize compute client: %v", err)
	}

	if err := listRoutes(svc, projectID, t); err != nil {
		t.Errorf("Failed to rist routes - %v", err)
	}
	if err = listNetworks(svc, projectID, t); err != nil {
		t.Errorf("Failed to rist routes - %v", err)
	}
}

func constructFilter(params map[string]interface{}) string {
	var fl *F
	for key, val := range params {
		switch val.(type) {
		case string:
			if fl == nil {
				fl = Regexp(key, val.(string))
			} else {
				fl.AndRegexp(key, val.(string))
			}
		case bool:
			if fl == nil {
				fl = EqualBool(key, val.(bool))
			} else {
				fl.AndEqualBool(key, val.(bool))
			}
		case int:
			if fl == nil {
				fl = EqualInt(key, val.(int))
			} else {
				fl.AndEqualInt(key, val.(int))
			}
		}
	}
	if fl == nil {
		return ""
	}
	return fl.String()
}

func listRoutes(svc *compute.Service, projectID string, t *testing.T) error {
	fstr := constructFilter(routeFilters)
	list, err := svc.Routes.List(projectID).Filter(fstr).Do()
	if err != nil {
		return fmt.Errorf("failed to list routes: %v", err)
	}
	t.Logf("Got %d matching routes matching filter '%s':", len(list.Items), fstr)
	for ix, v := range list.Items {
		t.Logf("%d. %s", ix, v.Name)
	}
	return nil
}

func listNetworks(svc *compute.Service, projectID string, t *testing.T) error {
	fstr := constructFilter(networkFilters)
	list, err := svc.Networks.List(projectID).Filter(fstr).Do()
	if err != nil {
		return fmt.Errorf("failed to list networks: %v", err)
	}
	t.Logf("Got %d matching networks matching filter '%s':", len(list.Items), fstr)
	for ix, v := range list.Items {
		t.Logf("%d. %s", ix, v.Name)
	}
	return nil
}
