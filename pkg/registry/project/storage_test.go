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

package project

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestListProjectsError(t *testing.T) {
	mockRegistry := registrytest.ProjectRegistry{
		Err: fmt.Errorf("test error"),
	}
	storage := RegistryStorage{
		registry: &mockRegistry,
	}
	projectsObj, err := storage.List(nil)
	projects := projectsObj.(api.ProjectList)
	if err != mockRegistry.Err {
		t.Errorf("Expected %#v, Got %#v", mockRegistry.Err, err)
	}
	if len(projects.Items) != 0 {
		t.Errorf("Unexpected non-zero project list: %#v", projects)
	}
}

func TestListEmptyProjectList(t *testing.T) {
	mockRegistry := registrytest.ProjectRegistry{}
	storage := RegistryStorage{
		registry: &mockRegistry,
	}
	projects, err := storage.List(labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(projects.(api.ProjectList).Items) != 0 {
		t.Errorf("Unexpected non-zero project list: %#v", projects)
	}
}

func TestListProjectList(t *testing.T) {
	mockRegistry := registrytest.ProjectRegistry{
		Projects: []api.Project{
			{
				JSONBase: api.JSONBase{
					ID: "foo",
				},
			},
			{
				JSONBase: api.JSONBase{
					ID: "bar",
				},
			},
		},
	}
	storage := RegistryStorage{
		registry: &mockRegistry,
	}
	projectsObj, err := storage.List(labels.Everything())
	projects := projectsObj.(api.ProjectList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(projects.Items) != 2 {
		t.Errorf("Unexpected project list: %#v", projects)
	}
	if projects.Items[0].ID != "foo" {
		t.Errorf("Unexpected project: %#v", projects.Items[0])
	}
	if projects.Items[1].ID != "bar" {
		t.Errorf("Unexpected project: %#v", projects.Items[1])
	}
}

func TestProjectDecode(t *testing.T) {
	mockRegistry := registrytest.ProjectRegistry{}
	storage := RegistryStorage{
		registry: &mockRegistry,
	}
	project := &api.Project{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	}
	body, err := api.Encode(project)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	projectOut := storage.New()
	if err := api.DecodeInto(body, projectOut); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(project, projectOut) {
		t.Errorf("Expected %#v, found %#v", project, projectOut)
	}
}

func TestProjectParsing(t *testing.T) {
	expectedProject := api.Project{
		JSONBase: api.JSONBase{
			ID: "myproject",
		},
		Labels: map[string]string{
			"name": "myproject",
		},
	}
	file, err := ioutil.TempFile("", "project")
	fileName := file.Name()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	data, err := json.Marshal(expectedProject)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	_, err = file.Write(data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = file.Close()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	data, err = ioutil.ReadFile(fileName)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var project api.Project
	err = json.Unmarshal(data, &project)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(project, expectedProject) {
		t.Errorf("Parsing failed: %s %#v %#v", string(data), project, expectedProject)
	}
}

func TestCreateProject(t *testing.T) {
	mockRegistry := registrytest.ProjectRegistry{}
	storage := RegistryStorage{
		registry: &mockRegistry,
	}
	project := &api.Project{
		JSONBase: api.JSONBase{ID: "test"},
		Labels: map[string]string{
			"name": "myprojectlabel",
		},
	}
	channel, err := storage.Create(project)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-channel:
		// expected case
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout from async channel")
	}
}
