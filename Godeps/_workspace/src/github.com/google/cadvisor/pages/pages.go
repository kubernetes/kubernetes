// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pages

import (
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"strings"

	httpmux "github.com/google/cadvisor/http/mux"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"

	auth "github.com/abbot/go-http-auth"
	"github.com/golang/glog"
)

var pageTemplate *template.Template

type link struct {
	// Text to show in the link.
	Text string

	// Web address to link to.
	Link string
}

type keyVal struct {
	Key   string
	Value string
}

type pageData struct {
	DisplayName            string
	ContainerName          string
	ParentContainers       []link
	Subcontainers          []link
	Spec                   info.ContainerSpec
	Stats                  []*info.ContainerStats
	MachineInfo            *info.MachineInfo
	IsRoot                 bool
	ResourcesAvailable     bool
	CpuAvailable           bool
	MemoryAvailable        bool
	NetworkAvailable       bool
	FsAvailable            bool
	CustomMetricsAvailable bool
	Root                   string
	DockerStatus           []keyVal
	DockerDriverStatus     []keyVal
	DockerImages           []manager.DockerImage
}

func init() {
	pageTemplate = template.New("containersTemplate").Funcs(funcMap)
	_, err := pageTemplate.Parse(containersHtmlTemplate)
	if err != nil {
		glog.Fatalf("Failed to parse template: %s", err)
	}
}

func containerHandlerNoAuth(containerManager manager.Manager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := serveContainersPage(containerManager, w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	}
}

func containerHandler(containerManager manager.Manager) auth.AuthenticatedHandlerFunc {
	return func(w http.ResponseWriter, r *auth.AuthenticatedRequest) {
		err := serveContainersPage(containerManager, w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	}
}

func dockerHandlerNoAuth(containerManager manager.Manager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := serveDockerPage(containerManager, w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	}
}

func dockerHandler(containerManager manager.Manager) auth.AuthenticatedHandlerFunc {
	return func(w http.ResponseWriter, r *auth.AuthenticatedRequest) {
		err := serveDockerPage(containerManager, w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	}
}

// Register http handlers
func RegisterHandlersDigest(mux httpmux.Mux, containerManager manager.Manager, authenticator *auth.DigestAuth) error {
	// Register the handler for the containers page.
	if authenticator != nil {
		mux.HandleFunc(ContainersPage, authenticator.Wrap(containerHandler(containerManager)))
		mux.HandleFunc(DockerPage, authenticator.Wrap(dockerHandler(containerManager)))
	} else {
		mux.HandleFunc(ContainersPage, containerHandlerNoAuth(containerManager))
		mux.HandleFunc(DockerPage, dockerHandlerNoAuth(containerManager))
	}
	return nil
}

func RegisterHandlersBasic(mux httpmux.Mux, containerManager manager.Manager, authenticator *auth.BasicAuth) error {
	// Register the handler for the containers and docker age.
	if authenticator != nil {
		mux.HandleFunc(ContainersPage, authenticator.Wrap(containerHandler(containerManager)))
		mux.HandleFunc(DockerPage, authenticator.Wrap(dockerHandler(containerManager)))
	} else {
		mux.HandleFunc(ContainersPage, containerHandlerNoAuth(containerManager))
		mux.HandleFunc(DockerPage, dockerHandlerNoAuth(containerManager))
	}
	return nil
}

func getContainerDisplayName(cont info.ContainerReference) string {
	// Pick a user-added alias as display name.
	displayName := ""
	for _, alias := range cont.Aliases {
		// ignore container id as alias.
		if strings.Contains(cont.Name, alias) {
			continue
		}
		// pick shortest display name if multiple aliases are available.
		if displayName == "" || len(displayName) >= len(alias) {
			displayName = alias
		}
	}

	if displayName == "" {
		displayName = cont.Name
	} else if len(displayName) > 50 {
		// truncate display name to fit in one line.
		displayName = displayName[:50] + "..."
	}

	// Add the full container name to the display name.
	if displayName != cont.Name {
		displayName = fmt.Sprintf("%s (%s)", displayName, cont.Name)
	}

	return displayName
}

// Escape the non-path characters on a container name.
func escapeContainerName(containerName string) string {
	parts := strings.Split(containerName, "/")
	for i := range parts {
		parts[i] = url.QueryEscape(parts[i])
	}
	return strings.Join(parts, "/")
}
