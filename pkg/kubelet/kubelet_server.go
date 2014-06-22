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

package kubelet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"gopkg.in/v1/yaml"
)

type KubeletServer struct {
	Kubelet       kubeletInterface
	UpdateChannel chan manifestUpdate
}

// kubeletInterface contains all the kubelet methods required by the server.
// For testablitiy.
type kubeletInterface interface {
	GetContainerID(name string) (string, bool, error)
	GetContainerStats(name string) (*api.ContainerStats, error)
	GetContainerInfo(name string) (string, error)
}

func (s *KubeletServer) error(w http.ResponseWriter, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Internal Error: %#v", err)
}

func (s *KubeletServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	switch {
	case u.Path == "/container" || u.Path == "/containers":
		defer req.Body.Close()
		data, err := ioutil.ReadAll(req.Body)
		if err != nil {
			s.error(w, err)
			return
		}
		if u.Path == "/container" {
			// This is to provide backward compatibility. It only supports a single manifest
			var manifest api.ContainerManifest
			err = yaml.Unmarshal(data, &manifest)
			if err != nil {
				s.error(w, err)
				return
			}
			s.UpdateChannel <- manifestUpdate{httpServerSource, []api.ContainerManifest{manifest}}
		} else if u.Path == "/containers" {
			var manifests []api.ContainerManifest
			err = yaml.Unmarshal(data, &manifests)
			if err != nil {
				s.error(w, err)
				return
			}
			s.UpdateChannel <- manifestUpdate{httpServerSource, manifests}
		}
	case u.Path == "/containerStats":
		container := u.Query().Get("container")
		if len(container) == 0 {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, "Missing container query arg.")
			return
		}
		stats, err := s.Kubelet.GetContainerStats(container)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, "Internal Error: %#v", err)
			return
		}
		if stats == nil {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "{}")
			return
		}
		data, err := json.Marshal(stats)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, "Internal Error: %#v", err)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Header().Add("Content-type", "application/json")
		w.Write(data)
	case u.Path == "/containerInfo":
		container := u.Query().Get("container")
		if len(container) == 0 {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, "Missing container selector arg.")
			return
		}
		id, found, err := s.Kubelet.GetContainerID(container)
		if !found {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "{}")
			return
		}
		body, err := s.Kubelet.GetContainerInfo(id)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, "Internal Error: %#v", err)
			return
		}
		w.Header().Add("Content-type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, body)
	default:
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "Not found.")
	}
}
