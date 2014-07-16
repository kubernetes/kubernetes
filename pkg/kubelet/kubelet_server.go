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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/google/cadvisor/info"
	"gopkg.in/v1/yaml"
)

// Server is a http.Handler which exposes kubelet functionality over HTTP.
type Server struct {
	Kubelet         kubeletInterface
	UpdateChannel   chan<- manifestUpdate
	DelegateHandler http.Handler
}

// kubeletInterface contains all the kubelet methods required by the server.
// For testablitiy.
type kubeletInterface interface {
	GetContainerInfo(podID, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	GetMachineStats(req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	GetPodInfo(name string) (api.PodInfo, error)
}

func (s *Server) error(w http.ResponseWriter, err error) {
	http.Error(w, fmt.Sprintf("Internal Error: %v", err), http.StatusInternalServerError)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer httplog.MakeLogged(req, &w).Log()

	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	// TODO: use an http.ServeMux instead of a switch.
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
	case u.Path == "/podInfo":
		podID := u.Query().Get("podID")
		if len(podID) == 0 {
			http.Error(w, "Missing 'podID=' query entry.", http.StatusBadRequest)
			return
		}
		info, err := s.Kubelet.GetPodInfo(podID)
		if err != nil {
			s.error(w, err)
			return
		}
		data, err := json.Marshal(info)
		if err != nil {
			s.error(w, err)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Header().Add("Content-type", "application/json")
		w.Write(data)
	case strings.HasPrefix(u.Path, "/stats"):
		s.serveStats(w, req)
	default:
		s.DelegateHandler.ServeHTTP(w, req)
	}
}

func (s *Server) serveStats(w http.ResponseWriter, req *http.Request) {
	// /stats/<podid>/<containerName>
	components := strings.Split(strings.TrimPrefix(path.Clean(req.URL.Path), "/"), "/")
	var stats *info.ContainerInfo
	var err error
	var query info.ContainerInfoRequest
	decoder := json.NewDecoder(req.Body)
	err = decoder.Decode(&query)
	if err != nil && err != io.EOF {
		s.error(w, err)
		return
	}
	switch len(components) {
	case 1:
		// Machine stats
		stats, err = s.Kubelet.GetMachineStats(&query)
	case 2:
		// pod stats
		// TODO(monnand) Implement this
		errors.New("pod level status currently unimplemented")
	case 3:
		stats, err = s.Kubelet.GetContainerInfo(components[1], components[2], &query)
	default:
		http.Error(w, "unknown resource.", http.StatusNotFound)
		return
	}
	if err != nil {
		s.error(w, err)
		return
	}
	if stats == nil {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "{}")
		return
	}
	data, err := json.Marshal(stats)
	if err != nil {
		s.error(w, err)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Header().Add("Content-type", "application/json")
	w.Write(data)
	return
}
