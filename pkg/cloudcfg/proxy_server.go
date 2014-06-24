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

package cloudcfg

import (
	"io/ioutil"
	"log"
	"net/http"
)

type ProxyServer struct {
	Host   string
	Client *http.Client
}

func (s *ProxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequest(r.Method, s.Host+r.URL.Path, r.Body)
	if err != nil {
		log.Printf("Error: %#v", err)
		return
	}
	res, err := s.Client.Do(req)
	if err != nil {
		log.Printf("Error: %#v", err)
		return
	}
	w.WriteHeader(res.StatusCode)
	defer res.Body.Close()
	data, err := ioutil.ReadAll(res.Body)
	if err != nil {
		log.Printf("Error: %#v", err)
		return
	}
	w.Write(data)
}
