// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package mocks

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
)

type ServerResponseData struct {
	StatuCode *int
	Body      *string
}

type Server struct {
	HttpServer      *httptest.Server
	DefaultResponse *ServerResponseData
	Responses       map[string]*ServerResponseData
}

func newUnstartedTestServer() (server *Server) {
	status := 200
	body := ""

	server = &Server{
		nil,
		&ServerResponseData{&status, &body},
		make(map[string]*ServerResponseData),
	}

	server.HttpServer = httptest.NewUnstartedServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")

			var response *ServerResponseData
			for k, v := range server.Responses {
				if strings.HasPrefix(r.URL.Path, k) {
					response = v
					break
				}
			}

			if response == nil {
				response = server.DefaultResponse
			}

			w.WriteHeader(*response.StatuCode)
			fmt.Fprintln(w, *response.Body)
		}))
	return
}

func NewTestServer() (server *Server) {
	server = newUnstartedTestServer()
	server.HttpServer.Start()
	return
}

func NewTlsTestServer() (server *Server) {
	server = newUnstartedTestServer()
	server.HttpServer.StartTLS()
	return
}

func (s *Server) Close() {
	if s.HttpServer != nil {
		s.HttpServer.Close()
	}
}

func (s *Server) SetResponse(status int, body string) {
	s.DefaultResponse = &ServerResponseData{StatuCode: &status, Body: &body}
}

func (s *Server) SetResponseJson(status int, v interface{}) {
	s.SetResponse(status, s.toJson(v))
}

func (s *Server) SetResponseForPath(path string, status int, body string) {
	s.Responses[path] = &ServerResponseData{&status, &body}
}

func (s *Server) SetResponseJsonForPath(path string, status int, v interface{}) {
	s.SetResponseForPath(path, status, s.toJson(v))
}

func (s *Server) GetAddressAndPort() (address string, port int, err error) {
	serverURL, err := url.Parse(s.HttpServer.URL)
	if err != nil {
		return
	}

	hostList := strings.Split(serverURL.Host, ":")
	address = hostList[0]
	port, err = strconv.Atoi(hostList[1])
	if err != nil {
		return
	}

	return
}

func (s *Server) toJson(v interface{}) string {
	res, err := json.Marshal(v)
	if err != nil {
		// Since this method is only for testing, don't return
		// any errors, just panic.
		panic("Error serializing struct into JSON")
	}
	// json.Marshal returns []byte, convert to string
	return string(res[:])
}
