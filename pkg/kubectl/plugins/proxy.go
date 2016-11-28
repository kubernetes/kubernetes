/*
Copyright 2016 The Kubernetes Authors.

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

package plugins

import (
	"net"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/kubectl"
)

func ServePluginAPIProxy(clientConfig *restclient.Config) (net.Listener, error) {

	filter := &kubectl.FilterServer{
		AcceptPaths: kubectl.MakeRegexpArrayOrDie(kubectl.DefaultPathAcceptRE),
		RejectPaths: kubectl.MakeRegexpArrayOrDie(kubectl.DefaultPathRejectRE),
		AcceptHosts: kubectl.MakeRegexpArrayOrDie(kubectl.DefaultHostAcceptRE),
	}

	server, err := kubectl.NewProxyServer("", "/", "", filter, clientConfig)
	if err != nil {
		return nil, err
	}

	l, err := server.Listen("127.0.0.1", 0)
	if err != nil {
		return nil, err
	}

	glog.V(8).Infof("Starting to serve tunnel for plugin on %s", l.Addr().String())
	go func() {
		glog.Fatal(server.ServeOnListener(l))
	}()

	return l, nil
}
