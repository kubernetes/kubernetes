/*
Copyright The Kubernetes Authors.

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

package ktesting

import (
	"testing"

	"github.com/onsi/gomega"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

func TestWithRESTConfig(t *testing.T) {
	tCtx := Init(t)
	tCtx = tCtx.WithRESTConfig(new(rest.Config))
	config := tCtx.RESTConfig()
	tCtx.Assert(config).NotTo(gomega.BeNil(), "RESTConfig()")
	tCtx.Assert(config.UserAgent).To(gomega.ContainSubstring("TestWithRESTConfig"), "UserAgent")
	mapper := tCtx.RESTMapper()
	tCtx.Assert(mapper).NotTo(gomega.BeNil(), "RESTMapper()")
	client := tCtx.Client()
	tCtx.Assert(client).NotTo(gomega.BeNil(), "Client()")
	dynamic := tCtx.Dynamic()
	tCtx.Assert(dynamic).NotTo(gomega.BeNil(), "Dynamic()")
	extensions := tCtx.APIExtensions()
	tCtx.Assert(extensions).NotTo(gomega.BeNil(), "APIExtensions()")

	otherCtx := tCtx.WithCancel()
	tCtx.Assert(otherCtx.RESTConfig()).To(gomega.Equal(config), "RESTConfig()")
	tCtx.Assert(otherCtx.RESTMapper()).To(gomega.BeIdenticalTo(mapper), "RESTMapper()")
	tCtx.Assert(otherCtx.Client()).To(gomega.BeIdenticalTo(client), "Client()")
	tCtx.Assert(otherCtx.Dynamic()).To(gomega.BeIdenticalTo(dynamic), "Dynamic()")
	tCtx.Assert(otherCtx.APIExtensions()).To(gomega.BeIdenticalTo(extensions), "APIExtensions()")

	tCtx.CleanupCtx(func(tCtx TContext) {
		tCtx.Assert(tCtx.RESTConfig()).To(gomega.Equal(config), "RESTConfig()")
		tCtx.Assert(tCtx.RESTMapper()).To(gomega.BeIdenticalTo(mapper), "RESTMapper()")
		tCtx.Assert(tCtx.Client()).To(gomega.BeIdenticalTo(client), "Client()")
		tCtx.Assert(tCtx.Dynamic()).To(gomega.BeIdenticalTo(dynamic), "Dynamic()")
		tCtx.Assert(tCtx.APIExtensions()).To(gomega.BeIdenticalTo(extensions), "APIExtensions()")
	})

	// Cancel, then let testing.T invoke test cleanup.
	tCtx.Cancel("test is complete")
}

func TestWithClients(t *testing.T) {
	tCtx := Init(t)
	config := &rest.Config{UserAgent: "my-user-agent"}
	mapper := &restmapper.DeferredDiscoveryRESTMapper{}
	client := clientset.NewForConfigOrDie(config)
	dynamic := dynamic.NewForConfigOrDie(config)
	extensions := apiextensions.NewForConfigOrDie(config)
	tCtx = tCtx.WithClients(config, mapper, client, dynamic, extensions)
	tCtx.Assert(tCtx.RESTConfig()).To(gomega.Equal(config), "RESTConfig()")
	tCtx.Assert(tCtx.RESTMapper()).To(gomega.BeIdenticalTo(mapper), "RESTMapper()")
	tCtx.Assert(tCtx.Client()).To(gomega.BeIdenticalTo(client), "Client()")
	tCtx.Assert(tCtx.Dynamic()).To(gomega.BeIdenticalTo(dynamic), "Dynamic()")
	tCtx.Assert(tCtx.APIExtensions()).To(gomega.BeIdenticalTo(extensions), "APIExtensions()")

	otherCtx := tCtx.WithCancel()
	tCtx.Assert(otherCtx.RESTConfig()).To(gomega.Equal(config), "RESTConfig()")
	tCtx.Assert(otherCtx.RESTMapper()).To(gomega.BeIdenticalTo(mapper), "RESTMapper()")
	tCtx.Assert(otherCtx.Client()).To(gomega.BeIdenticalTo(client), "Client()")
	tCtx.Assert(otherCtx.Dynamic()).To(gomega.BeIdenticalTo(dynamic), "Dynamic()")
	tCtx.Assert(otherCtx.APIExtensions()).To(gomega.BeIdenticalTo(extensions), "APIExtensions()")

	tCtx.CleanupCtx(func(tCtx TContext) {
		tCtx.Assert(tCtx.RESTConfig()).To(gomega.Equal(config), "RESTConfig()")
		tCtx.Assert(tCtx.RESTMapper()).To(gomega.BeIdenticalTo(mapper), "RESTMapper()")
		tCtx.Assert(tCtx.Client()).To(gomega.BeIdenticalTo(client), "Client()")
		tCtx.Assert(tCtx.Dynamic()).To(gomega.BeIdenticalTo(dynamic), "Dynamic()")
		tCtx.Assert(tCtx.APIExtensions()).To(gomega.BeIdenticalTo(extensions), "APIExtensions()")
	})

	// Cancel, then let testing.T invoke test cleanup.
	tCtx.Cancel("test is complete")
}
