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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
//
package app

import (
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
)

func startCSRSigningController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}] {
		return false, nil
	}
	if ctx.Options.ClusterSigningCertFile == "" || ctx.Options.ClusterSigningKeyFile == "" {
		return false, nil
	}
	c := ctx.ClientBuilder.ClientOrDie("certificate-controller")

	signer, err := signer.NewCSRSigningController(
		c,
		ctx.InformerFactory.Certificates().V1beta1().CertificateSigningRequests(),
		ctx.Options.ClusterSigningCertFile,
		ctx.Options.ClusterSigningKeyFile,
		ctx.Options.ClusterSigningDuration.Duration,
	)
	if err != nil {
		glog.Errorf("Failed to start certificate controller: %v", err)
		return false, nil
	}
	go signer.Run(1, ctx.Stop)

	return true, nil
}

func startCSRApprovingController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}] {
		return false, nil
	}
	c := ctx.ClientBuilder.ClientOrDie("certificate-controller")

	approver, err := approver.NewCSRApprovingController(
		c,
		ctx.InformerFactory.Certificates().V1beta1().CertificateSigningRequests(),
	)
	if err != nil {
		// TODO this is failing consistently in test-cmd and local-up-cluster.sh.  Fix them and make it consistent with all others which
		// cause a crash loop
		glog.Errorf("Failed to start certificate controller: %v", err)
		return false, nil
	}
	go approver.Run(1, ctx.Stop)

	return true, nil
}
