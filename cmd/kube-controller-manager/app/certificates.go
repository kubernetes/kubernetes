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
	certcontroller "k8s.io/kubernetes/pkg/controller/certificates"
)

func startCSRController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1alpha1", Resource: "certificatesigningrequests"}] {
		return false, nil
	}
	resyncPeriod := ResyncPeriod(&ctx.Options)()
	c := ctx.ClientBuilder.ClientOrDie("certificate-controller")
	certController, err := certcontroller.NewCertificateController(
		c,
		resyncPeriod,
		ctx.Options.ClusterSigningCertFile,
		ctx.Options.ClusterSigningKeyFile,
		certcontroller.NewGroupApprover(c.Certificates().CertificateSigningRequests(), ctx.Options.ApproveAllKubeletCSRsForGroup),
	)
	if err != nil {
		// TODO this is failing consistently in test-cmd and local-up-cluster.sh.  Fix them and make it consistent with all others which
		// cause a crash loop
		glog.Errorf("Failed to start certificate controller: %v", err)
		return false, nil
	}
	go certController.Run(1, ctx.Stop)
	return true, nil
}
