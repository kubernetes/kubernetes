/*
Copyright 2024 The Kubernetes Authors.

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

package images

import (
	"errors"
	"fmt"
	"runtime"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

// NOTE: Please do NOT add any to this list!!
//
// We are aiming to consolidate on: registry.k8s.io/e2e-test-images/agnhost
// The sources for which are in test/images/agnhost.
// If agnhost is missing functionality for your tests, please reach out to SIG Testing.
var _, file, line, _ = runtime.Caller(0)
var PermittedImages = sets.New(
	"gcr.io/authenticated-image-pulling/alpine",
	"gcr.io/authenticated-image-pulling/windows-nanoserver",
	"gcr.io/k8s-authenticated-test/agnhost",
	"invalid.registry.k8s.io/invalid/alpine",
	"registry.k8s.io/build-image/distroless-iptables",
	"registry.k8s.io/cloud-provider-gcp/gcp-compute-persistent-disk-csi-driver",
	"registry.k8s.io/e2e-test-images/agnhost",
	"registry.k8s.io/e2e-test-images/apparmor-loader",
	"registry.k8s.io/e2e-test-images/busybox",
	"registry.k8s.io/e2e-test-images/cuda-vector-add",
	"registry.k8s.io/e2e-test-images/httpd",
	"registry.k8s.io/e2e-test-images/ipc-utils",
	"registry.k8s.io/e2e-test-images/jessie-dnsutils",
	"registry.k8s.io/e2e-test-images/kitten",
	"registry.k8s.io/e2e-test-images/nautilus",
	"registry.k8s.io/e2e-test-images/nginx",
	"registry.k8s.io/e2e-test-images/node-perf/npb-ep",
	"registry.k8s.io/e2e-test-images/node-perf/npb-is",
	"registry.k8s.io/e2e-test-images/node-perf/tf-wide-deep",
	"registry.k8s.io/e2e-test-images/nonewprivs",
	"registry.k8s.io/e2e-test-images/nonroot",
	"registry.k8s.io/e2e-test-images/perl",
	"registry.k8s.io/e2e-test-images/redis",
	"registry.k8s.io/e2e-test-images/regression-issue-74839",
	"registry.k8s.io/e2e-test-images/resource-consumer",
	"registry.k8s.io/e2e-test-images/sample-apiserver",
	"registry.k8s.io/e2e-test-images/volume/iscsi",
	"registry.k8s.io/e2e-test-images/volume/nfs",
	"registry.k8s.io/e2e-test-images/volume/rbd",
	"registry.k8s.io/etcd",
	"registry.k8s.io/pause",
	"registry.k8s.io/prometheus-dummy-exporter",
	"registry.k8s.io/prometheus-to-sd",
	"registry.k8s.io/sd-dummy-exporter",
	"registry.k8s.io/sig-storage/csi-attacher",
	"registry.k8s.io/sig-storage/csi-external-health-monitor-controller",
	"registry.k8s.io/sig-storage/csi-node-driver-registrar",
	"registry.k8s.io/sig-storage/csi-provisioner",
	"registry.k8s.io/sig-storage/csi-resizer",
	"registry.k8s.io/sig-storage/csi-snapshotter",
	"registry.k8s.io/sig-storage/hello-populator",
	"registry.k8s.io/sig-storage/hostpathplugin",
	"registry.k8s.io/sig-storage/livenessprobe",
	"registry.k8s.io/sig-storage/nfs-provisioner",
	"registry.k8s.io/sig-storage/volume-data-source-validator",
)

func VerifyImages(images ...string) *ImageError {
	var errs []error

	for _, image := range images {
		baseImage := strings.Split(image, ":")[0]
		if !PermittedImages.Has(baseImage) {
			errs = append(errs, fmt.Errorf("usage of base image %s from %s is not permitted", baseImage, image))
		}
	}
	if len(errs) > 0 {
		return &ImageError{
			error: errors.Join(errs...),
			File:  file,
			Line:  line,
		}
	}

	// TODO: generate error when PermittedImages contains unused entries.

	return nil
}

// ImageError is an error with source code information.
type ImageError struct {
	error
	File string
	Line int
}
