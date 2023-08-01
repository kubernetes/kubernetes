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

/*
 * This test checks that various VolumeSources are working.
 *
 * There are two ways, how to test the volumes:
 * 1) With containerized server (NFS, Ceph, iSCSI, ...)
 * The test creates a server pod, exporting simple 'index.html' file.
 * Then it uses appropriate VolumeSource to import this file into a client pod
 * and checks that the pod can see the file. It does so by importing the file
 * into web server root and loading the index.html from it.
 *
 * These tests work only when privileged containers are allowed, exporting
 * various filesystems (ex: NFS) usually needs some mounting or
 * other privileged magic in the server pod.
 *
 * Note that the server containers are for testing purposes only and should not
 * be used in production.
 *
 * 2) With server outside of Kubernetes
 * Appropriate server must exist somewhere outside
 * the tested Kubernetes cluster. The test itself creates a new volume,
 * and checks, that Kubernetes can use it as a volume.
 */

package storage

import (
	"context"

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"

	"github.com/onsi/ginkgo/v2"
	admissionapi "k8s.io/pod-security-admission/api"
)

// TODO(#99468): Check if these tests are still needed.
var _ = SIGDescribe("Volumes", func() {
	f := framework.NewDefaultFramework("volume")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// note that namespace deletion is handled by delete-namespace flag
	// filled in BeforeEach
	var namespace *v1.Namespace
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("gci", "ubuntu", "custom")

		namespace = f.Namespace
		c = f.ClientSet
	})

	////////////////////////////////////////////////////////////////////////
	// NFS
	////////////////////////////////////////////////////////////////////////
	ginkgo.Describe("NFSv4", func() {
		ginkgo.It("should be mountable for NFSv4", func(ctx context.Context) {
			config, _, serverHost := e2evolume.NewNFSServer(ctx, c, namespace.Name, []string{})
			ginkgo.DeferCleanup(e2evolume.TestServerCleanup, f, config)

			tests := []e2evolume.Test{
				{
					Volume: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server:   serverHost,
							Path:     "/",
							ReadOnly: true,
						},
					},
					File:            "index.html",
					ExpectedContent: "Hello from NFS!",
				},
			}

			// Must match content of test/images/volumes-tester/nfs/index.html
			e2evolume.TestVolumeClient(ctx, f, config, nil, "" /* fsType */, tests)
		})
	})

	ginkgo.Describe("NFSv3", func() {
		ginkgo.It("should be mountable for NFSv3", func(ctx context.Context) {
			config, _, serverHost := e2evolume.NewNFSServer(ctx, c, namespace.Name, []string{})
			ginkgo.DeferCleanup(e2evolume.TestServerCleanup, f, config)

			tests := []e2evolume.Test{
				{
					Volume: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server:   serverHost,
							Path:     "/exports",
							ReadOnly: true,
						},
					},
					File:            "index.html",
					ExpectedContent: "Hello from NFS!",
				},
			}
			// Must match content of test/images/volume-tester/nfs/index.html
			e2evolume.TestVolumeClient(ctx, f, config, nil, "" /* fsType */, tests)
		})
	})
})
