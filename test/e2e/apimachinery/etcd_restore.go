/*
Copyright 2023 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"path"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
)

const (
	containerEngineCmd = "docker"
	certDir            = "/etc/srv/kubernetes"
	manifestDir        = "/etc/kubernetes/manifests"
	manifestTmpDir     = "/etc/kubernetes"
)

var _ = SIGDescribe("Etcd restore", framework.WithDisruptive(), func() {

	f := framework.NewDefaultFramework("etcd-restore")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		// This test requires:
		// - SSH
		// - master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		e2eskipper.SkipUnlessSSHKeyPresent()

		err := e2erc.RunRC(ctx, testutils.RCConfig{
			Client:    f.ClientSet,
			Name:      "restore",
			Namespace: f.Namespace.Name,
			Image:     imageutils.GetPauseImageName(),
			Replicas:  1,
		})
		framework.ExpectNoError(err)
	})

	ginkgo.It("should be able to restore from bumped snapshot", func(ctx context.Context) {
		lsOutput, _, err := masterExecOutput(ctx, fmt.Sprintf("ls %s | grep etcd", manifestDir))
		framework.ExpectNoError(err, "expected ls on static pod manifest dir to succeed")
		var etcdStaticPodManifests []string
		lsOutputSplit := strings.Split(lsOutput, "\n")
		for _, s := range lsOutputSplit {
			s = strings.TrimSpace(s)
			if s != "" {
				etcdStaticPodManifests = append(etcdStaticPodManifests, s)
			}
		}

		framework.Logf("found static pod manifests for etcd: %v", etcdStaticPodManifests)

		if len(etcdStaticPodManifests) < 1 {
			framework.Failf("expected at least one etcd static pod manifest, but got %d", len(etcdStaticPodManifests))
		}

		var hostDataDirs []string
		for _, manifest := range etcdStaticPodManifests {
			hostDataDir, _, err := masterExecOutput(ctx, fmt.Sprintf("jq -r -c '.spec.volumes[] | select( .name | contains(\"varetcd\")) .hostPath.path' %s", path.Join(manifestDir, manifest)))
			if err != nil {
				framework.Logf("expected etcd dataDir for manifest %s, error: %v", manifest, err)
			}

			mountDataFolder, _, err := masterExecOutput(ctx, fmt.Sprintf("jq -r -c '.spec.containers[0].env[] | select( .name | contains(\"DATA_DIRECTORY\")) .value' %s", path.Join(manifestDir, manifest)))
			if err != nil {
				framework.Logf("expected etcd dataDir mount path for manifest %s, error: %v", manifest, err)
			}
			hostDataDirs = append(hostDataDirs, path.Join(strings.TrimSpace(hostDataDir), path.Base(strings.TrimSpace(mountDataFolder))))
		}

		framework.Logf("found host data dirs for etcd: %v", hostDataDirs)

		etcdImage := "registry.k8s.io/etcd:3.5.10-0"
		for _, manifest := range etcdStaticPodManifests {
			etcdImage, _, err = masterExecOutput(ctx, fmt.Sprintf("jq -r -c '.spec.containers[] | select( .name | contains(\"etcd-container\")) .image' %s", path.Join(manifestDir, manifest)))
			if err != nil {
				framework.Failf("expected jq to return the etcd-container image, error: %v", err)
			}

			etcdImage = strings.TrimSpace(etcdImage)
			if etcdImage != "" {
				break
			}
		}

		framework.Logf("using etcd container image %s", etcdImage)
		// just taking the snapshot from the first control plane node
		manifest := etcdStaticPodManifests[0]
		snapshotFile := path.Join("/tmp", manifest+".snapshot")
		etcdCtlCmd := fmt.Sprintf("sudo %s run --rm --network=host -v %s:%s -v /tmp:/tmp %s etcdctl snapshot --key=%s --cert=%s --cacert=%s save %s",
			containerEngineCmd, certDir, certDir, etcdImage, path.Join(certDir, "etcd-peer.key"), path.Join(certDir, "etcd-peer.crt"), path.Join(certDir, "etcd-ca.crt"), snapshotFile)
		stdout, stderr, err := masterExecOutput(ctx, etcdCtlCmd)
		if err != nil {
			framework.Failf("error while taking snapshot with [%s]: [%v]; stdout: [%s]; stderr [%s]", etcdCtlCmd, err, stdout, stderr)
		}

		// expect that the snapshot actually was created on the host filesystem
		masterExec(ctx, fmt.Sprintf("ls -l %s", snapshotFile))

		postSnapshotNamespace, err := f.ClientSet.CoreV1().Namespaces().Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "etcd-restore-test"}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create post snapshot namespace")

		// take down all etcd containers by moving their static pod dirs
		for _, manifest := range etcdStaticPodManifests {
			masterExec(ctx, fmt.Sprintf("sudo mv %s %s", path.Join(manifestDir, manifest), path.Join(manifestTmpDir, manifest)))
		}

		// wait for all containers to shut down
		framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, time.Second*60, true, func(ctx context.Context) (bool, error) {
			stdout, stderr, err := masterExecOutput(ctx, "sudo crictl ps")
			if err != nil {
				framework.Failf("crictl returned error: %v; stderr: [%s]", err, stderr)
				return false, nil
			}

			if !strings.Contains(stdout, "etcd") {
				return true, nil
			}

			framework.Logf("etcd containers are still running, waiting for shut down: %s", stdout)
			return false, nil
		}))

		for _, dataDir := range hostDataDirs {
			userGroup, stderr, err := masterExecOutput(ctx, "sudo stat -c \"%u:%g\" "+path.Join(dataDir, "member"))
			if err != nil {
				framework.Failf("stat on the member dir returned error: %v; stderr: [%s]", err, stderr)
			}

			masterExec(ctx, fmt.Sprintf("sudo rm -rf %s", path.Join(dataDir, "member")))
			masterExec(ctx, fmt.Sprintf("sudo %s run --rm -v /tmp:/tmp -v %s:%s %s etcdctl snapshot restore %s --data-dir=%s --bump-revision 1000000000 --mark-compacted",
				containerEngineCmd, dataDir, dataDir, etcdImage, snapshotFile, path.Join(dataDir, "tmp")))
			masterExec(ctx, fmt.Sprintf("sudo mv %s %s", path.Join(dataDir, "tmp", "member"), path.Join(dataDir, "member")))
			masterExec(ctx, fmt.Sprintf("sudo chown -R %s %s", strings.TrimSpace(userGroup), path.Join(dataDir)))
		}

		// setup all etcd containers again
		for _, manifest := range etcdStaticPodManifests {
			masterExec(ctx, fmt.Sprintf("sudo mv %s %s", path.Join(manifestTmpDir, manifest), path.Join(manifestDir, manifest)))
		}

		// wait for the API to respond again, it takes some time for cri-o to schedule etcd again and settle
		framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, time.Minute*15, true, func(ctx context.Context) (bool, error) {
			namespaces, err := f.ClientSet.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
			if err != nil {
				framework.Logf("listing namespaces returned error: %v", err)
				return false, nil
			}

			if len(namespaces.Items) > 0 {
				return true, nil
			}

			return false, nil
		}))

		_, err = f.ClientSet.CoreV1().Namespaces().Get(ctx, postSnapshotNamespace.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected namespace %s to not exist after a successful restore from snapshot", postSnapshotNamespace.Name)
		}
	})
})
