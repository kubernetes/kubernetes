/*
Copyright 2021 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

type csiSecretType string

const (
	controllerPublishSecret csiSecretType = "ControllerPublishSecret"
	nodeStageSecret         csiSecretType = "NodeStageSecret"
	nodePublishSecret       csiSecretType = "NodePublishSecret"
	controllerExpandSecret  csiSecretType = "ControllerExpandSecret"

	// secretDeleteTimeout is how long to wait for a secret to be deleted.
	secretDeleteTimeout = 1 * time.Minute
	// secretDeleteTimeout is how long to wait for a (portected) secret not to be deleted.
	secretNotDeletedTimeout = 5 * time.Second
)

var _ = utils.SIGDescribe("Secret Protection", func() {
	var (
		err    error
		secret *v1.Secret
	)

	f := framework.NewDefaultFramework("secret-protection")

	ginkgo.BeforeEach(func() {
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(f.ClientSet, framework.TestContext.NodeSchedulableTimeout))

		secret = &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      "protect-secret-" + string(uuid.NewUUID()),
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1\n"),
			},
		}

		ginkgo.By("Creating a secret")
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		ginkgo.By("Checking that Secret Protection finalizer is set")
		framework.ExpectEqual(slice.ContainsString(secret.ObjectMeta.Finalizers, volumeutil.SecretProtectionFinalizer, nil), true, "Secret Protection finalizer(%v) is not set in %v", volumeutil.SecretProtectionFinalizer, secret.ObjectMeta.Finalizers)

	})

	ginkgo.It("Verify \"immediate\" deletion of a secret that is not used", func() {
		ginkgo.By("Deleting the secret")
		err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Error deleting secret")

		ginkgo.By("Waiting for the unprotected secret to be deleted")
		err = waitForSecretDeleted(f.ClientSet, f.Namespace.Name, secret.Name, framework.Poll, secretDeleteTimeout)
		framework.ExpectNoError(err, "Error waiting for secret deleted")
	})

	ginkgo.It("Verify that secret used by a Pod is not removed immediately", func() {
		ginkgo.By("Creating a pod using the secret")
		podConfig := &e2epod.Config{
			NS:      f.Namespace.Name,
			ImageID: e2epod.GetDefaultTestImageID(),
			InlineVolumeSources: []*v1.VolumeSource{
				{
					Secret: &v1.SecretVolumeSource{
						SecretName: secret.Name,
					},
				},
			},
		}
		pod, err := e2epod.CreateSecPod(f.ClientSet, podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err, "Error creating secret")

		ginkgo.By("Deleting the secret")
		err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Error deleting secret")

		ginkgo.By("Waiting for the protected secret not to be deleted")
		err = waitForSecretDeleted(f.ClientSet, f.Namespace.Name, secret.Name, framework.Poll, secretNotDeletedTimeout)
		framework.ExpectError(err, "Protected Secret is unexpectedly deleted")

		ginkgo.By("Deleting the pod using the secret")
		err = e2epod.DeletePodWithWait(f.ClientSet, pod)

		ginkgo.By("Waiting for the unprotected secret to be deleted")
		err = waitForSecretDeleted(f.ClientSet, f.Namespace.Name, secret.Name, framework.Poll, secretDeleteTimeout)
		framework.ExpectNoError(err, "Error waiting for secret deleted")
	})

	ginkgo.It("Verify that secret used by a CSI PV as controllerPublishSecret is not removed immediately", func() {
		testDeleteSecretUsedByPV(f.ClientSet, f.Namespace.Name, secret.Name, controllerPublishSecret)
	})

	ginkgo.It("Verify that secret used by a CSI PV as nodeStageSecret is not removed immediately", func() {
		testDeleteSecretUsedByPV(f.ClientSet, f.Namespace.Name, secret.Name, nodeStageSecret)
	})

	ginkgo.It("Verify that secret used by a CSI PV as nodePublishSecret is not removed immediately", func() {
		testDeleteSecretUsedByPV(f.ClientSet, f.Namespace.Name, secret.Name, nodePublishSecret)
	})

	ginkgo.It("Verify that secret used by a CSI PV as controllerExpandSecret is not removed immediately", func() {
		testDeleteSecretUsedByPV(f.ClientSet, f.Namespace.Name, secret.Name, controllerExpandSecret)
	})
})

func testDeleteSecretUsedByPV(cs clientset.Interface, namespace, secretName string, secretType csiSecretType) {
	ginkgo.By(fmt.Sprintf("Creating a CSI PV using the secret as %v", secretType))
	pv := createPVUsingSecret(cs, namespace, secretName, secretType)

	ginkgo.By("Deleting the secret")
	err := cs.CoreV1().Secrets(namespace).Delete(context.TODO(), secretName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "Error deleting secret")

	ginkgo.By("Waiting for the protected secret not to be deleted")
	err = waitForSecretDeleted(cs, namespace, secretName, framework.Poll, secretNotDeletedTimeout)
	framework.ExpectError(err, "Protected Secret is unexpectedly deleted")

	ginkgo.By("Deleting the PV using the secret")
	err = e2epv.DeletePersistentVolume(cs, pv.Name)
	framework.ExpectNoError(err, "Failed to delete PV")

	ginkgo.By("Waiting for the unprotected secret to be deleted")
	err = waitForSecretDeleted(cs, namespace, secretName, framework.Poll, secretDeleteTimeout)
	framework.ExpectNoError(err, "Error waiting for secret deleted")
}

func createPVUsingSecret(cs clientset.Interface, namespace, secretName string, secretType csiSecretType) *v1.PersistentVolume {
	secretRef := &v1.SecretReference{
		Namespace: namespace,
		Name:      secretName,
	}
	csiVolSource := &v1.CSIPersistentVolumeSource{
		Driver:       "com.example.dummy",
		VolumeHandle: string(uuid.NewUUID()),
	}

	switch secretType {
	case controllerPublishSecret:
		csiVolSource.ControllerPublishSecretRef = secretRef
	case nodeStageSecret:
		csiVolSource.NodeStageSecretRef = secretRef
	case nodePublishSecret:
		csiVolSource.NodePublishSecretRef = secretRef
	case controllerExpandSecret:
		csiVolSource.ControllerExpandSecretRef = secretRef
	default:
		framework.Failf("Unkown secretType %v is specified", secretType)
	}

	pvConfig := e2epv.PersistentVolumeConfig{
		PVSource: v1.PersistentVolumeSource{
			CSI: csiVolSource,
		},
	}

	pv := e2epv.MakePersistentVolume(pvConfig)
	pv, err := e2epv.CreatePV(cs, pv)
	framework.ExpectNoError(err, "Error creating secret")

	return pv
}

// waitForSecretDeleted waits for a Secret to get deleted or until timeout occurs, whichever comes first.
func waitForSecretDeleted(c clientset.Interface, namespace, secretName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for Secret %s to get deleted", timeout, secretName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		_, err := c.CoreV1().Secrets(namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
		if err == nil {
			framework.Logf("Secret %s found (%v)", secretName, time.Since(start))
			continue
		}
		if apierrors.IsNotFound(err) {
			framework.Logf("Secret %s was removed", secretName)
			return nil
		}
		framework.Logf("Get secret %s in failed, ignoring for %v: %v", secretName, poll, err)
	}
	return fmt.Errorf("Secret %s still exists within %v", secretName, timeout)
}
