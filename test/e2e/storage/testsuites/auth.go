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

package testsuites

import (
	"context"
	"math/rand"
	"strings"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
)

type authTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitauthTestSuite returns authTestSuite that implements TestSuite interface
func InitCustomAuthTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &authTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "auth",
			TestPatterns: patterns,
		},
	}
}

func InitAuthTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.AuthDynamicPV,
	}
	return InitCustomAuthTestSuite(patterns)
}

func (s *authTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *authTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	_, ok := driver.(storageframework.DynamicPVTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}

	_, ok = driver.(storageframework.AuthTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %s does not support auth -- skipping", dInfo.Name)
	}
}

func (a *authTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		driverInfo    *storageframework.DriverInfo
		config        *storageframework.PerTestConfig
		driverCleanup func()

		resource  *storageframework.VolumeResource
		podConfig *e2epod.Config

		// get from AuthTestDriver interface
		scAuthParams   map[string]string
		authSecretData []map[string]string
		authMatchGroup [][]storageframework.CSIStorageClassAuthParamKey

		// store created secretNames for cleanup
		secretNames []string
	}
	var l local

	f := framework.NewFrameworkWithCustomTimeouts("auth", storageframework.GetDriverTimeouts(driver))

	init := func() {
		l = local{}

		l.driverInfo = driver.GetDriverInfo()
		l.config, l.driverCleanup = driver.PrepareTest(f)
		testVolumeSizeRange := a.GetTestSuiteInfo().SupportedSizeRange

		authDriver, ok := driver.(storageframework.AuthTestDriver)
		framework.ExpectEqual(ok, true, "Driver not yet implement interface: AuthTestDriver")

		scAuthParams := authDriver.GetStorageClassAuthParameters(l.config)
		framework.ExpectNotEqual(scAuthParams, nil,
			"GetStorageClassAuthParameters()  must return at least one map in AuthDynamicPV test pattern")
		framework.ExpectEqual(len(scAuthParams) > 0, true,
			"GetStorageClassAuthParameters()  must return at least one map in AuthDynamicPV test pattern")
		l.scAuthParams = scAuthParams

		authSecretData := authDriver.GetAuthSecretData()
		framework.ExpectNotEqual(authSecretData, nil,
			"GetAuthSecretData()  must return at least one map in AuthDynamicPV test pattern")
		framework.ExpectEqual(len(authSecretData) > 0, true,
			"GetAuthSecretData()  must return at least one map in AuthDynamicPV test pattern")
		l.authSecretData = authSecretData

		authMatchGroup := authDriver.GetAuthMatchGroup()
		framework.ExpectNotEqual(authMatchGroup, nil,
			"GetAuthMatchGroup()  must return at least one map in AuthDynamicPV test pattern")
		framework.ExpectEqual(len(authMatchGroup) > 0, true,
			"GetAuthMatchGroup()  must return at least one map in AuthDynamicPV test pattern")
		l.authMatchGroup = authMatchGroup

		// create auth secret define in StorageClass auth params
		for paramKey, paramValue := range l.scAuthParams {
			if strings.HasSuffix(paramKey, "secret-name") {
				secretName := paramValue
				err := createOrUpdateSecret(f.ClientSet, f.Namespace.Name,
					secretName, l.authSecretData[0])
				framework.ExpectNoError(err, "Failed to create secret: ", secretName)
				l.secretNames = append(l.secretNames, secretName)
			}
		}

		l.resource = storageframework.CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)

		l.podConfig = &e2epod.Config{
			NS:            f.Namespace.Name,
			PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
			SeLinuxLabel:  e2epod.GetLinuxLabel(),
			NodeSelection: l.config.ClientNodeSelection,
			ImageID:       e2epod.GetDefaultTestImageID(),
		}
	}

	cleanup := func() {
		var errs []error

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource())
			l.resource = nil
		}

		for _, secretName := range l.secretNames {
			err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(),
				secretName, metav1.DeleteOptions{})
			errs = append(errs, err)
		}

		errs = append(errs, storageutils.TryFunc(l.driverCleanup))
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
	}

	ginkgo.It("pod should binding pvc success using current auth secret data", func() {
		init()
		defer cleanup()

		// If the pod can enter the running state, it means that the pvc has correctly
		// used the expected secret to attach to the node and publish to the correct node path
		pod, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(f.ClientSet, pod))
		}()
	})

	ginkgo.It("pod should binding pvc fail when the secret data in a match group is inconsistent", func() {
		init()
		defer cleanup()

		for _, group := range l.authMatchGroup {
			if len(group) < 1 {
				break
			}

			// pick a random CSIStage in match group, and make the secret data
			// inconsistent with other CSIStage in the group
			randomIndex := rand.Intn(len(group))
			inconsistentCSIStage := group[randomIndex]

			inconsistentSecretName := l.scAuthParams[inconsistentCSIStage.ToString()]
			inconsistentSecretData := makeInconsistentSecretData(l.authSecretData[0])

			err := createOrUpdateSecret(f.ClientSet, f.Namespace.Name,
				inconsistentSecretName, inconsistentSecretData)
			framework.ExpectNoError(err, "Failed to create secret: ", inconsistentSecretName)

			pod, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
			framework.ExpectError(err)

			defer func() {
				framework.ExpectNoError(e2epod.DeletePodWithWait(f.ClientSet, pod))
			}()
		}

	})

	ginkgo.It("The no-auth-storageclass and the auth-storageclass can exist at the same time and be correctly used in different pods", func() {
		init()
		defer cleanup()

		ginkgo.By("creating a pod use auth-pvc")
		podUseAuthPvc, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err)

		ginkgo.By("creating a StorageClass without auth")
		storageClassWithoutAuth := l.resource.Sc.DeepCopy()
		// clean up ObjectMeta that remove some non-essential fields left by DeepCopy,
		// such as uuid
		storageClassWithoutAuth.ObjectMeta = metav1.ObjectMeta{
			Name: l.resource.Sc.Name + "-without-auth",
		}
		cleanStorageClassAuthParams(storageClassWithoutAuth, l.scAuthParams)
		_, err = f.ClientSet.StorageV1().StorageClasses().Create(context.TODO(),
			storageClassWithoutAuth, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a PVC without auth")
		pvcWithoutAuth := l.resource.Pvc.DeepCopy()
		pvcWithoutAuth.ObjectMeta = metav1.ObjectMeta{
			Name: l.resource.Pvc.Name + "-without-auth",
		}
		pvcWithoutAuth.Spec.VolumeName = ""
		pvcWithoutAuth.Spec.StorageClassName = &storageClassWithoutAuth.Name
		_, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(),
			pvcWithoutAuth, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		l.podConfig.PVCs = []*v1.PersistentVolumeClaim{pvcWithoutAuth}

		ginkgo.By("creating a pod use no-auth-pvc")
		podUseNoAuthPvc, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err)

		defer func() {
			var errs []error

			errs = append(errs, e2epod.DeletePodWithWait(f.ClientSet, podUseAuthPvc))
			errs = append(errs, e2epod.DeletePodWithWait(f.ClientSet, podUseNoAuthPvc))

			errs = append(errs, e2epv.DeletePersistentVolumeClaim(f.ClientSet, pvcWithoutAuth.Name, f.Namespace.Name))

			framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		}()

	})

	ginkgo.It("Two auth-storageclass with different secret data can exist at the same time and be correctly used in different pods", func() {
		init()
		defer cleanup()

		if len(l.authSecretData) < 2 {
			e2eskipper.Skipf("authSecretData less than 2, skipping multiple auth-storageclass test")
			return
		}

		// we assume that the secret created in init() is secretA, and the secret created separately
		// for multiple auth-storageclass is secretB
		ginkgo.By("creating a pod use pvc with SecretA")
		podUsePvcWithSecretA, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err)

		ginkgo.By("generating a StorageClass with SecretB")
		storageClassWithSecretB := l.resource.Sc.DeepCopy()
		storageClassWithSecretB.ObjectMeta = metav1.ObjectMeta{
			Name: l.resource.Sc.Name + "-with-secret-b",
		}

		ginkgo.By("creating SecretB and fill in storageClassWithSecretB's params")
		secretDataIndex := rand.Intn(len(l.authSecretData)-1) + 1
		for paramKey, paramValue := range l.scAuthParams {
			if strings.HasSuffix(paramKey, "secret-name") {
				secretBName := paramValue + "-secret-b"
				err := createOrUpdateSecret(f.ClientSet, f.Namespace.Name,
					secretBName, l.authSecretData[secretDataIndex])
				framework.ExpectNoError(err, "Failed to create secret: ", secretBName)

				// set SecretB in storageclass params
				storageClassWithSecretB.Parameters[paramKey] = secretBName
			}
		}

		ginkgo.By("creating a StorageClass with SecretB")
		_, err = f.ClientSet.StorageV1().StorageClasses().Create(context.TODO(),
			storageClassWithSecretB, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a PVC use StorageClass with secretB")
		pvcWithSecretB := l.resource.Pvc.DeepCopy()
		pvcWithSecretB.ObjectMeta = metav1.ObjectMeta{
			Name: l.resource.Pvc.Name + "-with-secret-b",
		}
		pvcWithSecretB.Spec.VolumeName = ""
		pvcWithSecretB.Spec.StorageClassName = &storageClassWithSecretB.Name
		_, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(),
			pvcWithSecretB, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		l.podConfig.PVCs = []*v1.PersistentVolumeClaim{pvcWithSecretB}

		ginkgo.By("creating a Pod with secretB-pvc")
		podUsePvcWithSecretB, err := e2epod.CreateSecPod(f.ClientSet, l.podConfig, f.Timeouts.PodStartShort)
		framework.ExpectNoError(err)

		defer func() {
			var errs []error

			errs = append(errs, e2epod.DeletePodWithWait(f.ClientSet, podUsePvcWithSecretB))
			errs = append(errs, e2epod.DeletePodWithWait(f.ClientSet, podUsePvcWithSecretA))

			errs = append(errs, e2epv.DeletePersistentVolumeClaim(f.ClientSet, pvcWithSecretB.Name, f.Namespace.Name))

			framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		}()
	})
}

func createOrUpdateSecret(kubeClient clientset.Interface, namespace, name string, data map[string]string) error {
	secret := makeSecret(namespace, name)
	secret.StringData = data

	if createdSecret, err := kubeClient.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{}); err == nil {
		framework.Logf("Updating secret %v in ns %v", name, namespace)
		createdSecret.StringData = secret.StringData
		_, err = kubeClient.CoreV1().Secrets(namespace).Update(context.TODO(), createdSecret, metav1.UpdateOptions{})
		return err
	}

	framework.Logf("Creating secret %v in ns %v", name, namespace)
	_, err := kubeClient.CoreV1().Secrets(namespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	return err
}

func makeSecret(namespace, name string) *v1.Secret {
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	return secret
}

func makeInconsistentSecretData(data map[string]string) map[string]string {
	if data == nil {
		return nil
	}

	inconsistentData := map[string]string{}
	for k, v := range data {
		inconsistentData[k] = v + "err"
	}

	return inconsistentData
}

func cleanStorageClassAuthParams(sc *storagev1.StorageClass, authParams map[string]string) {
	if authParams == nil {
		return
	}

	for key := range authParams {
		delete(sc.Parameters, key)
	}
}
