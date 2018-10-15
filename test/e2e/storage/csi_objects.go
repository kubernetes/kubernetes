/*
Copyright 2018 The Kubernetes Authors.

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

// This file is used to deploy the CSI hostPath plugin
// More Information: https://github.com/kubernetes-csi/drivers/tree/master/pkg/hostpath

package storage

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"

	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/manifest"

	. "github.com/onsi/ginkgo"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	csicrd "k8s.io/csi-api/pkg/crd"
)

var (
	csiImageVersion  = flag.String("storage.csi.image.version", "", "overrides the default tag used for hostpathplugin/csi-attacher/csi-provisioner/driver-registrar images")
	csiImageRegistry = flag.String("storage.csi.image.registry", "quay.io/k8scsi", "overrides the default repository used for hostpathplugin/csi-attacher/csi-provisioner/driver-registrar images")
	csiImageVersions = map[string]string{
		"hostpathplugin":   "v0.4.0",
		"csi-attacher":     "v0.4.0",
		"csi-provisioner":  "v0.4.0",
		"driver-registrar": "v0.4.0",
	}
)

func csiContainerImage(image string) string {
	var fullName string
	fullName += *csiImageRegistry + "/" + image + ":"
	if *csiImageVersion != "" {
		fullName += *csiImageVersion
	} else {
		fullName += csiImageVersions[image]
	}
	return fullName
}

// Create the driver registrar cluster role if it doesn't exist, no teardown so that tests
// are parallelizable. This role will be shared with many of the CSI tests.
func csiDriverRegistrarClusterRole(
	config framework.VolumeTestConfig,
) *rbacv1.ClusterRole {
	// TODO(Issue: #62237) Remove impersonation workaround and cluster role when issue resolved
	By("Creating an impersonating superuser kubernetes clientset to define cluster role")
	rc, err := framework.LoadConfig()
	framework.ExpectNoError(err)
	rc.Impersonate = restclient.ImpersonationConfig{
		UserName: "superuser",
		Groups:   []string{"system:masters"},
	}
	superuserClientset, err := clientset.NewForConfig(rc)
	framework.ExpectNoError(err, "Failed to create superuser clientset: %v", err)
	By("Creating the CSI driver registrar cluster role")
	clusterRoleClient := superuserClientset.RbacV1().ClusterRoles()
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: csiDriverRegistrarClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{

			{
				APIGroups: []string{""},
				Resources: []string{"events"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"nodes"},
				Verbs:     []string{"get", "update", "patch"},
			},
		},
	}

	ret, err := clusterRoleClient.Create(role)
	if err != nil {
		if apierrs.IsAlreadyExists(err) {
			return ret
		}
		framework.ExpectNoError(err, "Failed to create %s cluster role: %v", role.GetName(), err)
	}

	return ret
}

func csiServiceAccount(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	componentName string,
	teardown bool,
) *v1.ServiceAccount {
	creatingString := "Creating"
	if teardown {
		creatingString = "Deleting"
	}
	By(fmt.Sprintf("%v a CSI service account for %v", creatingString, componentName))
	serviceAccountName := config.Prefix + "-" + componentName + "-service-account"
	serviceAccountClient := client.CoreV1().ServiceAccounts(config.Namespace)
	sa := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: config.Namespace,
		},
	}

	serviceAccountClient.Delete(sa.GetName(), &metav1.DeleteOptions{})
	err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := serviceAccountClient.Get(sa.GetName(), metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return nil
	}

	ret, err := serviceAccountClient.Create(sa)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s service account: %v", sa.GetName(), err)
	}

	return ret
}

func csiClusterRoleBindings(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	sa *v1.ServiceAccount,
	clusterRolesNames []string,
) {
	bindingString := "Binding"
	if teardown {
		bindingString = "Unbinding"
	}
	By(fmt.Sprintf("%v cluster roles %v to the CSI service account %v", bindingString, clusterRolesNames, sa.GetName()))
	clusterRoleBindingClient := client.RbacV1().ClusterRoleBindings()
	for _, clusterRoleName := range clusterRolesNames {
		binding := &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: clusterRoleName + "-" + config.Namespace + "-" + string(uuid.NewUUID()),
			},
			Subjects: []rbacv1.Subject{
				{
					Kind:      rbacv1.ServiceAccountKind,
					Name:      sa.GetName(),
					Namespace: sa.GetNamespace(),
				},
			},
			RoleRef: rbacv1.RoleRef{
				Kind:     "ClusterRole",
				Name:     clusterRoleName,
				APIGroup: "rbac.authorization.k8s.io",
			},
		}

		clusterRoleBindingClient.Delete(binding.GetName(), &metav1.DeleteOptions{})
		err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
			_, err := clusterRoleBindingClient.Get(binding.GetName(), metav1.GetOptions{})
			return apierrs.IsNotFound(err), nil
		})
		framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

		if teardown {
			return
		}

		_, err = clusterRoleBindingClient.Create(binding)
		if err != nil {
			framework.ExpectNoError(err, "Failed to create %s role binding: %v", binding.GetName(), err)
		}
	}
}

func csiControllerRole(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
) string {
	action := "Creating"
	if teardown {
		action = "Deleting"
	}

	By(fmt.Sprintf("%v CSI controller role", action))

	role, err := manifest.RoleFromManifest("test/e2e/testing-manifests/storage-csi/controller-role.yaml", config.Namespace)
	framework.ExpectNoError(err, "Failed to create Role from manifest")

	client.RbacV1().Roles(role.Namespace).Delete(role.Name, nil)
	err = wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := client.RbacV1().Roles(role.Namespace).Get(role.Name, metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return role.Name
	}

	_, err = client.RbacV1().Roles(role.Namespace).Create(role)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s role binding: %v", role.Name, err)
	}
	return role.Name
}

func csiControllerRoleBinding(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	roleName string,
	sa *v1.ServiceAccount,
) {
	bindingString := "Binding"
	if teardown {
		bindingString = "Unbinding"
	}
	By(fmt.Sprintf("%v roles %v to the CSI service account %v", bindingString, roleName, sa.GetName()))
	roleBindingClient := client.RbacV1().RoleBindings(config.Namespace)
	binding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-" + roleName + "-" + config.Namespace + "-role-binding",
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      sa.GetName(),
				Namespace: sa.GetNamespace(),
			},
		},
		RoleRef: rbacv1.RoleRef{
			Kind:     "Role",
			Name:     roleName,
			APIGroup: "rbac.authorization.k8s.io",
		},
	}

	roleBindingClient.Delete(binding.GetName(), &metav1.DeleteOptions{})
	err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := roleBindingClient.Get(binding.GetName(), metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return
	}

	_, err = roleBindingClient.Create(binding)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s role binding: %v", binding.GetName(), err)
	}
}

func csiHostPathPod(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	f *framework.Framework,
	sa *v1.ServiceAccount,
) *v1.Pod {
	priv := true
	mountPropagation := v1.MountPropagationBidirectional
	hostPathType := v1.HostPathDirectoryOrCreate
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      config.Prefix + "-pod",
			Namespace: config.Namespace,
			Labels: map[string]string{
				"app": "hostpath-driver",
			},
		},
		Spec: v1.PodSpec{
			ServiceAccountName: sa.GetName(),
			NodeName:           config.ServerNodeName,
			RestartPolicy:      v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:            "external-provisioner",
					Image:           csiContainerImage("csi-provisioner"),
					ImagePullPolicy: v1.PullAlways,
					SecurityContext: &v1.SecurityContext{
						Privileged: &priv,
					},
					Args: []string{
						"--v=5",
						"--provisioner=csi-hostpath",
						"--csi-address=/csi/csi.sock",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "socket-dir",
							MountPath: "/csi",
						},
					},
				},
				{
					Name:            "driver-registrar",
					Image:           csiContainerImage("driver-registrar"),
					ImagePullPolicy: v1.PullAlways,
					SecurityContext: &v1.SecurityContext{
						Privileged: &priv,
					},
					Args: []string{
						"--v=5",
						"--csi-address=/csi/csi.sock",
						"--kubelet-registration-path=/var/lib/kubelet/plugins/csi-hostpath/csi.sock",
					},
					Env: []v1.EnvVar{
						{
							Name: "KUBE_NODE_NAME",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "spec.nodeName",
								},
							},
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "socket-dir",
							MountPath: "/csi",
						},
						{
							Name:      "registration-dir",
							MountPath: "/registration",
						},
					},
				},
				{
					Name:            "external-attacher",
					Image:           csiContainerImage("csi-attacher"),
					ImagePullPolicy: v1.PullAlways,
					SecurityContext: &v1.SecurityContext{
						Privileged: &priv,
					},
					Args: []string{
						"--v=5",
						"--csi-address=$(ADDRESS)",
					},
					Env: []v1.EnvVar{
						{
							Name:  "ADDRESS",
							Value: "/csi/csi.sock",
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "socket-dir",
							MountPath: "/csi",
						},
					},
				},
				{
					Name:            "hostpath-driver",
					Image:           csiContainerImage("hostpathplugin"),
					ImagePullPolicy: v1.PullAlways,
					SecurityContext: &v1.SecurityContext{
						Privileged: &priv,
					},
					Args: []string{
						"--v=5",
						"--endpoint=$(CSI_ENDPOINT)",
						"--nodeid=$(KUBE_NODE_NAME)",
					},
					Env: []v1.EnvVar{
						{
							Name:  "CSI_ENDPOINT",
							Value: "unix://" + "/csi/csi.sock",
						},
						{
							Name: "KUBE_NODE_NAME",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "spec.nodeName",
								},
							},
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "socket-dir",
							MountPath: "/csi",
						},
						{
							Name:             "mountpoint-dir",
							MountPath:        "/var/lib/kubelet/pods",
							MountPropagation: &mountPropagation,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "socket-dir",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/kubelet/plugins/csi-hostpath",
							Type: &hostPathType,
						},
					},
				},
				{
					Name: "registration-dir",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/kubelet/plugins",
							Type: &hostPathType,
						},
					},
				},
				{
					Name: "mountpoint-dir",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/kubelet/pods",
							Type: &hostPathType,
						},
					},
				},
			},
		},
	}

	err := framework.DeletePodWithWait(f, client, pod)
	framework.ExpectNoError(err, "Failed to delete pod %s/%s: %v",
		pod.GetNamespace(), pod.GetName(), err)

	if teardown {
		return nil
	}

	// Creating the pod can fail initially while the service
	// account's secret isn't provisioned yet ('No API token found
	// for service account "csi-service-account", retry after the
	// token is automatically created and added to the service
	// account', see https://github.com/kubernetes/kubernetes/issues/68776).
	// We could use a DaemonSet, but then the name of the csi-pod changes
	// during each test run. It's simpler to just try for a while here.
	podClient := f.PodClient()
	ret := podClient.CreateEventually(pod)

	// Wait for pod to come up
	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(client, ret))
	return ret
}

func deployGCEPDCSIDriver(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	f *framework.Framework,
	nodeSA *v1.ServiceAccount,
	controllerSA *v1.ServiceAccount,
) {
	// Get API Objects from manifests
	nodeds, err := manifest.DaemonSetFromManifest("test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml", config.Namespace)
	framework.ExpectNoError(err, "Failed to create DaemonSet from manifest")
	nodeds.Spec.Template.Spec.ServiceAccountName = nodeSA.GetName()

	controllerss, err := manifest.StatefulSetFromManifest("test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml", config.Namespace)
	framework.ExpectNoError(err, "Failed to create StatefulSet from manifest")
	controllerss.Spec.Template.Spec.ServiceAccountName = controllerSA.GetName()

	controllerservice, err := manifest.SvcFromManifest("test/e2e/testing-manifests/storage-csi/gce-pd/controller_service.yaml")
	framework.ExpectNoError(err, "Failed to create Service from manifest")

	// Got all objects from manifests now try to delete objects
	err = client.CoreV1().Services(config.Namespace).Delete(controllerservice.GetName(), nil)
	if err != nil {
		if !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to delete Service: %v", controllerservice.GetName())
		}
	}

	err = client.AppsV1().StatefulSets(config.Namespace).Delete(controllerss.Name, nil)
	if err != nil {
		if !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to delete StatefulSet: %v", controllerss.GetName())
		}
	}
	err = client.AppsV1().DaemonSets(config.Namespace).Delete(nodeds.Name, nil)
	if err != nil {
		if !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to delete DaemonSet: %v", nodeds.GetName())
		}
	}
	if teardown {
		return
	}

	// Create new API Objects through client
	_, err = client.CoreV1().Services(config.Namespace).Create(controllerservice)
	framework.ExpectNoError(err, "Failed to create Service: %v", controllerservice.Name)

	_, err = client.AppsV1().StatefulSets(config.Namespace).Create(controllerss)
	framework.ExpectNoError(err, "Failed to create StatefulSet: %v", controllerss.Name)

	_, err = client.AppsV1().DaemonSets(config.Namespace).Create(nodeds)
	framework.ExpectNoError(err, "Failed to create DaemonSet: %v", nodeds.Name)

}

func createCSICRDs(c apiextensionsclient.Interface) {
	By("Creating CSI CRDs")
	crds := []*apiextensionsv1beta1.CustomResourceDefinition{
		csicrd.CSIDriverCRD(),
		csicrd.CSINodeInfoCRD(),
	}

	for _, crd := range crds {
		_, err := c.ApiextensionsV1beta1().CustomResourceDefinitions().Get(crd.Name, metav1.GetOptions{})
		if err == nil {
			continue
		} else if !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to check for existing of CSI CRD %q: %v", crd.Name, err)
		}
		_, err = c.ApiextensionsV1beta1().CustomResourceDefinitions().Create(crd)
		framework.ExpectNoError(err, "Failed to create CSI CRD %q: %v", crd.Name, err)
	}
}

func shredFile(filePath string) {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		framework.Logf("File %v was not found, skipping shredding", filePath)
		return
	}
	framework.Logf("Shredding file %v", filePath)
	_, _, err := framework.RunCmd("shred", "--remove", filePath)
	if err != nil {
		framework.Logf("Failed to shred file %v: %v", filePath, err)
	}
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		framework.Logf("File %v successfully shredded", filePath)
		return
	}
	// Shred failed Try to remove the file for good meausure
	err = os.Remove(filePath)
	framework.ExpectNoError(err, "Failed to remove service account file %s", filePath)

}

// createGCESecrets downloads the GCP IAM Key for the default compute service account
// and puts it in a secret for the GCE PD CSI Driver to consume
func createGCESecrets(client clientset.Interface, config framework.VolumeTestConfig) {
	saEnv := "E2E_GOOGLE_APPLICATION_CREDENTIALS"
	saFile := fmt.Sprintf("/tmp/%s/cloud-sa.json", string(uuid.NewUUID()))

	os.MkdirAll(path.Dir(saFile), 0750)
	defer os.Remove(path.Dir(saFile))

	premadeSAFile, ok := os.LookupEnv(saEnv)
	if !ok {
		framework.Logf("Could not find env var %v, please either create cloud-sa"+
			" secret manually or rerun test after setting %v to the filepath of"+
			" the GCP Service Account to give to the GCE Persistent Disk CSI Driver", saEnv, saEnv)
		return
	}

	framework.Logf("Found CI service account key at %v", premadeSAFile)
	// Need to copy it saFile
	stdout, stderr, err := framework.RunCmd("cp", premadeSAFile, saFile)
	framework.ExpectNoError(err, "error copying service account key: %s\nstdout: %s\nstderr: %s", err, stdout, stderr)
	defer shredFile(saFile)
	// Create Secret with this Service Account
	fileBytes, err := ioutil.ReadFile(saFile)
	framework.ExpectNoError(err, "Failed to read file %v", saFile)

	s := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cloud-sa",
			Namespace: config.Namespace,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			filepath.Base(saFile): fileBytes,
		},
	}

	_, err = client.CoreV1().Secrets(config.Namespace).Create(s)
	framework.ExpectNoError(err, "Failed to create Secret %v", s.GetName())
}
