/*
Copyright 2017 The Kubernetes Authors.

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

package etcd

import (
	"errors"
	"fmt"
	"path"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd/spec"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const crdFailureThreshold = 5

// DeployEtcdOperator deploys the etcd-operator along with appropriate RBAC resources
func DeployEtcdOperator(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface, waiter apiclient.Waiter) error {
	start := time.Now()

	clusterRole := getEtcdClusterRole()
	if _, err := client.RbacV1beta1().ClusterRoles().Create(&clusterRole); err != nil {
		return fmt.Errorf("[self-hosted] Failed to create etcd-operator ClusterRole [%v]", err)
	}

	serviceAccount := getEtcdServiceAccount()
	if _, err := client.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&serviceAccount); err != nil {
		return fmt.Errorf("[self-hosted] Failed to create etcd-operator ServiceAccount [%v]", err)
	}

	clusterRoleBinding := getEtcdClusterRoleBinding()
	if _, err := client.RbacV1beta1().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return fmt.Errorf("[self-hosted] Failed to create etcd-operator ClusterRoleBinding [%v]", err)
	}

	etcdOperatorDep := getEtcdOperatorDeployment(cfg)
	if _, err := client.Extensions().Deployments(metav1.NamespaceSystem).Create(&etcdOperatorDep); err != nil {
		return fmt.Errorf("[self-hosted] Failed to create etcd-operator deployment [%v]", err)
	}

	waiter.WaitForPodsWithLabel(fmt.Sprintf("%s=%s", operatorLabelKey, kubeadmconstants.EtcdOperator))
	fmt.Printf("[self-hosted] etcd-operator deployment ready after %f seconds\n", time.Since(start).Seconds())

	return nil
}

// SetupEtcdCluster will perform the following steps:
// 1. two client secrets are created to store TLS assets: one for the operator and apiserver
// 2. a CRD spec for the etcd cluster is sent to the apiserver. the operator will then begin provisioning
//    the cluster behind the scenes and begin pivotting data from the bootstrap member
// 3. wait for the CRD spec to exist
func SetupEtcdCluster(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	start := time.Now()

	fmt.Println("[self-hosted] Create secrets to hold etcd TLS certs")
	if err := createTLSSecrets(cfg, client); err != nil {
		return fmt.Errorf("[self-hosted] Failed to create secrets for etcd cluster [%v]", err)
	}

	// setup CRD clients. crdDataClient is used to create CRD data objects (EtcdClusters), crdClient
	// is for creating the CRD itself.
	crdDataClient, crdClient, err := getCRDClients()
	if err != nil {
		return err
	}

	// CRD needs to exist before a cluster is created
	fmt.Println("[self-hosted] Waiting for `EtcdCluster` CRD to exist")
	waitForCRDToExist(crdClient)

	// send CRD data for our etcd cluster
	etcdClusterData := getEtcdCluster(cfg)
	if err := crdDataClient.Post().
		Resource(spec.CRDResourcePlural).
		Namespace(metav1.NamespaceSystem).
		Body(etcdClusterData).
		Do().
		Error(); err != nil {
		return fmt.Errorf("[self-hosted] API server rejected CRD call: %v", err)
	}

	// wait for cluster to exist
	fmt.Println("[self-hosted] Verifying CRD data exists")
	if err != waitForCRDData(crdDataClient) {
		return err
	}

	time.Sleep(2 * time.Second)

	fmt.Printf("[self-hosted] Self-hosted etcd ready after %f seconds\n", time.Since(start).Seconds())
	return nil
}

func getCRDClients() (*rest.RESTClient, apiextensionsclient.Interface, error) {
	kubeConfigPath := path.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName)
	config, err := clientcmd.BuildConfigFromFlags("", kubeConfigPath)
	if err != nil {
		return nil, nil, err
	}

	scheme := runtime.NewScheme()
	if err := spec.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}

	config.GroupVersion = &spec.SchemeGroupVersion
	config.APIPath = "/apis"
	config.ContentType = runtime.ContentTypeJSON
	config.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: serializer.NewCodecFactory(scheme)}

	restcli, err := rest.RESTClientFor(config)
	if err != nil {
		return nil, nil, err
	}

	apiextsclient, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		return nil, nil, err
	}

	return restcli, apiextsclient, nil
}

func waitForCRDToExist(client apiextensionsclient.Interface) error {
	return apiclient.TryRunCommand(func() error {
		crd, err := client.ApiextensionsV1beta1().CustomResourceDefinitions().Get(spec.CRDName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		for _, cond := range crd.Status.Conditions {
			switch cond.Type {
			case apiextensionsv1beta1.Established:
				if cond.Status == apiextensionsv1beta1.ConditionTrue {
					return nil
				}
			case apiextensionsv1beta1.NamesAccepted:
				if cond.Status == apiextensionsv1beta1.ConditionFalse {
					return fmt.Errorf("[self-hosted] name conflict: %v", cond.Reason)
				}
			}
		}
		return fmt.Errorf("[self-hosted] %s CRD does not exist", spec.CRDName)
	}, crdFailureThreshold)
}

func waitForCRDData(client *rest.RESTClient) error {
	return apiclient.TryRunCommand(func() error {
		cluster := &spec.EtcdCluster{}
		if err := client.Get().
			Resource(spec.CRDResourcePlural).
			Namespace(metav1.NamespaceSystem).
			Name(kubeadmconstants.EtcdClusterName).
			Do().Into(cluster); err != nil {
			return err
		}

		switch cluster.Status.Phase {
		case "Running":
			return nil
		case "Failed":
			return errors.New("[self-hosted] Failed to create etcd cluster")
		default:
			return fmt.Errorf("[self-hosted] Could not find %s EtcdCluster", kubeadmconstants.EtcdClusterName)
		}
	}, crdFailureThreshold)
}
