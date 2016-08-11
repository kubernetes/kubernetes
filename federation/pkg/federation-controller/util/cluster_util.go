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

package util

import (
	"fmt"
	"net"
	"os"
	"time"

	"github.com/golang/glog"
	federation_v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	KubeAPIQPS              = 20.0
	KubeAPIBurst            = 30
	KubeconfigSecretDataKey = "kubeconfig"
	getSecretTimeout        = 1 * time.Minute
)

func BuildClusterConfig(c *federation_v1beta1.Cluster) (*restclient.Config, error) {
	var serverAddress string
	var clusterConfig *restclient.Config
	hostIP, err := utilnet.ChooseHostInterface()
	if err != nil {
		return nil, err
	}

	for _, item := range c.Spec.ServerAddressByClientCIDRs {
		_, cidrnet, err := net.ParseCIDR(item.ClientCIDR)
		if err != nil {
			return nil, err
		}
		myaddr := net.ParseIP(hostIP.String())
		if cidrnet.Contains(myaddr) == true {
			serverAddress = item.ServerAddress
			break
		}
	}
	if serverAddress != "" {
		if c.Spec.SecretRef == nil {
			glog.Infof("didnt find secretRef for cluster %s. Trying insecure access", c.Name)
			clusterConfig, err = clientcmd.BuildConfigFromFlags(serverAddress, "")
		} else {
			kubeconfigGetter := KubeconfigGetterForCluster(c)
			clusterConfig, err = clientcmd.BuildConfigFromKubeconfigGetter(serverAddress, kubeconfigGetter)
		}
		if err != nil {
			return nil, err
		}
		clusterConfig.QPS = KubeAPIQPS
		clusterConfig.Burst = KubeAPIBurst
	}
	return clusterConfig, nil
}

// This is to inject a different kubeconfigGetter in tests.
// We dont use the standard one which calls NewInCluster in tests to avoid having to setup service accounts and mount files with secret tokens.
var KubeconfigGetterForCluster = func(c *federation_v1beta1.Cluster) clientcmd.KubeconfigGetter {
	return func() (*clientcmdapi.Config, error) {
		secretRefName := ""
		if c.Spec.SecretRef != nil {
			secretRefName = c.Spec.SecretRef.Name
		} else {
			glog.Infof("didnt find secretRef for cluster %s. Trying insecure access", c.Name)
		}
		return KubeconfigGetterForSecret(secretRefName)()
	}
}

// KubeconfigGettterForSecret is used to get the kubeconfig from the given secret.
var KubeconfigGetterForSecret = func(secretName string) clientcmd.KubeconfigGetter {
	return func() (*clientcmdapi.Config, error) {
		var data []byte
		if secretName != "" {
			// Get the namespace this is running in from the env variable.
			namespace := os.Getenv("POD_NAMESPACE")
			if namespace == "" {
				return nil, fmt.Errorf("unexpected: POD_NAMESPACE env var returned empty string")
			}
			// Get a client to talk to the k8s apiserver, to fetch secrets from it.
			client, err := client.NewInCluster()
			if err != nil {
				return nil, fmt.Errorf("error in creating in-cluster client: %s", err)
			}
			data = []byte{}
			var secret *api.Secret
			err = wait.PollImmediate(1*time.Second, getSecretTimeout, func() (bool, error) {
				secret, err = client.Secrets(namespace).Get(secretName)
				if err == nil {
					return true, nil
				}
				glog.Warningf("error in fetching secret: %s", err)
				return false, nil
			})
			if err != nil {
				return nil, fmt.Errorf("timed out waiting for secret: %s", err)
			}
			if secret == nil {
				return nil, fmt.Errorf("unexpected: received null secret %s", secretName)
			}
			ok := false
			data, ok = secret.Data[KubeconfigSecretDataKey]
			if !ok {
				return nil, fmt.Errorf("secret does not have data with key: %s", KubeconfigSecretDataKey)
			}
		}
		return clientcmd.Load(data)
	}
}

// Retruns Clientset for the given cluster.
func GetClientsetForCluster(cluster *federation_v1beta1.Cluster) (*federation_release_1_4.Clientset, error) {
	clusterConfig, err := BuildClusterConfig(cluster)
	if err != nil && clusterConfig != nil {
		clientset := federation_release_1_4.NewForConfigOrDie(restclient.AddUserAgent(clusterConfig, userAgentName))
		return clientset, nil
	}
	return nil, err
}
