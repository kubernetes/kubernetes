/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package bridge

import (
	"fmt"
	etcd "github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"hash/fnv"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/restclient"
	kclient "k8s.io/kubernetes/pkg/client/unversioned"
	kclientcmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	kselector "k8s.io/kubernetes/pkg/fields"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	"k8s.io/kubernetes/pkg/util/wait"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	// Maximum number of attempts to connect to etcd server.
	maxConnectAttempts = 12
	// The name of the "master" Kubernetes Service.
	kubernetesSvcName = "kubernetes"
)

func BuildDNSNameString(labels ...string) string {
	var res string
	for _, label := range labels {
		if res == "" {
			res = label
		} else {
			res = fmt.Sprintf("%s.%s", label, res)
		}
	}
	return res
}

func BuildPortSegmentString(portName string, portProtocol kapi.Protocol) string {
	if portName == "" {
		// we don't create a random name
		return ""
	}

	if portProtocol == "" {
		glog.Errorf("Port Protocol not set. port segment string cannot be created.")
		return ""
	}

	return fmt.Sprintf("_%s._%s", portName, strings.ToLower(string(portProtocol)))
}

func expandKubeMasterURL(argKubeMasterURL *string) (string, error) {
	parsedURL, err := url.Parse(os.ExpandEnv(*argKubeMasterURL))
	if err != nil {
		return "", fmt.Errorf("failed to parse --kube-master-url %s - %v", *argKubeMasterURL, err)
	}
	if parsedURL.Scheme == "" || parsedURL.Host == "" || parsedURL.Host == ":" {
		return "", fmt.Errorf("invalid --kube-master-url specified %s", *argKubeMasterURL)
	}
	return parsedURL.String(), nil
}

func SanitizeIP(ip string) string {
	return strings.Replace(ip, ".", "-", -1)
}

func GetHash(text string) string {
	h := fnv.New32a()
	h.Write([]byte(text))
	return fmt.Sprintf("%x", h.Sum32())
}

// TODO: evaluate using pkg/client/clientcmd
func NewKubeClient(argKubeMasterURL *string, argKubecfgFile *string) (*kclient.Client, error) {
	var (
		config    *restclient.Config
		err       error
		masterURL string
	)
	// If the user specified --kube-master-url, expand env vars and verify it.
	if *argKubeMasterURL != "" {
		masterURL, err = expandKubeMasterURL(argKubeMasterURL)
		if err != nil {
			return nil, err
		}
	}

	if masterURL != "" && *argKubecfgFile == "" {
		// Only --kube-master-url was provided.
		config = &restclient.Config{
			Host:          masterURL,
			ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}},
		}
	} else {
		// We either have:
		//  1) --kube-master-url and --kubecfg-file
		//  2) just --kubecfg-file
		//  3) neither flag
		// In any case, the logic is the same.  If (3), this will automatically
		// fall back on the service account token.
		overrides := &kclientcmd.ConfigOverrides{}
		overrides.ClusterInfo.Server = masterURL                                     // might be "", but that is OK
		rules := &kclientcmd.ClientConfigLoadingRules{ExplicitPath: *argKubecfgFile} // might be "", but that is OK
		if config, err = kclientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides).ClientConfig(); err != nil {
			return nil, err
		}
	}

	glog.Infof("Using %s for kubernetes master", config.Host)
	glog.Infof("Using kubernetes API %v", config.GroupVersion)
	return kclient.New(config)
}

func NewEtcdClient(etcdServer string) (*etcd.Client, error) {
	var (
		client *etcd.Client
		err    error
	)
	for attempt := 1; attempt <= maxConnectAttempts; attempt++ {
		if _, err = etcdutil.GetEtcdVersion(etcdServer); err == nil {
			break
		}
		if attempt == maxConnectAttempts {
			break
		}
		glog.Infof("[Attempt: %d] Attempting access to etcd after 5 second sleep", attempt)
		time.Sleep(5 * time.Second)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to connect to etcd server: %v, error: %v", etcdServer, err)
	}
	glog.Infof("Etcd server found: %v", etcdServer)

	// loop until we have > 0 machines && machines[0] != ""
	poll, timeout := 1*time.Second, 10*time.Second
	if err := wait.Poll(poll, timeout, func() (bool, error) {
		if client = etcd.NewClient([]string{etcdServer}); client == nil {
			return false, fmt.Errorf("etcd.NewClient returned nil")
		}
		client.SyncCluster()
		machines := client.GetCluster()
		if len(machines) == 0 || len(machines[0]) == 0 {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, fmt.Errorf("Timed out after %s waiting for at least 1 synchronized etcd server in the cluster. Error: %v", timeout, err)
	}
	return client, nil
}

// Returns a cache.ListWatch that gets all changes to services.
func CreateServiceLW(kubeClient *kclient.Client) *kcache.ListWatch {
	return kcache.NewListWatchFromClient(kubeClient, "services", kapi.NamespaceAll, kselector.Everything())
}

// Returns a cache.ListWatch that gets all changes to endpoints.
func CreateEndpointsLW(kubeClient *kclient.Client) *kcache.ListWatch {
	return kcache.NewListWatchFromClient(kubeClient, "endpoints", kapi.NamespaceAll, kselector.Everything())
}

// Returns a cache.ListWatch that gets all changes to pods.
func CreateEndpointsPodLW(kubeClient *kclient.Client) *kcache.ListWatch {
	return kcache.NewListWatchFromClient(kubeClient, "pods", kapi.NamespaceAll, kselector.Everything())
}

// waitForKubernetesService waits for the "Kuberntes" master service.
// Since the health probe on the kube2sky container is essentially an nslookup
// of this service, we cannot serve any DNS records if it doesn't show up.
// Once the Service is found, we start replying on this containers readiness
// probe endpoint.
func WaitForKubernetesService(client *kclient.Client) (svc *kapi.Service) {
	name := fmt.Sprintf("%v/%v", kapi.NamespaceDefault, kubernetesSvcName)
	glog.Infof("Waiting for service: %v", name)
	var err error
	servicePollInterval := 1 * time.Second
	for {
		svc, err = client.Services(kapi.NamespaceDefault).Get(kubernetesSvcName)
		if err != nil || svc == nil {
			glog.Infof("Ignoring error while waiting for service %v: %v. Sleeping %v before retrying.", name, err, servicePollInterval)
			time.Sleep(servicePollInterval)
			continue
		}
		break
	}
	return
}
