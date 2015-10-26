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

package controllermanager

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"strconv"
	"time"

	"k8s.io/kubernetes/cmd/kube-controller-manager/app"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
	"k8s.io/kubernetes/pkg/controller/daemon"
	kendpoint "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/pkg/controller/node"
	"k8s.io/kubernetes/pkg/controller/persistentvolume"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/controller/resourcequota"
	"k8s.io/kubernetes/pkg/controller/route"
	"k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util"

	"k8s.io/kubernetes/contrib/mesos/pkg/profile"
	kmendpoint "k8s.io/kubernetes/contrib/mesos/pkg/service"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	*app.CMServer
	UseHostPortEndpoints bool
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := &CMServer{
		CMServer: app.NewCMServer(),
	}
	s.CloudProvider = mesos.ProviderName
	s.UseHostPortEndpoints = true
	return s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	s.CMServer.AddFlags(fs)
	fs.BoolVar(&s.UseHostPortEndpoints, "host_port_endpoints", s.UseHostPortEndpoints, "Map service endpoints to hostIP:hostPort instead of podIP:containerPort. Default true.")
}

func (s *CMServer) resyncPeriod() time.Duration {
	factor := rand.Float64() + 1
	return time.Duration(float64(time.Hour) * 12.0 * factor)
}

func (s *CMServer) Run(_ []string) error {
	if s.Kubeconfig == "" && s.Master == "" {
		glog.Warningf("Neither --kubeconfig nor --master was specified.  Using default API client.  This might not work.")
	}

	// This creates a client, first loading any specified kubeconfig
	// file, and then overriding the Master flag, if non-empty.
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.Kubeconfig},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.Master}}).ClientConfig()
	if err != nil {
		return err
	}

	kubeconfig.QPS = 20.0
	kubeconfig.Burst = 30

	kubeClient, err := client.New(kubeconfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		if s.EnableProfiling {
			profile.InstallHandler(mux)
		}
		mux.Handle("/metrics", prometheus.Handler())
		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address.String(), strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	endpoints := s.createEndpointController(kubeClient)
	go endpoints.Run(s.ConcurrentEndpointSyncs, util.NeverStop)

	go replicationcontroller.NewReplicationManager(kubeClient, s.resyncPeriod, replicationcontroller.BurstReplicas).
		Run(s.ConcurrentRCSyncs, util.NeverStop)

	go daemon.NewDaemonSetsController(kubeClient, s.resyncPeriod).
		Run(s.ConcurrentDSCSyncs, util.NeverStop)

	//TODO(jdef) should eventually support more cloud providers here
	if s.CloudProvider != mesos.ProviderName {
		glog.Fatalf("Only provider %v is supported, you specified %v", mesos.ProviderName, s.CloudProvider)
	}
	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	nodeController := nodecontroller.NewNodeController(cloud, kubeClient,
		s.PodEvictionTimeout, util.NewTokenBucketRateLimiter(s.DeletingPodsQps, s.DeletingPodsBurst),
		s.NodeMonitorGracePeriod, s.NodeStartupGracePeriod, s.NodeMonitorPeriod, (*net.IPNet)(&s.ClusterCIDR), s.AllocateNodeCIDRs)
	nodeController.Run(s.NodeSyncPeriod)

	serviceController := servicecontroller.New(cloud, kubeClient, s.ClusterName)
	if err := serviceController.Run(s.ServiceSyncPeriod, s.NodeSyncPeriod); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}

	if s.AllocateNodeCIDRs {
		routes, ok := cloud.Routes()
		if !ok {
			glog.Fatal("Cloud provider must support routes if allocate-node-cidrs is set")
		}
		routeController := routecontroller.New(routes, kubeClient, s.ClusterName, (*net.IPNet)(&s.ClusterCIDR))
		routeController.Run(s.NodeSyncPeriod)
	}

	resourceQuotaController := resourcequotacontroller.NewResourceQuotaController(kubeClient)
	resourceQuotaController.Run(s.ResourceQuotaSyncPeriod)

	namespaceController := namespacecontroller.NewNamespaceController(kubeClient, &unversioned.APIVersions{}, s.NamespaceSyncPeriod)
	namespaceController.Run()

	pvclaimBinder := volumeclaimbinder.NewPersistentVolumeClaimBinder(kubeClient, s.PVClaimBinderSyncPeriod)
	pvclaimBinder.Run()
	pvRecycler, err := volumeclaimbinder.NewPersistentVolumeRecycler(kubeClient, s.PVClaimBinderSyncPeriod, app.ProbeRecyclableVolumePlugins(s.VolumeConfigFlags))
	if err != nil {
		glog.Fatalf("Failed to start persistent volume recycler: %+v", err)
	}
	pvRecycler.Run()

	var rootCA []byte

	if s.RootCAFile != "" {
		rootCA, err = ioutil.ReadFile(s.RootCAFile)
		if err != nil {
			return fmt.Errorf("error reading root-ca-file at %s: %v", s.RootCAFile, err)
		}
		if _, err := util.CertsFromPEM(rootCA); err != nil {
			return fmt.Errorf("error parsing root-ca-file at %s: %v", s.RootCAFile, err)
		}
	} else {
		rootCA = kubeconfig.CAData
	}

	if len(s.ServiceAccountKeyFile) > 0 {
		privateKey, err := serviceaccount.ReadPrivateKey(s.ServiceAccountKeyFile)
		if err != nil {
			glog.Errorf("Error reading key for service account token controller: %v", err)
		} else {
			serviceaccount.NewTokensController(
				kubeClient,
				serviceaccount.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run()
		}
	}

	serviceaccount.NewServiceAccountsController(
		kubeClient,
		serviceaccount.DefaultServiceAccountsControllerOptions(),
	).Run()

	select {}
}

func (s *CMServer) createEndpointController(client *client.Client) kmendpoint.EndpointController {
	if s.UseHostPortEndpoints {
		glog.V(2).Infof("Creating hostIP:hostPort endpoint controller")
		return kmendpoint.NewEndpointController(client)
	}
	glog.V(2).Infof("Creating podIP:containerPort endpoint controller")
	stockEndpointController := kendpoint.NewEndpointController(client, s.resyncPeriod)
	return stockEndpointController
}
