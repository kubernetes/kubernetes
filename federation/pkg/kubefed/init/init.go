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

// TODO(madhusdancs):
// 1. Make printSuccess prepend protocol/scheme to the IPs/hostnames.
// 1. Add a dry-run support.
// 2. Make all the API object names customizable.
//    Ex: federation-apiserver, federation-controller-manager, etc.
// 3. Make image name and tag customizable.
// 4. Separate etcd container from API server pod as a first step towards enabling HA.
// 5. Generate credentials of the following types for the API server:
//    i.  "known_tokens.csv"
//    ii. "basic_auth.csv"
// 6. Add the ability to customize DNS domain suffix. It should probably be derived
//    from cluster config.
// 7. Make etcd PVC size configurable.
// 8. Make API server and controller manager replicas customizable via the HA work.
package init

import (
	"fmt"
	"io"
	"strings"
	"time"

	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	triple "k8s.io/kubernetes/pkg/util/cert/triple"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"

	"github.com/spf13/cobra"
)

const (
	APIServerCN                 = "federation-apiserver"
	ControllerManagerCN         = "federation-controller-manager"
	AdminCN                     = "admin"
	HostClusterLocalDNSZoneName = "cluster.local."

	lbAddrRetryInterval = 5 * time.Second
)

var (
	init_long = templates.LongDesc(`
		Initialize a federation control plane.

        Federation control plane is hosted inside a Kubernetes
        cluster. The host cluster must be specified using the
        --host-cluster-context flag.`)
	init_example = templates.Examples(`
		# Initialize federation control plane for a federation
		# named foo in the host cluster whose local kubeconfig
		# context is bar.
		kubectl init foo --host-cluster-context=bar`)

	componentLabel = map[string]string{
		"app": "federated-cluster",
	}

	apiserverSvcSelector = map[string]string{
		"app":    "federated-cluster",
		"module": "federation-apiserver",
	}

	apiserverPodLabels = map[string]string{
		"app":    "federated-cluster",
		"module": "federation-apiserver",
	}

	controllerManagerPodLabels = map[string]string{
		"app":    "federated-cluster",
		"module": "federation-controller-manager",
	}

	hyperkubeImageName = "gcr.io/google_containers/hyperkube-amd64"
)

// NewCmdInit defines the `init` command that bootstraps a federation
// control plane inside a set of host clusters.
func NewCmdInit(cmdOut io.Writer, config util.AdminConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "init FEDERATION_NAME --host-cluster-context=HOST_CONTEXT",
		Short:   "init initializes a federation control plane",
		Long:    init_long,
		Example: init_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := initFederation(cmdOut, config, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	defaultImage := fmt.Sprintf("%s:%s", hyperkubeImageName, version.Get())

	util.AddSubcommandFlags(cmd)
	cmd.Flags().String("dns-zone-name", "", "DNS suffix for this federation. Federated Service DNS names are published with this suffix.")
	cmd.Flags().String("image", defaultImage, "Image to use for federation API server and controller manager binaries.")
	cmd.Flags().String("dns-provider", "google-clouddns", "Dns provider to be used for this deployment.")
	return cmd
}

type entityKeyPairs struct {
	ca                *triple.KeyPair
	server            *triple.KeyPair
	controllerManager *triple.KeyPair
	admin             *triple.KeyPair
}

// initFederation initializes a federation control plane.
// See the design doc in https://github.com/kubernetes/kubernetes/pull/34484
// for details.
func initFederation(cmdOut io.Writer, config util.AdminConfig, cmd *cobra.Command, args []string) error {
	initFlags, err := util.GetSubcommandFlags(cmd, args)
	if err != nil {
		return err
	}
	dnsZoneName := cmdutil.GetFlagString(cmd, "dns-zone-name")
	image := cmdutil.GetFlagString(cmd, "image")
	dnsProvider := cmdutil.GetFlagString(cmd, "dns-provider")

	hostFactory := config.HostFactory(initFlags.Host, initFlags.Kubeconfig)
	hostClientset, err := hostFactory.ClientSet()
	if err != nil {
		return err
	}

	serverName := fmt.Sprintf("%s-apiserver", initFlags.Name)
	serverCredName := fmt.Sprintf("%s-credentials", serverName)
	cmName := fmt.Sprintf("%s-controller-manager", initFlags.Name)
	cmKubeconfigName := fmt.Sprintf("%s-kubeconfig", cmName)

	// 1. Create a namespace for federation system components
	_, err = createNamespace(hostClientset, initFlags.FederationSystemNamespace)
	if err != nil {
		return err
	}

	// 2. Expose a network endpoint for the federation API server
	svc, err := createService(hostClientset, initFlags.FederationSystemNamespace, serverName)
	if err != nil {
		return err
	}
	ips, hostnames, err := waitForLoadBalancerAddress(hostClientset, svc)
	if err != nil {
		return err
	}

	// 3. Generate TLS certificates and credentials
	entKeyPairs, err := genCerts(initFlags.FederationSystemNamespace, initFlags.Name, svc.Name, HostClusterLocalDNSZoneName, ips, hostnames)
	if err != nil {
		return err
	}

	_, err = createAPIServerCredentialsSecret(hostClientset, initFlags.FederationSystemNamespace, serverCredName, entKeyPairs)
	if err != nil {
		return err
	}

	// 4. Create a kubeconfig secret
	_, err = createControllerManagerKubeconfigSecret(hostClientset, initFlags.FederationSystemNamespace, initFlags.Name, svc.Name, cmKubeconfigName, entKeyPairs)
	if err != nil {
		return err
	}

	// 5. Create a persistent volume and a claim to store the federation
	// API server's state. This is where federation API server's etcd
	// stores its data.
	pvc, err := createPVC(hostClientset, initFlags.FederationSystemNamespace, svc.Name)
	if err != nil {
		return err
	}

	// Since only one IP address can be specified as advertise address,
	// we arbitrarily pick the first availabe IP address
	advertiseAddress := ""
	if len(ips) > 0 {
		advertiseAddress = ips[0]
	}

	endpoint := advertiseAddress
	if advertiseAddress == "" && len(hostnames) > 0 {
		endpoint = hostnames[0]
	}

	// 6. Create federation API server
	_, err = createAPIServer(hostClientset, initFlags.FederationSystemNamespace, serverName, image, serverCredName, pvc.Name, advertiseAddress)
	if err != nil {
		return err
	}

	// 7. Create federation controller manager
	_, err = createControllerManager(hostClientset, initFlags.FederationSystemNamespace, initFlags.Name, svc.Name, cmName, image, cmKubeconfigName, dnsZoneName, dnsProvider)
	if err != nil {
		return err
	}

	// 8. Write the federation API server endpoint info, credentials
	// and context to kubeconfig
	err = updateKubeconfig(config, initFlags.Name, endpoint, entKeyPairs)
	if err != nil {
		return err
	}

	return printSuccess(cmdOut, ips, hostnames)
}

func createNamespace(clientset *client.Clientset, namespace string) (*api.Namespace, error) {
	ns := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name: namespace,
		},
	}

	return clientset.Core().Namespaces().Create(ns)
}

func createService(clientset *client.Clientset, namespace, svcName string) (*api.Service, error) {
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      svcName,
			Namespace: namespace,
			Labels:    componentLabel,
		},
		Spec: api.ServiceSpec{
			Type:     api.ServiceTypeLoadBalancer,
			Selector: apiserverSvcSelector,
			Ports: []api.ServicePort{
				{
					Name:       "https",
					Protocol:   "TCP",
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}

	return clientset.Core().Services(namespace).Create(svc)
}

func waitForLoadBalancerAddress(clientset *client.Clientset, svc *api.Service) ([]string, []string, error) {
	ips := []string{}
	hostnames := []string{}

	err := wait.PollImmediateInfinite(lbAddrRetryInterval, func() (bool, error) {
		pollSvc, err := clientset.Core().Services(svc.Namespace).Get(svc.Name)
		if err != nil {
			return false, nil
		}
		if ings := pollSvc.Status.LoadBalancer.Ingress; len(ings) > 0 {
			for _, ing := range ings {
				if len(ing.IP) > 0 {
					ips = append(ips, ing.IP)
				}
				if len(ing.Hostname) > 0 {
					hostnames = append(hostnames, ing.Hostname)
				}
			}
			if len(ips) > 0 || len(hostnames) > 0 {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		return nil, nil, err
	}

	return ips, hostnames, nil
}

func genCerts(svcNamespace, name, svcName, localDNSZoneName string, ips, hostnames []string) (*entityKeyPairs, error) {
	ca, err := triple.NewCA(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create CA key and certificate: %v", err)
	}
	server, err := triple.NewServerKeyPair(ca, APIServerCN, svcName, svcNamespace, localDNSZoneName, ips, hostnames)
	if err != nil {
		return nil, fmt.Errorf("failed to create federation API server key and certificate: %v", err)
	}
	cm, err := triple.NewClientKeyPair(ca, ControllerManagerCN)
	if err != nil {
		return nil, fmt.Errorf("failed to create federation controller manager client key and certificate: %v", err)
	}
	admin, err := triple.NewClientKeyPair(ca, AdminCN)
	if err != nil {
		return nil, fmt.Errorf("failed to create client key and certificate for an admin: %v", err)
	}
	return &entityKeyPairs{
		ca:                ca,
		server:            server,
		controllerManager: cm,
		admin:             admin,
	}, nil
}

func createAPIServerCredentialsSecret(clientset *client.Clientset, namespace, credentialsName string, entKeyPairs *entityKeyPairs) (*api.Secret, error) {
	// Build the secret object with API server credentials.
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      credentialsName,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			"ca.crt":     certutil.EncodeCertPEM(entKeyPairs.ca.Cert),
			"server.crt": certutil.EncodeCertPEM(entKeyPairs.server.Cert),
			"server.key": certutil.EncodePrivateKeyPEM(entKeyPairs.server.Key),
		},
	}

	// Boilerplate to create the secret in the host cluster.
	return clientset.Core().Secrets(namespace).Create(secret)
}

func createControllerManagerKubeconfigSecret(clientset *client.Clientset, namespace, name, svcName, kubeconfigName string, entKeyPairs *entityKeyPairs) (*api.Secret, error) {
	basicClientConfig := kubeadmutil.CreateBasicClientConfig(
		name,
		fmt.Sprintf("https://%s", svcName),
		certutil.EncodeCertPEM(entKeyPairs.ca.Cert),
	)

	config := kubeadmutil.MakeClientConfigWithCerts(
		basicClientConfig,
		name,
		"federation-controller-manager",
		certutil.EncodePrivateKeyPEM(entKeyPairs.controllerManager.Key),
		certutil.EncodeCertPEM(entKeyPairs.controllerManager.Cert),
	)

	return util.CreateKubeconfigSecret(clientset, config, namespace, kubeconfigName, false)
}

func createPVC(clientset *client.Clientset, namespace, svcName string) (*api.PersistentVolumeClaim, error) {
	capacity, err := resource.ParseQuantity("10Gi")
	if err != nil {
		return nil, err
	}

	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%s-etcd-claim", svcName),
			Namespace: namespace,
			Labels:    componentLabel,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "yes",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceStorage: capacity,
				},
			},
		},
	}

	return clientset.Core().PersistentVolumeClaims(namespace).Create(pvc)
}

func createAPIServer(clientset *client.Clientset, namespace, name, image, credentialsName, pvcName, advertiseAddress string) (*extensions.Deployment, error) {
	command := []string{
		"/hyperkube",
		"federation-apiserver",
		"--bind-address=0.0.0.0",
		"--etcd-servers=http://localhost:2379",
		"--service-cluster-ip-range=10.0.0.0/16",
		"--secure-port=443",
		"--client-ca-file=/etc/federation/apiserver/ca.crt",
		"--tls-cert-file=/etc/federation/apiserver/server.crt",
		"--tls-private-key-file=/etc/federation/apiserver/server.key",
	}

	if advertiseAddress != "" {
		command = append(command, fmt.Sprintf("--advertise-address=%s", advertiseAddress))
	}

	dataVolumeName := "etcddata"

	dep := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    componentLabel,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 1,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   name,
					Labels: apiserverPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:    "apiserver",
							Image:   image,
							Command: command,
							Ports: []api.ContainerPort{
								{
									Name:          "https",
									ContainerPort: 443,
								},
								{
									Name:          "local",
									ContainerPort: 8080,
								},
							},
							VolumeMounts: []api.VolumeMount{
								{
									Name:      credentialsName,
									MountPath: "/etc/federation/apiserver",
									ReadOnly:  true,
								},
							},
						},
						{
							Name:  "etcd",
							Image: "quay.io/coreos/etcd:v2.3.3",
							Command: []string{
								"/etcd",
								"--data-dir",
								"/var/etcd/data",
							},
							VolumeMounts: []api.VolumeMount{
								{
									Name:      dataVolumeName,
									MountPath: "/var/etcd",
								},
							},
						},
					},
					Volumes: []api.Volume{
						{
							Name: credentialsName,
							VolumeSource: api.VolumeSource{
								Secret: &api.SecretVolumeSource{
									SecretName: credentialsName,
								},
							},
						},
						{
							Name: dataVolumeName,
							VolumeSource: api.VolumeSource{
								PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
									ClaimName: pvcName,
								},
							},
						},
					},
				},
			},
		},
	}

	return clientset.Extensions().Deployments(namespace).Create(dep)
}

func createControllerManager(clientset *client.Clientset, namespace, name, svcName, cmName, image, kubeconfigName, dnsZoneName, dnsProvider string) (*extensions.Deployment, error) {
	dep := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      cmName,
			Namespace: namespace,
			Labels:    componentLabel,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 1,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   cmName,
					Labels: controllerManagerPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "controller-manager",
							Image: image,
							Command: []string{
								"/hyperkube",
								"federation-controller-manager",
								fmt.Sprintf("--master=https://%s", svcName),
								"--kubeconfig=/etc/federation/controller-manager/kubeconfig",
								fmt.Sprintf("--dns-provider=%s", dnsProvider),
								"--dns-provider-config=",
								fmt.Sprintf("--federation-name=%s", name),
								fmt.Sprintf("--zone-name=%s", dnsZoneName),
							},
							VolumeMounts: []api.VolumeMount{
								{
									Name:      kubeconfigName,
									MountPath: "/etc/federation/controller-manager",
									ReadOnly:  true,
								},
							},
							Env: []api.EnvVar{
								{
									Name: "POD_NAMESPACE",
									ValueFrom: &api.EnvVarSource{
										FieldRef: &api.ObjectFieldSelector{
											FieldPath: "metadata.namespace",
										},
									},
								},
							},
						},
					},
					Volumes: []api.Volume{
						{
							Name: kubeconfigName,
							VolumeSource: api.VolumeSource{
								Secret: &api.SecretVolumeSource{
									SecretName: kubeconfigName,
								},
							},
						},
					},
				},
			},
		},
	}

	return clientset.Extensions().Deployments(namespace).Create(dep)
}

func printSuccess(cmdOut io.Writer, ips, hostnames []string) error {
	svcEndpoints := append(ips, hostnames...)
	_, err := fmt.Fprintf(cmdOut, "Federation API server is running at: %s\n", strings.Join(svcEndpoints, ", "))
	return err
}

func updateKubeconfig(config util.AdminConfig, name, endpoint string, entKeyPairs *entityKeyPairs) error {
	po := config.PathOptions()
	kubeconfig, err := po.GetStartingConfig()
	if err != nil {
		return err
	}

	// Populate API server endpoint info.
	cluster := clientcmdapi.NewCluster()
	// Prefix "https" as the URL scheme to endpoint.
	if !strings.HasPrefix(endpoint, "https://") {
		endpoint = fmt.Sprintf("https://%s", endpoint)
	}
	cluster.Server = endpoint
	cluster.CertificateAuthorityData = certutil.EncodeCertPEM(entKeyPairs.ca.Cert)

	// Populate credentials.
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientCertificateData = certutil.EncodeCertPEM(entKeyPairs.admin.Cert)
	authInfo.ClientKeyData = certutil.EncodePrivateKeyPEM(entKeyPairs.admin.Key)
	authInfo.Username = AdminCN

	// Populate context.
	context := clientcmdapi.NewContext()
	context.Cluster = name
	context.AuthInfo = name

	// Update the config struct with API server endpoint info,
	// credentials and context.
	kubeconfig.Clusters[name] = cluster
	kubeconfig.AuthInfos[name] = authInfo
	kubeconfig.Contexts[name] = context

	// Write the update kubeconfig.
	if err := clientcmd.ModifyConfig(po, *kubeconfig, true); err != nil {
		return err
	}

	return nil
}
