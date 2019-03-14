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

package dns

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mholt/caddy/caddyfile"
	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const (
	// KubeDNSServiceAccountName describes the name of the ServiceAccount for the kube-dns addon
	KubeDNSServiceAccountName  = "kube-dns"
	kubeDNSStubDomain          = "stubDomains"
	kubeDNSUpstreamNameservers = "upstreamNameservers"
	kubeDNSFederation          = "federations"
)

// DeployedDNSAddon returns the type of DNS addon currently deployed
func DeployedDNSAddon(client clientset.Interface) (kubeadmapi.DNSAddOnType, string, error) {
	deploymentsClient := client.AppsV1().Deployments(metav1.NamespaceSystem)
	deployments, err := deploymentsClient.List(metav1.ListOptions{LabelSelector: "k8s-app=kube-dns"})
	if err != nil {
		return "", "", errors.Wrap(err, "couldn't retrieve DNS addon deployments")
	}

	switch len(deployments.Items) {
	case 0:
		return "", "", nil
	case 1:
		addonName := deployments.Items[0].Name
		addonType := kubeadmapi.CoreDNS
		if addonName == kubeadmconstants.KubeDNSDeploymentName {
			addonType = kubeadmapi.KubeDNS
		}
		addonImage := deployments.Items[0].Spec.Template.Spec.Containers[0].Image
		addonImageParts := strings.Split(addonImage, ":")
		addonVersion := addonImageParts[len(addonImageParts)-1]
		return addonType, addonVersion, nil
	default:
		return "", "", errors.Errorf("multiple DNS addon deployments found: %v", deployments.Items)
	}
}

// EnsureDNSAddon creates the kube-dns or CoreDNS addon
func EnsureDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	if cfg.DNS.Type == kubeadmapi.CoreDNS {
		return coreDNSAddon(cfg, client)
	}
	return kubeDNSAddon(cfg, client)
}

func kubeDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	if err := CreateServiceAccount(client); err != nil {
		return err
	}

	dnsip, err := kubeadmconstants.GetDNSIP(cfg.Networking.ServiceSubnet)
	if err != nil {
		return err
	}

	var dnsBindAddr, dnsProbeAddr string
	if dnsip.To4() == nil {
		dnsBindAddr = "::1"
		dnsProbeAddr = "[" + dnsBindAddr + "]"
	} else {
		dnsBindAddr = "127.0.0.1"
		dnsProbeAddr = dnsBindAddr
	}

	dnsDeploymentBytes, err := kubeadmutil.ParseTemplate(KubeDNSDeployment,
		struct{ DeploymentName, KubeDNSImage, DNSMasqImage, SidecarImage, DNSBindAddr, DNSProbeAddr, DNSDomain, ControlPlaneTaintKey string }{
			DeploymentName:       kubeadmconstants.KubeDNSDeploymentName,
			KubeDNSImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSKubeDNSImageName),
			DNSMasqImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSDnsMasqNannyImageName),
			SidecarImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSSidecarImageName),
			DNSBindAddr:          dnsBindAddr,
			DNSProbeAddr:         dnsProbeAddr,
			DNSDomain:            cfg.Networking.DNSDomain,
			ControlPlaneTaintKey: kubeadmconstants.LabelNodeRoleMaster,
		})
	if err != nil {
		return errors.Wrap(err, "error when parsing kube-dns deployment template")
	}

	dnsServiceBytes, err := kubeadmutil.ParseTemplate(KubeDNSService, struct{ DNSIP string }{
		DNSIP: dnsip.String(),
	})
	if err != nil {
		return errors.Wrap(err, "error when parsing kube-proxy configmap template")
	}

	if err := createKubeDNSAddon(dnsDeploymentBytes, dnsServiceBytes, client); err != nil {
		return err
	}
	fmt.Println("[addons] Applied essential addon: kube-dns")
	return nil
}

// CreateServiceAccount creates the necessary serviceaccounts that kubeadm uses/might use, if they don't already exist.
func CreateServiceAccount(client clientset.Interface) error {

	return apiclient.CreateOrUpdateServiceAccount(client, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeDNSServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	})
}

func createKubeDNSAddon(deploymentBytes, serviceBytes []byte, client clientset.Interface) error {
	kubednsDeployment := &apps.Deployment{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), deploymentBytes, kubednsDeployment); err != nil {
		return errors.Wrap(err, "unable to decode kube-dns deployment")
	}

	// Create the Deployment for kube-dns or update it in case it already exists
	if err := apiclient.CreateOrUpdateDeployment(client, kubednsDeployment); err != nil {
		return err
	}

	kubednsService := &v1.Service{}
	return createDNSService(kubednsService, serviceBytes, client)
}

func coreDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	// Get the YAML manifest
	coreDNSDeploymentBytes, err := kubeadmutil.ParseTemplate(CoreDNSDeployment, struct{ DeploymentName, Image, ControlPlaneTaintKey string }{
		DeploymentName:       kubeadmconstants.CoreDNSDeploymentName,
		Image:                images.GetDNSImage(cfg, kubeadmconstants.CoreDNSImageName),
		ControlPlaneTaintKey: kubeadmconstants.LabelNodeRoleMaster,
	})
	if err != nil {
		return errors.Wrap(err, "error when parsing CoreDNS deployment template")
	}

	// Get the kube-dns ConfigMap for translation to equivalent CoreDNS Config.
	kubeDNSConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeDNSConfigMap, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	stubDomain, err := translateStubDomainOfKubeDNSToForwardCoreDNS(kubeDNSStubDomain, kubeDNSConfigMap)
	if err != nil {
		return err
	}

	upstreamNameserver, err := translateUpstreamNameServerOfKubeDNSToUpstreamProxyCoreDNS(kubeDNSUpstreamNameservers, kubeDNSConfigMap)
	if err != nil {
		return err
	}
	coreDNSDomain := cfg.Networking.DNSDomain
	federations, err := translateFederationsofKubeDNSToCoreDNS(kubeDNSFederation, coreDNSDomain, kubeDNSConfigMap)
	if err != nil {
		return err
	}

	// Get the config file for CoreDNS
	coreDNSConfigMapBytes, err := kubeadmutil.ParseTemplate(CoreDNSConfigMap, struct{ DNSDomain, UpstreamNameserver, Federation, StubDomain string }{
		DNSDomain:          coreDNSDomain,
		UpstreamNameserver: upstreamNameserver,
		Federation:         federations,
		StubDomain:         stubDomain,
	})
	if err != nil {
		return errors.Wrap(err, "error when parsing CoreDNS configMap template")
	}

	dnsip, err := kubeadmconstants.GetDNSIP(cfg.Networking.ServiceSubnet)
	if err != nil {
		return err
	}

	coreDNSServiceBytes, err := kubeadmutil.ParseTemplate(KubeDNSService, struct{ DNSIP string }{
		DNSIP: dnsip.String(),
	})

	if err != nil {
		return errors.Wrap(err, "error when parsing CoreDNS service template")
	}

	if err := createCoreDNSAddon(coreDNSDeploymentBytes, coreDNSServiceBytes, coreDNSConfigMapBytes, client); err != nil {
		return err
	}
	fmt.Println("[addons] Applied essential addon: CoreDNS")
	return nil
}

func createCoreDNSAddon(deploymentBytes, serviceBytes, configBytes []byte, client clientset.Interface) error {
	coreDNSConfigMap := &v1.ConfigMap{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), configBytes, coreDNSConfigMap); err != nil {
		return errors.Wrap(err, "unable to decode CoreDNS configmap")
	}

	// Create the ConfigMap for CoreDNS or retain it in case it already exists
	if err := apiclient.CreateOrRetainConfigMap(client, coreDNSConfigMap, kubeadmconstants.CoreDNSConfigMap); err != nil {
		return err
	}

	coreDNSClusterRoles := &rbac.ClusterRole{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSClusterRole), coreDNSClusterRoles); err != nil {
		return errors.Wrap(err, "unable to decode CoreDNS clusterroles")
	}

	// Create the Clusterroles for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateClusterRole(client, coreDNSClusterRoles); err != nil {
		return err
	}

	coreDNSClusterRolesBinding := &rbac.ClusterRoleBinding{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSClusterRoleBinding), coreDNSClusterRolesBinding); err != nil {
		return errors.Wrap(err, "unable to decode CoreDNS clusterrolebindings")
	}

	// Create the Clusterrolebindings for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateClusterRoleBinding(client, coreDNSClusterRolesBinding); err != nil {
		return err
	}

	coreDNSServiceAccount := &v1.ServiceAccount{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSServiceAccount), coreDNSServiceAccount); err != nil {
		return errors.Wrap(err, "unable to decode CoreDNS serviceaccount")
	}

	// Create the ConfigMap for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateServiceAccount(client, coreDNSServiceAccount); err != nil {
		return err
	}

	coreDNSDeployment := &apps.Deployment{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), deploymentBytes, coreDNSDeployment); err != nil {
		return errors.Wrap(err, "unable to decode CoreDNS deployment")
	}

	// Create the Deployment for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateDeployment(client, coreDNSDeployment); err != nil {
		return err
	}

	coreDNSService := &v1.Service{}
	return createDNSService(coreDNSService, serviceBytes, client)
}

func createDNSService(dnsService *v1.Service, serviceBytes []byte, client clientset.Interface) error {
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), serviceBytes, dnsService); err != nil {
		return errors.Wrap(err, "unable to decode the DNS service")
	}

	// Can't use a generic apiclient helper func here as we have to tolerate more than AlreadyExists.
	if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Create(dnsService); err != nil {
		// Ignore if the Service is invalid with this error message:
		// 	Service "kube-dns" is invalid: spec.clusterIP: Invalid value: "10.96.0.10": provided IP is already allocated

		if !apierrors.IsAlreadyExists(err) && !apierrors.IsInvalid(err) {
			return errors.Wrap(err, "unable to create a new DNS service")
		}

		if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Update(dnsService); err != nil {
			return errors.Wrap(err, "unable to create/update the DNS service")
		}
	}
	return nil
}

// translateStubDomainOfKubeDNSToForwardCoreDNS translates StubDomain Data in kube-dns ConfigMap
// in the form of Proxy for the CoreDNS Corefile.
func translateStubDomainOfKubeDNSToForwardCoreDNS(dataField string, kubeDNSConfigMap *v1.ConfigMap) (string, error) {
	if kubeDNSConfigMap == nil {
		return "", nil
	}

	if proxy, ok := kubeDNSConfigMap.Data[dataField]; ok {
		stubDomainData := make(map[string][]string)
		err := json.Unmarshal([]byte(proxy), &stubDomainData)
		if err != nil {
			return "", errors.Wrap(err, "failed to parse JSON from 'kube-dns ConfigMap")
		}

		var proxyStanza []interface{}
		for domain, proxyIP := range stubDomainData {
			pStanza := map[string]interface{}{}
			pStanza["keys"] = []string{domain + ":53"}
			pStanza["body"] = [][]string{
				{"errors"},
				{"cache", "30"},
				{"loop"},
				append([]string{"forward", "."}, proxyIP...),
			}
			proxyStanza = append(proxyStanza, pStanza)
		}
		stanzasBytes, err := json.Marshal(proxyStanza)
		if err != nil {
			return "", err
		}

		corefileStanza, err := caddyfile.FromJSON(stanzasBytes)
		if err != nil {
			return "", err
		}

		return prepCorefileFormat(string(corefileStanza), 4), nil
	}
	return "", nil
}

// translateUpstreamNameServerOfKubeDNSToUpstreamProxyCoreDNS translates UpstreamNameServer Data in kube-dns ConfigMap
// in the form of Proxy for the CoreDNS Corefile.
func translateUpstreamNameServerOfKubeDNSToUpstreamProxyCoreDNS(dataField string, kubeDNSConfigMap *v1.ConfigMap) (string, error) {
	if kubeDNSConfigMap == nil {
		return "", nil
	}

	if upstreamValues, ok := kubeDNSConfigMap.Data[dataField]; ok {
		var upstreamProxyIP []string

		err := json.Unmarshal([]byte(upstreamValues), &upstreamProxyIP)
		if err != nil {
			return "", errors.Wrap(err, "failed to parse JSON from 'kube-dns ConfigMap")
		}

		coreDNSProxyStanzaList := strings.Join(upstreamProxyIP, " ")
		return coreDNSProxyStanzaList, nil
	}
	return "/etc/resolv.conf", nil
}

// translateFederationsofKubeDNSToCoreDNS translates Federations Data in kube-dns ConfigMap
// to Federation for CoreDNS Corefile.
func translateFederationsofKubeDNSToCoreDNS(dataField, coreDNSDomain string, kubeDNSConfigMap *v1.ConfigMap) (string, error) {
	if kubeDNSConfigMap == nil {
		return "", nil
	}

	if federation, ok := kubeDNSConfigMap.Data[dataField]; ok {
		var (
			federationStanza []interface{}
			body             [][]string
		)
		federationData := make(map[string]string)

		err := json.Unmarshal([]byte(federation), &federationData)
		if err != nil {
			return "", errors.Wrap(err, "failed to parse JSON from kube-dns ConfigMap")
		}
		fStanza := map[string]interface{}{}

		for name, domain := range federationData {
			body = append(body, []string{name, domain})
		}
		federationStanza = append(federationStanza, fStanza)
		fStanza["keys"] = []string{"federation " + coreDNSDomain}
		fStanza["body"] = body
		stanzasBytes, err := json.Marshal(federationStanza)
		if err != nil {
			return "", err
		}

		corefileStanza, err := caddyfile.FromJSON(stanzasBytes)
		if err != nil {
			return "", err
		}

		return prepCorefileFormat(string(corefileStanza), 8), nil
	}
	return "", nil
}

// prepCorefileFormat indents the output of the Corefile caddytext and replaces tabs with spaces
// to neatly format the configmap, making it readable.
func prepCorefileFormat(s string, indentation int) string {
	r := []string{}
	for _, line := range strings.Split(s, "\n") {
		indented := strings.Repeat(" ", indentation) + line
		r = append(r, indented)
	}
	corefile := strings.Join(r, "\n")
	return "\n" + strings.Replace(corefile, "\t", "   ", -1)
}
