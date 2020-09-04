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
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"

	"github.com/caddyserver/caddy/caddyfile"
	"github.com/coredns/corefile-migration/migration"
	"github.com/pkg/errors"
	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	utilsnet "k8s.io/utils/net"
)

const (
	// KubeDNSServiceAccountName describes the name of the ServiceAccount for the kube-dns addon
	KubeDNSServiceAccountName  = "kube-dns"
	kubeDNSStubDomain          = "stubDomains"
	kubeDNSUpstreamNameservers = "upstreamNameservers"
	unableToDecodeCoreDNS      = "unable to decode CoreDNS"
	coreDNSReplicas            = 2
	kubeDNSReplicas            = 1
)

// DeployedDNSAddon returns the type of DNS addon currently deployed
func DeployedDNSAddon(client clientset.Interface) (kubeadmapi.DNSAddOnType, string, error) {
	deploymentsClient := client.AppsV1().Deployments(metav1.NamespaceSystem)
	deployments, err := deploymentsClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "k8s-app=kube-dns"})
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

// deployedDNSReplicas returns the replica count for the current DNS deployment
func deployedDNSReplicas(client clientset.Interface, replicas int32) (*int32, error) {
	deploymentsClient := client.AppsV1().Deployments(metav1.NamespaceSystem)
	deployments, err := deploymentsClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "k8s-app=kube-dns"})
	if err != nil {
		return &replicas, errors.Wrap(err, "couldn't retrieve DNS addon deployments")
	}
	switch len(deployments.Items) {
	case 0:
		return &replicas, nil
	case 1:
		return deployments.Items[0].Spec.Replicas, nil
	default:
		return &replicas, errors.Errorf("multiple DNS addon deployments found: %v", deployments.Items)
	}
}

// EnsureDNSAddon creates the kube-dns or CoreDNS addon
func EnsureDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	if cfg.DNS.Type == kubeadmapi.CoreDNS {
		replicas, err := deployedDNSReplicas(client, coreDNSReplicas)
		if err != nil {
			return err
		}
		return coreDNSAddon(cfg, client, replicas)
	}
	replicas, err := deployedDNSReplicas(client, kubeDNSReplicas)
	if err != nil {
		return err
	}
	return kubeDNSAddon(cfg, client, replicas)
}

func kubeDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, replicas *int32) error {
	if err := CreateServiceAccount(client); err != nil {
		return err
	}

	dnsip, err := kubeadmconstants.GetDNSIP(cfg.Networking.ServiceSubnet, features.Enabled(cfg.FeatureGates, features.IPv6DualStack))
	if err != nil {
		return err
	}

	var dnsBindAddr, dnsProbeAddr string
	if utilsnet.IsIPv6(dnsip) {
		dnsBindAddr = "::1"
		dnsProbeAddr = "[" + dnsBindAddr + "]"
	} else {
		dnsBindAddr = "127.0.0.1"
		dnsProbeAddr = dnsBindAddr
	}

	dnsDeploymentBytes, err := kubeadmutil.ParseTemplate(KubeDNSDeployment,
		struct {
			DeploymentName, KubeDNSImage, DNSMasqImage, SidecarImage, DNSBindAddr, DNSProbeAddr, DNSDomain, ControlPlaneTaintKey string
			Replicas                                                                                                             *int32
		}{
			DeploymentName:       kubeadmconstants.KubeDNSDeploymentName,
			KubeDNSImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSKubeDNSImageName),
			DNSMasqImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSDnsMasqNannyImageName),
			SidecarImage:         images.GetDNSImage(cfg, kubeadmconstants.KubeDNSSidecarImageName),
			DNSBindAddr:          dnsBindAddr,
			DNSProbeAddr:         dnsProbeAddr,
			DNSDomain:            cfg.Networking.DNSDomain,
			ControlPlaneTaintKey: kubeadmconstants.LabelNodeRoleMaster,
			Replicas:             replicas,
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
	fmt.Println("[addons] WARNING: kube-dns is deprecated and will not be supported in a future version")
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

func coreDNSAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, replicas *int32) error {
	// Get the YAML manifest
	coreDNSDeploymentBytes, err := kubeadmutil.ParseTemplate(CoreDNSDeployment, struct {
		DeploymentName, Image, ControlPlaneTaintKey string
		Replicas                                    *int32
	}{
		DeploymentName:       kubeadmconstants.CoreDNSDeploymentName,
		Image:                images.GetDNSImage(cfg, kubeadmconstants.CoreDNSImageName),
		ControlPlaneTaintKey: kubeadmconstants.LabelNodeRoleMaster,
		Replicas:             replicas,
	})
	if err != nil {
		return errors.Wrap(err, "error when parsing CoreDNS deployment template")
	}

	// Get the kube-dns ConfigMap for translation to equivalent CoreDNS Config.
	kubeDNSConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.KubeDNSConfigMap, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	stubDomain, err := translateStubDomainOfKubeDNSToForwardCoreDNS(kubeDNSStubDomain, kubeDNSConfigMap)
	if err != nil {
		return err
	}

	upstreamNameserver, err := translateUpstreamNameServerOfKubeDNSToUpstreamForwardCoreDNS(kubeDNSUpstreamNameservers, kubeDNSConfigMap)
	if err != nil {
		return err
	}
	coreDNSDomain := cfg.Networking.DNSDomain

	// Get the config file for CoreDNS
	coreDNSConfigMapBytes, err := kubeadmutil.ParseTemplate(CoreDNSConfigMap, struct{ DNSDomain, UpstreamNameserver, StubDomain string }{
		DNSDomain:          coreDNSDomain,
		UpstreamNameserver: upstreamNameserver,
		StubDomain:         stubDomain,
	})
	if err != nil {
		return errors.Wrap(err, "error when parsing CoreDNS configMap template")
	}

	dnsip, err := kubeadmconstants.GetDNSIP(cfg.Networking.ServiceSubnet, features.Enabled(cfg.FeatureGates, features.IPv6DualStack))
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
		return errors.Wrapf(err, "%s ConfigMap", unableToDecodeCoreDNS)
	}

	// Create the ConfigMap for CoreDNS or update/migrate it in case it already exists
	_, corefile, currentInstalledCoreDNSVersion, err := GetCoreDNSInfo(client)
	if err != nil {
		return errors.Wrap(err, "unable to fetch CoreDNS current installed version and ConfigMap.")
	}

	corefileMigrationRequired, err := isCoreDNSConfigMapMigrationRequired(corefile, currentInstalledCoreDNSVersion)
	if err != nil {
		return err
	}

	// Assume that migration is always possible, rely on migrateCoreDNSCorefile() to fail if not.
	canMigrateCorefile := true

	if corefileMigrationRequired {
		if err := migrateCoreDNSCorefile(client, coreDNSConfigMap, corefile, currentInstalledCoreDNSVersion); err != nil {
			// Errors in Corefile Migration is verified during preflight checks. This part will be executed when a user has chosen
			// to ignore preflight check errors.
			canMigrateCorefile = false
			klog.Warningf("the CoreDNS Configuration was not migrated: %v. The existing CoreDNS Corefile configuration has been retained.", err)
			if err := apiclient.CreateOrRetainConfigMap(client, coreDNSConfigMap, kubeadmconstants.CoreDNSConfigMap); err != nil {
				return err
			}
		}
	} else {
		if err := apiclient.CreateOrUpdateConfigMap(client, coreDNSConfigMap); err != nil {
			return err
		}
	}

	coreDNSClusterRoles := &rbac.ClusterRole{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSClusterRole), coreDNSClusterRoles); err != nil {
		return errors.Wrapf(err, "%s ClusterRole", unableToDecodeCoreDNS)
	}

	// Create the Clusterroles for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateClusterRole(client, coreDNSClusterRoles); err != nil {
		return err
	}

	coreDNSClusterRolesBinding := &rbac.ClusterRoleBinding{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSClusterRoleBinding), coreDNSClusterRolesBinding); err != nil {
		return errors.Wrapf(err, "%s ClusterRoleBinding", unableToDecodeCoreDNS)
	}

	// Create the Clusterrolebindings for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateClusterRoleBinding(client, coreDNSClusterRolesBinding); err != nil {
		return err
	}

	coreDNSServiceAccount := &v1.ServiceAccount{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), []byte(CoreDNSServiceAccount), coreDNSServiceAccount); err != nil {
		return errors.Wrapf(err, "%s ServiceAccount", unableToDecodeCoreDNS)
	}

	// Create the ConfigMap for CoreDNS or update it in case it already exists
	if err := apiclient.CreateOrUpdateServiceAccount(client, coreDNSServiceAccount); err != nil {
		return err
	}

	coreDNSDeployment := &apps.Deployment{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), deploymentBytes, coreDNSDeployment); err != nil {
		return errors.Wrapf(err, "%s Deployment", unableToDecodeCoreDNS)
	}

	// Create the deployment for CoreDNS or retain it in case the CoreDNS migration has failed during upgrade
	if !canMigrateCorefile {
		if err := apiclient.CreateOrRetainDeployment(client, coreDNSDeployment, kubeadmconstants.CoreDNSDeploymentName); err != nil {
			return err
		}
	} else {
		// Create the Deployment for CoreDNS or update it in case it already exists
		if err := apiclient.CreateOrUpdateDeployment(client, coreDNSDeployment); err != nil {
			return err
		}
	}

	coreDNSService := &v1.Service{}
	return createDNSService(coreDNSService, serviceBytes, client)
}

func createDNSService(dnsService *v1.Service, serviceBytes []byte, client clientset.Interface) error {
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), serviceBytes, dnsService); err != nil {
		return errors.Wrap(err, "unable to decode the DNS service")
	}

	// Can't use a generic apiclient helper func here as we have to tolerate more than AlreadyExists.
	if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Create(context.TODO(), dnsService, metav1.CreateOptions{}); err != nil {
		// Ignore if the Service is invalid with this error message:
		// 	Service "kube-dns" is invalid: spec.clusterIP: Invalid value: "10.96.0.10": provided IP is already allocated

		if !apierrors.IsAlreadyExists(err) && !apierrors.IsInvalid(err) {
			return errors.Wrap(err, "unable to create a new DNS service")
		}

		if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Update(context.TODO(), dnsService, metav1.UpdateOptions{}); err != nil {
			return errors.Wrap(err, "unable to create/update the DNS service")
		}
	}
	return nil
}

// isCoreDNSConfigMapMigrationRequired checks if a migration of the CoreDNS ConfigMap is required.
func isCoreDNSConfigMapMigrationRequired(corefile, currentInstalledCoreDNSVersion string) (bool, error) {
	var isMigrationRequired bool
	if corefile == "" || migration.Default("", corefile) {
		return isMigrationRequired, nil
	}
	deprecated, err := migration.Deprecated(currentInstalledCoreDNSVersion, kubeadmconstants.CoreDNSVersion, corefile)
	if err != nil {
		return isMigrationRequired, errors.Wrap(err, "unable to get list of changes to the configuration.")
	}

	// Check if there are any plugins/options which needs to be removed or is a new default
	for _, dep := range deprecated {
		if dep.Severity == "removed" || dep.Severity == "newDefault" {
			isMigrationRequired = true
		}
	}

	return isMigrationRequired, nil
}

func migrateCoreDNSCorefile(client clientset.Interface, cm *v1.ConfigMap, corefile, currentInstalledCoreDNSVersion string) error {
	// Since the current configuration present is not the default version, try and migrate it.
	updatedCorefile, err := migration.Migrate(currentInstalledCoreDNSVersion, kubeadmconstants.CoreDNSVersion, corefile, false)
	if err != nil {
		return errors.Wrap(err, "unable to migrate CoreDNS ConfigMap")
	}

	// Take a copy of the existing Corefile data as `Corefile-backup` and update the ConfigMap
	if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Update(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"Corefile":        updatedCorefile,
			"Corefile-backup": corefile,
		},
	}, metav1.UpdateOptions{}); err != nil {
		return errors.Wrap(err, "unable to update the CoreDNS ConfigMap")
	}

	// Point the CoreDNS deployment to the `Corefile-backup` data.
	if err := setCorefile(client, "Corefile-backup"); err != nil {
		return err
	}

	fmt.Println("[addons] Migrating CoreDNS Corefile")
	changes, err := migration.Deprecated(currentInstalledCoreDNSVersion, kubeadmconstants.CoreDNSVersion, corefile)
	if err != nil {
		return errors.Wrap(err, "unable to get list of changes to the configuration.")
	}
	// show the migration changes
	klog.V(2).Infof("the CoreDNS configuration has been migrated and applied: %v.", updatedCorefile)
	klog.V(2).Infoln("the old migration has been saved in the CoreDNS ConfigMap under the name [Corefile-backup]")
	klog.V(2).Infoln("The changes in the new CoreDNS Configuration are as follows:")
	for _, change := range changes {
		klog.V(2).Infof("%v", change.ToString())
	}
	return nil
}

// GetCoreDNSInfo gets the current CoreDNS installed and the current Corefile Configuration of CoreDNS.
func GetCoreDNSInfo(client clientset.Interface) (*v1.ConfigMap, string, string, error) {
	coreDNSConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.CoreDNSConfigMap, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, "", "", err
	}
	if apierrors.IsNotFound(err) {
		return nil, "", "", nil
	}
	corefile, ok := coreDNSConfigMap.Data["Corefile"]
	if !ok {
		return nil, "", "", errors.New("unable to find the CoreDNS Corefile data")
	}

	_, currentCoreDNSversion, err := DeployedDNSAddon(client)
	if err != nil {
		return nil, "", "", err
	}

	return coreDNSConfigMap, corefile, currentCoreDNSversion, nil
}

func setCorefile(client clientset.Interface, coreDNSCorefileName string) error {
	dnsDeployment, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.CoreDNSDeploymentName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"volumes":[{"name": "config-volume", "configMap":{"name": "coredns", "items":[{"key": "%s", "path": "Corefile"}]}}]}}}}`, coreDNSCorefileName)

	if _, err := client.AppsV1().Deployments(dnsDeployment.ObjectMeta.Namespace).Patch(context.TODO(), dnsDeployment.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}); err != nil {
		return errors.Wrap(err, "unable to patch the CoreDNS deployment")
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
		for domain, proxyHosts := range stubDomainData {
			proxyIP, err := omitHostnameInTranslation(proxyHosts)
			if err != nil {
				return "", errors.Wrap(err, "invalid format to parse for proxy")
			}
			if len(proxyIP) == 0 {
				continue
			}

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

// translateUpstreamNameServerOfKubeDNSToUpstreamForwardCoreDNS translates UpstreamNameServer Data in kube-dns ConfigMap
// in the form of Proxy for the CoreDNS Corefile.
func translateUpstreamNameServerOfKubeDNSToUpstreamForwardCoreDNS(dataField string, kubeDNSConfigMap *v1.ConfigMap) (string, error) {
	if kubeDNSConfigMap == nil {
		return "/etc/resolv.conf", nil
	}

	if upstreamValues, ok := kubeDNSConfigMap.Data[dataField]; ok {
		var upstreamProxyValues []string

		err := json.Unmarshal([]byte(upstreamValues), &upstreamProxyValues)
		if err != nil {
			return "", errors.Wrap(err, "failed to parse JSON from 'kube-dns ConfigMap")
		}

		upstreamProxyValues, err = omitHostnameInTranslation(upstreamProxyValues)
		if err != nil {
			return "", errors.Wrap(err, "invalid format to parse for proxy")
		}

		coreDNSProxyStanzaList := strings.Join(upstreamProxyValues, " ")
		return coreDNSProxyStanzaList, nil
	}
	return "/etc/resolv.conf", nil
}

// prepCorefileFormat indents the output of the Corefile caddytext and replaces tabs with spaces
// to neatly format the configmap, making it readable.
func prepCorefileFormat(s string, indentation int) string {
	var r []string
	if s == "" {
		return ""
	}
	for _, line := range strings.Split(s, "\n") {
		indented := strings.Repeat(" ", indentation) + line
		r = append(r, indented)
	}
	corefile := strings.Join(r, "\n")
	return "\n" + strings.Replace(corefile, "\t", "   ", -1)
}

// omitHostnameInTranslation checks if the data extracted from the kube-dns ConfigMap contains a valid
// IP address. Hostname to nameservers is not supported on CoreDNS and will
// skip that particular instance, if there is any hostname present.
func omitHostnameInTranslation(forwardIPs []string) ([]string, error) {
	index := 0
	for _, value := range forwardIPs {
		proxyHost, _, err := kubeadmutil.ParseHostPort(value)
		if err != nil {
			return nil, err
		}
		parseIP := net.ParseIP(proxyHost)
		if parseIP == nil {
			klog.Warningf("your kube-dns configuration contains a hostname %v. It will be omitted in the translation to CoreDNS as hostnames are unsupported", proxyHost)
		} else {
			forwardIPs[index] = value
			index++
		}
	}
	forwardIPs = forwardIPs[:index]

	return forwardIPs, nil
}
