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

package proxy

import (
	"bytes"
	"fmt"
	"io"

	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const (
	// KubeProxyServiceAccountName describes the name of the ServiceAccount for the kube-proxy addon
	KubeProxyServiceAccountName = "kube-proxy"

	// KubeProxyConfigMapRoleName sets the name of ClusterRole for ConfigMap
	KubeProxyConfigMapRoleName = "kube-proxy"
)

// EnsureProxyAddon creates the kube-proxy addons
func EnsureProxyAddon(cfg *kubeadmapi.ClusterConfiguration, localEndpoint *kubeadmapi.APIEndpoint, client clientset.Interface, out io.Writer, printManifest bool) error {
	cmByte, err := createKubeProxyConfigMap(cfg, localEndpoint, client, printManifest)
	if err != nil {
		return err
	}

	dsByte, err := createKubeProxyAddon(cfg, client, printManifest)
	if err != nil {
		return err
	}

	if err := printOrCreateKubeProxyObjects(cmByte, dsByte, client, out, printManifest); err != nil {
		return err
	}

	return nil
}

// Create SA, RBACRules or print manifests of them to out if printManifest is true
func printOrCreateKubeProxyObjects(cmByte []byte, dsByte []byte, client clientset.Interface, out io.Writer, printManifest bool) error {
	var saBytes, crbBytes, roleBytes, roleBindingBytes []byte
	var err error

	sa := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeProxyServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	}

	crb := &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: constants.KubeProxyClusterRoleBindingName,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     constants.KubeProxyClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind:      rbac.ServiceAccountKind,
				Name:      KubeProxyServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
	}

	role := &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeProxyConfigMapRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			{
				Verbs:         []string{"get"},
				APIGroups:     []string{""},
				Resources:     []string{"configmaps"},
				ResourceNames: []string{constants.KubeProxyConfigMap},
			},
		},
	}

	rb := &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeProxyConfigMapRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     KubeProxyConfigMapRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: constants.NodeBootstrapTokenAuthGroup,
			},
		},
	}

	// Create the objects if printManifest is false
	if !printManifest {
		if err := apiclient.CreateOrUpdateServiceAccount(client, sa); err != nil {
			return errors.Wrap(err, "error when creating kube-proxy service account")
		}

		if err := apiclient.CreateOrUpdateClusterRoleBinding(client, crb); err != nil {
			return err
		}

		if err := apiclient.CreateOrUpdateRole(client, role); err != nil {
			return err
		}

		if err := apiclient.CreateOrUpdateRoleBinding(client, rb); err != nil {
			return err
		}

		fmt.Fprintln(out, "[addons] Applied essential addon: kube-proxy")

		return nil

	}

	gv := schema.GroupVersion{Group: "", Version: "v1"}
	if saBytes, err = kubeadmutil.MarshalToYaml(sa, gv); err != nil {
		return err
	}

	gv = schema.GroupVersion{Group: "rbac.authorization.k8s.io", Version: "v1"}
	if crbBytes, err = kubeadmutil.MarshalToYaml(crb, gv); err != nil {
		return err
	}

	if roleBytes, err = kubeadmutil.MarshalToYaml(role, gv); err != nil {
		return err
	}

	if roleBindingBytes, err = kubeadmutil.MarshalToYaml(rb, gv); err != nil {
		return err
	}

	fmt.Fprintln(out, "---")
	fmt.Fprintf(out, "%s", saBytes)
	fmt.Fprintln(out, "---")
	fmt.Fprintf(out, "%s", crbBytes)
	fmt.Fprintln(out, "---")
	fmt.Fprintf(out, "%s", roleBytes)
	fmt.Fprintln(out, "---")
	fmt.Fprintf(out, "%s", roleBindingBytes)
	fmt.Fprint(out, "---")
	fmt.Fprintf(out, "%s", cmByte)
	fmt.Fprint(out, "---")
	fmt.Fprintf(out, "%s", dsByte)

	return nil
}

func createKubeProxyConfigMap(cfg *kubeadmapi.ClusterConfiguration, localEndpoint *kubeadmapi.APIEndpoint, client clientset.Interface, printManifest bool) ([]byte, error) {
	// Generate ControlPlane Endpoint kubeconfig file
	controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, localEndpoint)
	if err != nil {
		return []byte(""), err
	}

	kubeProxyCfg, ok := cfg.ComponentConfigs[componentconfigs.KubeProxyGroup]
	if !ok {
		return []byte(""), errors.New("no kube-proxy component config found in the active component config set")
	}

	proxyBytes, err := kubeProxyCfg.Marshal()
	if err != nil {
		return []byte(""), errors.Wrap(err, "error when marshaling")
	}
	var prefixBytes bytes.Buffer
	apiclient.PrintBytesWithLinePrefix(&prefixBytes, proxyBytes, "    ")
	configMapBytes, err := kubeadmutil.ParseTemplate(KubeProxyConfigMap19,
		struct {
			ControlPlaneEndpoint string
			ProxyConfig          string
			ProxyConfigMap       string
			ProxyConfigMapKey    string
		}{
			ControlPlaneEndpoint: controlPlaneEndpoint,
			ProxyConfig:          prefixBytes.String(),
			ProxyConfigMap:       constants.KubeProxyConfigMap,
			ProxyConfigMapKey:    constants.KubeProxyConfigMapKey,
		})
	if err != nil {
		return []byte(""), errors.Wrap(err, "error when parsing kube-proxy configmap template")
	}

	if printManifest {
		return configMapBytes, nil
	}

	kubeproxyConfigMap := &v1.ConfigMap{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), configMapBytes, kubeproxyConfigMap); err != nil {
		return []byte(""), errors.Wrap(err, "unable to decode kube-proxy configmap")
	}

	if !kubeProxyCfg.IsUserSupplied() {
		componentconfigs.SignConfigMap(kubeproxyConfigMap)
	}

	// Create the ConfigMap for kube-proxy or update it in case it already exists
	return []byte(""), apiclient.CreateOrUpdateConfigMap(client, kubeproxyConfigMap)
}

func createKubeProxyAddon(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, printManifest bool) ([]byte, error) {
	daemonSetbytes, err := kubeadmutil.ParseTemplate(KubeProxyDaemonSet19, struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
		Image:             images.GetKubernetesImage(constants.KubeProxy, cfg),
		ProxyConfigMap:    constants.KubeProxyConfigMap,
		ProxyConfigMapKey: constants.KubeProxyConfigMapKey,
	})
	if err != nil {
		return []byte(""), errors.Wrap(err, "error when parsing kube-proxy daemonset template")
	}

	if printManifest {
		return daemonSetbytes, nil
	}

	kubeproxyDaemonSet := &apps.DaemonSet{}
	if err := kuberuntime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), daemonSetbytes, kubeproxyDaemonSet); err != nil {
		return []byte(""), errors.Wrap(err, "unable to decode kube-proxy daemonset")
	}
	// Propagate the http/https proxy host environment variables to the container
	env := &kubeproxyDaemonSet.Spec.Template.Spec.Containers[0].Env
	*env = append(*env, kubeadmutil.MergeKubeadmEnvVars(kubeadmutil.GetProxyEnvVars())...)

	// Create the DaemonSet for kube-proxy or update it in case it already exists
	return []byte(""), apiclient.CreateOrUpdateDaemonSet(client, kubeproxyDaemonSet)
}
