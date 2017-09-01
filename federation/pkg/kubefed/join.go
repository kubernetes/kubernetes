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

package kubefed

import (
	"fmt"
	"io"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	kubectlcmd "k8s.io/kubernetes/pkg/kubectl/cmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	// defaultClusterCIDR is the default CIDR range accepted by the
	// joining API server. See `apis/federation.ClusterSpec` for
	// details.
	// TODO(madhusudancs): Make this value customizable.
	defaultClientCIDR           = "0.0.0.0/0"
	CMNameSuffix                = "controller-manager"
	serviceAccountSecretTimeout = 30 * time.Second
)

var (
	join_long = templates.LongDesc(`
		Join adds a cluster to a federation.

        Current context is assumed to be a federation API
        server. Please use the --context flag otherwise.`)
	join_example = templates.Examples(`
		# Join a cluster to a federation by specifying the
		# cluster name and the context name of the federation
		# control plane's host cluster. Cluster name must be
		# a valid RFC 1123 subdomain name. Cluster context
		# must be specified if the cluster name is different
		# than the cluster's context in the local kubeconfig.
		kubefed join foo --host-cluster-context=bar`)
)

type joinFederation struct {
	commonOptions util.SubcommandOptions
	options       joinFederationOptions
}

type joinFederationOptions struct {
	clusterContext string
	secretName     string
	dryRun         bool
}

func (o *joinFederationOptions) Bind(flags *pflag.FlagSet) {
	flags.StringVar(&o.clusterContext, "cluster-context", "", "Name of the cluster's context in the local kubeconfig. Defaults to cluster name if unspecified.")
	flags.StringVar(&o.secretName, "secret-name", "", "Name of the secret where the cluster's credentials will be stored in the host cluster. This name should be a valid RFC 1035 label. Defaults to cluster name if unspecified.")
	flags.MarkDeprecated("secret-name", "kubefed now generates a secret name, and this flag will be removed in a future release.")
}

// NewCmdJoin defines the `join` command that joins a cluster to a
// federation.
func NewCmdJoin(f cmdutil.Factory, cmdOut io.Writer, config util.AdminConfig) *cobra.Command {
	opts := &joinFederation{}

	cmd := &cobra.Command{
		Use:     "join CLUSTER_NAME --host-cluster-context=HOST_CONTEXT",
		Short:   "Join a cluster to a federation",
		Long:    join_long,
		Example: join_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(opts.Complete(cmd, args, config))
			cmdutil.CheckErr(opts.Run(f, cmdOut, config, cmd))
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ClusterV1Beta1GeneratorName)

	flags := cmd.Flags()
	opts.commonOptions.Bind(flags)
	opts.options.Bind(flags)

	return cmd
}

// Complete ensures that options are valid and marshals them if necessary.
func (j *joinFederation) Complete(cmd *cobra.Command, args []string, config util.AdminConfig) error {
	err := j.commonOptions.SetName(cmd, args)
	if err != nil {
		return err
	}

	j.options.dryRun = cmdutil.GetDryRunFlag(cmd)

	if j.options.clusterContext == "" {
		j.options.clusterContext = j.commonOptions.Name
	}

	glog.V(2).Infof("Args and flags: name %s, host: %s, host-system-namespace: %s, kubeconfig: %s, cluster-context: %s, secret-name: %s, dry-run: %s", j.commonOptions.Name, j.commonOptions.Host, j.commonOptions.FederationSystemNamespace, j.commonOptions.Kubeconfig, j.options.clusterContext, j.options.secretName, j.options.dryRun)

	glog.V(2).Infof("Performing preflight checks.")
	err = j.performPreflightChecks(config)
	if err != nil {
		return err
	}
	return nil
}

// performPreflightChecks checks that the host and joining clusters are in
// a consistent state.
// TODO: This currently only verifies a few things. Add more checks.
func (j *joinFederation) performPreflightChecks(config util.AdminConfig) error {
	joiningClusterFactory := j.joningClusterFactory(config)

	// If RBAC is not available, then skip checking for a service account.
	// If RBAC availability cannot be determined, return an error.
	rbacVersionedClientset, err := util.GetVersionedClientForRBACOrFail(joiningClusterFactory)
	if err != nil {
		if _, ok := err.(*util.NoRBACAPIError); ok {
			return nil
		}
		return err
	}

	// Make sure there is no existing service account in the joining cluster.
	saName := util.ClusterServiceAccountName(j.commonOptions.Name, j.commonOptions.Host)
	sa, err := rbacVersionedClientset.Core().ServiceAccounts(j.commonOptions.FederationSystemNamespace).Get(saName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	} else if err != nil {
		return err
	} else if sa != nil {
		return fmt.Errorf("service account already exists in joining cluster")
	}

	return nil
}

// joiningClusterClientset returns a factory for the joining cluster.
func (j *joinFederation) joningClusterFactory(config util.AdminConfig) cmdutil.Factory {
	return config.ClusterFactory(j.options.clusterContext, j.commonOptions.Kubeconfig)
}

// Run is the implementation of the `join federation` command.
func (j *joinFederation) Run(f cmdutil.Factory, cmdOut io.Writer, config util.AdminConfig, cmd *cobra.Command) error {
	clusterContext := j.options.clusterContext
	dryRun := j.options.dryRun
	federationNamespace := j.commonOptions.FederationSystemNamespace
	host := j.commonOptions.Host
	kubeconfig := j.commonOptions.Kubeconfig
	joiningClusterName := j.commonOptions.Name
	secretName := j.options.secretName
	if secretName == "" {
		secretName = k8s_api_v1.SimpleNameGenerator.GenerateName(j.commonOptions.Name + "-")
	}

	joiningClusterFactory := j.joningClusterFactory(config)
	joiningClusterClientset, err := joiningClusterFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not create client for joining cluster: %v", err)
		return err
	}

	hostFactory := config.ClusterFactory(host, kubeconfig)
	hostClientset, err := hostFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not create client for host cluster: %v", err)
		return err
	}

	federationName, err := getFederationName(hostClientset, federationNamespace)
	if err != nil {
		glog.V(2).Infof("Failed to get the federation name: %v", err)
		return err
	}

	glog.V(2).Info("Creating federation system namespace in joining cluster")
	_, err = createFederationSystemNamespace(joiningClusterClientset, federationNamespace, federationName, joiningClusterName, dryRun)
	if err != nil {
		glog.V(2).Infof("Error creating federation system namespace in joining cluster: %v", err)
		return err
	}
	glog.V(2).Info("Created federation system namespace in joining cluster")

	po := config.PathOptions()
	po.LoadingRules.ExplicitPath = kubeconfig
	clientConfig, err := po.GetStartingConfig()
	if err != nil {
		glog.V(2).Infof("Could not load clientConfig from %s: %v", kubeconfig, err)
		return err
	}

	serviceAccountName := ""
	clusterRoleName := ""
	// Check for RBAC in the joining cluster. If it supports RBAC, then create
	// a service account and use its credentials; otherwise, use the credentials
	// from the local kubeconfig.
	glog.V(2).Info("Creating cluster credentials secret")
	rbacClientset, err := util.GetVersionedClientForRBACOrFail(joiningClusterFactory)
	if err == nil {
		if _, serviceAccountName, clusterRoleName, err = createRBACSecret(hostClientset, rbacClientset, federationNamespace, federationName, joiningClusterName, host, clusterContext, secretName, dryRun); err != nil {
			glog.V(2).Infof("Could not create cluster credentials secret: %v", err)
			return err
		}
	} else {
		if _, ok := err.(*util.NoRBACAPIError); ok {

			// We are not using the `kubectl create secret` machinery through
			// `RunCreateSubcommand` as we do to the cluster resource below
			// because we have a bunch of requirements that the machinery does
			// not satisfy.
			// 1. We want to create the secret in a specific namespace, which
			//    is neither the "default" namespace nor the one specified
			//    via the `--namespace` flag.
			// 2. `SecretGeneratorV1` requires LiteralSources in a string-ified
			//    form that it parses to generate the secret data key-value
			//    pairs. We, however, have the key-value pairs ready without a
			//    need for parsing.
			// 3. The result printing mechanism needs to be mostly quiet. We
			//    don't have to print the created secret in the default case.
			// Having said that, secret generation machinery could be altered to
			// suit our needs, but it is far less invasive and readable this way.
			_, err = createSecret(hostClientset, clientConfig, federationNamespace, federationName, joiningClusterName, clusterContext, secretName, dryRun)
			if err != nil {
				glog.V(2).Infof("Failed creating the cluster credentials secret: %v", err)
				return err
			}
		} else {
			glog.V(2).Infof("Failed to get or verify absence of RBAC client: %v", err)
			return err
		}
	}
	glog.V(2).Info("Cluster credentials secret created")

	glog.V(2).Info("Creating a generator for the cluster API object")
	generator, err := clusterGenerator(clientConfig, joiningClusterName, clusterContext, secretName, serviceAccountName, clusterRoleName)
	if err != nil {
		glog.V(2).Infof("Failed to create a generator for the cluster API object: %v", err)
		return err
	}
	glog.V(2).Info("Created a generator for the cluster API object")

	glog.V(2).Info("Running create cluster command against the federation API server")
	err = kubectlcmd.RunCreateSubcommand(f, cmd, cmdOut, &kubectlcmd.CreateSubcommandOptions{
		Name:                joiningClusterName,
		StructuredGenerator: generator,
		DryRun:              dryRun,
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
	if err != nil {
		glog.V(2).Infof("Failed running create cluster command against the federation API server: %v", err)
		return err
	}
	glog.V(2).Info("Successfully ran create cluster command against the federation API server")

	// We further need to create a configmap named kube-config in the
	// just registered cluster which will be consumed by the kube-dns
	// of this cluster.
	glog.V(2).Info("Creating configmap in host cluster")
	_, err = createConfigMap(hostClientset, config, federationNamespace, federationName, joiningClusterName, clusterContext, kubeconfig, dryRun)
	if err != nil {
		glog.V(2).Infof("Failed to create configmap in cluster: %v", err)
		return err
	}
	glog.V(2).Info("Created configmap in host cluster")

	return err
}

// minifyConfig is a wrapper around `clientcmdapi.MinifyConfig()` that
// sets the current context to the given context before calling
// `clientcmdapi.MinifyConfig()`.
func minifyConfig(clientConfig *clientcmdapi.Config, context string) (*clientcmdapi.Config, error) {
	// MinifyConfig inline-modifies the passed clientConfig. So we make a
	// copy of it before passing the config to it. A shallow copy is
	// sufficient because the underlying fields will be reconstructed by
	// MinifyConfig anyway.
	newClientConfig := *clientConfig
	newClientConfig.CurrentContext = context
	err := clientcmdapi.MinifyConfig(&newClientConfig)
	if err != nil {
		return nil, err
	}
	return &newClientConfig, nil
}

// createSecret extracts the kubeconfig for a given cluster and populates
// a secret with that kubeconfig.
func createSecret(clientset internalclientset.Interface, clientConfig *clientcmdapi.Config, namespace, federationName, joiningClusterName, contextName, secretName string, dryRun bool) (runtime.Object, error) {
	// Minify the kubeconfig to ensure that there is only information
	// relevant to the cluster we are registering.
	newClientConfig, err := minifyConfig(clientConfig, contextName)
	if err != nil {
		glog.V(2).Infof("Failed to minify the kubeconfig for the given context %q: %v", contextName, err)
		return nil, err
	}

	// Flatten the kubeconfig to ensure that all the referenced file
	// contents are inlined.
	err = clientcmdapi.FlattenConfig(newClientConfig)
	if err != nil {
		glog.V(2).Infof("Failed to flatten the kubeconfig for the given context %q: %v", contextName, err)
		return nil, err
	}

	return util.CreateKubeconfigSecret(clientset, newClientConfig, namespace, secretName, federationName, joiningClusterName, dryRun)
}

// createConfigMap creates a configmap with name kube-dns in the joining cluster
// which stores the information about this federation zone name.
// If the configmap with this name already exists, its updated with this information.
func createConfigMap(hostClientSet internalclientset.Interface, config util.AdminConfig, fedSystemNamespace, federationName, joiningClusterName, targetClusterContext, kubeconfigPath string, dryRun bool) (*api.ConfigMap, error) {
	cmDep, err := getCMDeployment(hostClientSet, fedSystemNamespace)
	if err != nil {
		return nil, err
	}
	domainMap, ok := cmDep.Annotations[util.FedDomainMapKey]
	if !ok {
		return nil, fmt.Errorf("kube-dns config map data missing from controller manager annotations")
	}

	targetFactory := config.ClusterFactory(targetClusterContext, kubeconfigPath)
	targetClientSet, err := targetFactory.ClientSet()
	if err != nil {
		return nil, err
	}

	existingConfigMap, err := targetClientSet.Core().ConfigMaps(metav1.NamespaceSystem).Get(util.KubeDnsConfigmapName, metav1.GetOptions{})
	if isNotFound(err) {
		newConfigMap := &api.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      util.KubeDnsConfigmapName,
				Namespace: metav1.NamespaceSystem,
				Annotations: map[string]string{
					federation.FederationNameAnnotation: federationName,
					federation.ClusterNameAnnotation:    joiningClusterName,
				},
			},
			Data: map[string]string{
				util.FedDomainMapKey: domainMap,
			},
		}
		newConfigMap = populateStubDomainsIfRequired(newConfigMap, cmDep.Annotations)

		if dryRun {
			return newConfigMap, nil
		}
		return targetClientSet.Core().ConfigMaps(metav1.NamespaceSystem).Create(newConfigMap)
	}
	if err != nil {
		return nil, err
	}

	if existingConfigMap.Data == nil {
		existingConfigMap.Data = make(map[string]string)
	}
	if _, ok := existingConfigMap.Data[util.FedDomainMapKey]; ok {
		// Append this federation info
		existingConfigMap.Data[util.FedDomainMapKey] = appendConfigMapString(existingConfigMap.Data[util.FedDomainMapKey], cmDep.Annotations[util.FedDomainMapKey])

	} else {
		// For some reason the configMap exists but this data is empty
		existingConfigMap.Data[util.FedDomainMapKey] = cmDep.Annotations[util.FedDomainMapKey]
	}

	if dryRun {
		return existingConfigMap, nil
	}
	return targetClientSet.Core().ConfigMaps(metav1.NamespaceSystem).Update(existingConfigMap)
}

// clusterGenerator extracts the cluster information from the supplied
// kubeconfig and builds a StructuredGenerator for the
// `federation/cluster` API resource.
func clusterGenerator(clientConfig *clientcmdapi.Config, name, contextName, secretName, serviceAccountName, clusterRoleName string) (kubectl.StructuredGenerator, error) {
	// Get the context from the config.
	ctx, found := clientConfig.Contexts[contextName]
	if !found {
		return nil, fmt.Errorf("cluster context %q not found", contextName)
	}

	// Get the cluster object corresponding to the supplied context.
	cluster, found := clientConfig.Clusters[ctx.Cluster]
	if !found {
		return nil, fmt.Errorf("cluster endpoint not found for %q", name)
	}

	// Extract the scheme portion of the cluster APIServer endpoint and
	// default it to `https` if it isn't specified.
	scheme := extractScheme(cluster.Server)
	serverAddress := cluster.Server
	if scheme == "" {
		// Use "https" as the default scheme.
		scheme := "https"
		serverAddress = strings.Join([]string{scheme, serverAddress}, "://")
	}

	generator := &kubectl.ClusterGeneratorV1Beta1{
		Name:               name,
		ClientCIDR:         defaultClientCIDR,
		ServerAddress:      serverAddress,
		SecretName:         secretName,
		ServiceAccountName: serviceAccountName,
		ClusterRoleName:    clusterRoleName,
	}
	return generator, nil
}

// extractScheme parses the given URL to extract the scheme portion
// out of it.
func extractScheme(url string) string {
	scheme := ""
	segs := strings.SplitN(url, "://", 2)
	if len(segs) == 2 {
		scheme = segs[0]
	}
	return scheme
}

func getCMDeployment(hostClientSet internalclientset.Interface, fedNamespace string) (*extensions.Deployment, error) {
	depList, err := hostClientSet.Extensions().Deployments(fedNamespace).List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	for _, dep := range depList.Items {
		if strings.HasSuffix(dep.Name, CMNameSuffix) {
			return &dep, nil
		}
	}
	return nil, fmt.Errorf("could not find the deployment for controller manager in host cluster")
}

func appendConfigMapString(existing string, toAppend string) string {
	if existing == "" {
		return toAppend
	}

	values := strings.Split(existing, ",")
	for _, v := range values {
		// Somehow this federation string is already present,
		// Nothing should be done
		if v == toAppend {
			return existing
		}
	}
	return fmt.Sprintf("%s,%s", existing, toAppend)
}

// getFederationName gets the federation name from the appropriate annotation on the
// control manager deployment.
func getFederationName(hostClientSet internalclientset.Interface, fedNamespace string) (string, error) {
	d, err := getCMDeployment(hostClientSet, fedNamespace)
	if err != nil {
		return "", err
	}

	name, ok := d.Annotations[federation.FederationNameAnnotation]
	if !ok {
		return "", fmt.Errorf("Federation control manager does not have federation name annotation. Please recreate the federation with a newer version of kubefed, or use an older version of kubefed to join this cluster.")
	}

	return name, nil
}

func populateStubDomainsIfRequired(configMap *api.ConfigMap, annotations map[string]string) *api.ConfigMap {
	dnsProvider := annotations[util.FedDNSProvider]
	dnsZoneName := annotations[util.FedDNSZoneName]
	nameServer := annotations[util.FedNameServer]

	if dnsProvider != util.FedDNSProviderCoreDNS || dnsZoneName == "" || nameServer == "" {
		return configMap
	}
	configMap.Data[util.KubeDnsStubDomains] = fmt.Sprintf(`{"%s":["%s"]}`, dnsZoneName, nameServer)
	return configMap
}

// createFederationSystemNamespace creates the federation-system namespace in the cluster
// associated with clusterClientset, if it doesn't already exist.
func createFederationSystemNamespace(clusterClientset internalclientset.Interface, federationNamespace, federationName, joiningClusterName string, dryRun bool) (*api.Namespace, error) {
	federationNS := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: federationNamespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
				federation.ClusterNameAnnotation:    joiningClusterName,
			},
		},
	}

	if dryRun {
		return federationNS, nil
	}

	_, err := clusterClientset.Core().Namespaces().Create(federationNS)
	if err != nil && !errors.IsAlreadyExists(err) {
		glog.V(2).Infof("Could not create federation-system namespace in client: %v", err)
		return nil, err
	}
	return federationNS, nil
}

// createRBACSecret creates a secret in the joining cluster using a service account, and
// populates that secret into the host cluster to allow it to access the joining cluster.
func createRBACSecret(hostClusterClientset, joiningClusterClientset internalclientset.Interface, namespace, federationName, joiningClusterName, hostClusterContext, joiningClusterContext, secretName string, dryRun bool) (*api.Secret, string, string, error) {
	glog.V(2).Info("Creating service account in joining cluster")
	saName, err := createServiceAccount(joiningClusterClientset, namespace, federationName, joiningClusterName, hostClusterContext, dryRun)
	if err != nil {
		glog.V(2).Infof("Error creating service account in joining cluster: %v", err)
		return nil, "", "", err
	}
	glog.V(2).Infof("Created service account in joining cluster")

	glog.V(2).Info("Creating role binding for service account in joining cluster")
	crb, err := createClusterRoleBinding(joiningClusterClientset, saName, namespace, federationName, joiningClusterName, dryRun)
	if err != nil {
		glog.V(2).Infof("Error creating role binding for service account in joining cluster: %v", err)
		return nil, "", "", err
	}
	glog.V(2).Info("Created role binding for service account in joining cluster")

	glog.V(2).Info("Creating secret in host cluster")
	secret, err := populateSecretInHostCluster(joiningClusterClientset, hostClusterClientset, saName, namespace, federationName, joiningClusterName, secretName, dryRun)
	if err != nil {
		glog.V(2).Infof("Error creating secret in host cluster: %v", err)
		return nil, "", "", err
	}
	glog.V(2).Info("Created secret in host cluster")
	return secret, saName, crb.Name, nil
}

// createServiceAccount creates a service account in the cluster associated with clusterClientset with
// credentials that will be used by the host cluster to access its API server.
func createServiceAccount(clusterClientset internalclientset.Interface, namespace, federationName, joiningClusterName, hostContext string, dryRun bool) (string, error) {
	saName := util.ClusterServiceAccountName(joiningClusterName, hostContext)
	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: namespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
				federation.ClusterNameAnnotation:    joiningClusterName,
			},
		},
	}

	if dryRun {
		return saName, nil
	}

	// Create a new service account.
	_, err := clusterClientset.Core().ServiceAccounts(namespace).Create(sa)
	if err != nil {
		return "", err
	}

	return saName, nil
}

// createClusterRoleBinding creates an RBAC role and binding that allows the
// service account identified by saName to access all resources in all namespaces
// in the cluster associated with clusterClientset.
func createClusterRoleBinding(clusterClientset internalclientset.Interface, saName, namespace, federationName, joiningClusterName string, dryRun bool) (*rbac.ClusterRoleBinding, error) {
	roleName := util.ClusterRoleName(federationName, saName)
	role := &rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleName,
			Namespace: namespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
				federation.ClusterNameAnnotation:    joiningClusterName,
			},
		},
		Rules: []rbac.PolicyRule{
			rbac.NewRule(rbac.VerbAll).Groups(rbac.APIGroupAll).Resources(rbac.ResourceAll).RuleOrDie(),
			rbac.NewRule("get").URLs("/healthz").RuleOrDie(),
		},
	}

	// TODO: This should limit its access to only necessary resources.
	rolebinding, err := rbac.NewClusterBinding(roleName).SAs(namespace, saName).Binding()
	rolebinding.ObjectMeta.Namespace = namespace
	rolebinding.ObjectMeta.Annotations = map[string]string{
		federation.FederationNameAnnotation: federationName,
		federation.ClusterNameAnnotation:    joiningClusterName,
	}
	if err != nil {
		glog.V(2).Infof("Could not create role binding for service account: %v", err)
		return nil, err
	}

	if dryRun {
		return &rolebinding, nil
	}

	_, err = clusterClientset.Rbac().ClusterRoles().Create(role)
	if err != nil {
		glog.V(2).Infof("Could not create role for service account in joining cluster: %v", err)
		return nil, err
	}

	_, err = clusterClientset.Rbac().ClusterRoleBindings().Create(&rolebinding)
	if err != nil {
		glog.V(2).Infof("Could not create role binding for service account in joining cluster: %v", err)
		return nil, err
	}

	return &rolebinding, nil
}

// populateSecretInHostCluster copies the service account secret for saName from the cluster
// referenced by clusterClientset to the client referenced by hostClientset, putting it in a secret
// named secretName in the provided namespace.
func populateSecretInHostCluster(clusterClientset, hostClientset internalclientset.Interface, saName, namespace, federationName, joiningClusterName, secretName string, dryRun bool) (*api.Secret, error) {
	if dryRun {
		// The secret is created indirectly with the service account, and so there is no local copy to return in a dry run.
		return nil, nil
	}
	// Get the secret from the joining cluster.
	var sa *api.ServiceAccount
	err := wait.PollImmediate(1*time.Second, serviceAccountSecretTimeout, func() (bool, error) {
		var err error
		sa, err = clusterClientset.Core().ServiceAccounts(namespace).Get(saName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return len(sa.Secrets) == 1, nil
	})
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("Getting secret named: %s", sa.Secrets[0].Name)
	var secret *api.Secret
	err = wait.PollImmediate(1*time.Second, serviceAccountSecretTimeout, func() (bool, error) {
		var err error
		secret, err = clusterClientset.Core().Secrets(namespace).Get(sa.Secrets[0].Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		glog.V(2).Infof("Could not get service account secret from joining cluster: %v", err)
		return nil, err
	}

	// Create a parallel secret in the host cluster.
	v1Secret := api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
				federation.ClusterNameAnnotation:    joiningClusterName,
			},
		},
		Data: secret.Data,
	}

	glog.V(2).Infof("Creating secret in host cluster named: %s", v1Secret.Name)
	_, err = hostClientset.Core().Secrets(namespace).Create(&v1Secret)
	if err != nil {
		glog.V(2).Infof("Could not create secret in host cluster: %v", err)
		return nil, err
	}
	return &v1Secret, nil
}
