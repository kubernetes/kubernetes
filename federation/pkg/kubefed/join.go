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
	"k8s.io/apimachinery/pkg/util/wait"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	kubectlcmd "k8s.io/kubernetes/pkg/kubectl/cmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
)

const (
	// defaultClusterCIDR is the default CIDR range accepted by the
	// joining API server. See `apis/federation.ClusterSpec` for
	// details.
	// TODO(madhusudancs): Make this value customizable.
	defaultClientCIDR = "0.0.0.0/0"
	CMNameSuffix      = "controller-manager"
)

var (
	join_long = templates.LongDesc(`
		Join a cluster to a federation.

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
	if j.options.secretName == "" {
		j.options.secretName = j.commonOptions.Name
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
	joiningClusterFactory := config.ClusterFactory(j.options.clusterContext, j.commonOptions.Kubeconfig)
	joiningClusterClient, err := joiningClusterFactory.ClientSet()

	// Make sure there is no existing service account in the joining cluster.
	saName := util.ClusterServiceAccountName(j.commonOptions.Name, j.commonOptions.Host)
	sa, err := joiningClusterClient.Core().ServiceAccounts(j.commonOptions.FederationSystemNamespace).Get(saName, metav1.GetOptions{})
	if sa != nil {
		return fmt.Errorf("service account already exists in")
	}
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	return nil
}

// Run is the implementation of the `join federation` command.
func (j *joinFederation) Run(f cmdutil.Factory, cmdOut io.Writer, config util.AdminConfig, cmd *cobra.Command) error {
	clusterContext := j.options.clusterContext
	dryRun := j.options.dryRun
	federationNamespace := j.commonOptions.FederationSystemNamespace
	host := j.commonOptions.Host
	kubeconfig := j.commonOptions.Kubeconfig
	federationName := j.commonOptions.Name
	secretName := j.options.secretName

	po := config.PathOptions()
	po.LoadingRules.ExplicitPath = kubeconfig
	clientConfig, err := po.GetStartingConfig()
	if err != nil {
		return err
	}

	joiningClusterFactory := config.ClusterFactory(clusterContext, kubeconfig)
	joiningClusterClient, err := joiningClusterFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not create client for joining cluster: %v", err)
		return err
	}

	// TODO: This needs to check if the API server for the joining cluster supports RBAC.
	// If the cluster does not, this should fall back to the old secret creation strategy.
	glog.V(2).Info("Creating federation system namespace in joining cluster")
	err = createFederationSystemNamespace(joiningClusterClient, federationNamespace, dryRun)
	if err != nil {
		glog.V(2).Info("Error creating federation system namespace in joining cluster: %v", err)
		return err
	}

	glog.V(2).Info("Creating service account in joining cluster")
	saName, err := createServiceAccount(joiningClusterClient, federationNamespace, federationName, host, dryRun)
	if err != nil {
		glog.V(2).Info("Error creating service account in joining cluster: %v", err)
		return err
	}
	glog.V(2).Infof("Cluster credentials secret created")

	glog.V(2).Info("Creating role binding for service account in joining cluster")
	err = createClusterRoleBinding(joiningClusterClient, saName, clusterContext, federationNamespace, dryRun)
	if err != nil {
		glog.V(2).Info("Error creating role binding for service account in joining cluster: %v", err)
		return err
	}

	hostFactory := config.ClusterFactory(host, kubeconfig)
	hostClientset, err := hostFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not get host clientset: %v", err)
		return err
	}

	glog.V(2).Info("Creating secret in host cluster")
	err = populateSecretInHostCluster(joiningClusterClient, hostClientset, saName, federationNamespace, secretName, dryRun)
	if err != nil {
		glog.V(2).Info("Error creating secret in host cluster: %v", err)
		return err
	}

	glog.V(2).Info("Generating client")
	generator, err := clusterGenerator(clientConfig, federationName, clusterContext, secretName)
	if err != nil {
		glog.V(2).Infof("Failed creating cluster generator: %v", err)
		return err
	}
	glog.V(2).Infof("Created cluster generator: %#v", generator)

	glog.V(2).Info("Running kubectl")
	err = kubectlcmd.RunCreateSubcommand(f, cmd, cmdOut, &kubectlcmd.CreateSubcommandOptions{
		Name:                federationName,
		StructuredGenerator: generator,
		DryRun:              dryRun,
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
	if err != nil {
		glog.V(2).Infof("Failed rnning kubectl subcommand %v", err)
		return err
	}

	// We further need to create a configmap named kube-config in the
	// just registered cluster which will be consumed by the kube-dns
	// of this cluster.
	_, err = createConfigMap(hostClientset, config, federationNamespace, clusterContext, kubeconfig, dryRun)
	if err != nil {
		glog.V(2).Infof("Failed creating the config map in cluster: %v", err)
		return err
	}

	return err
}

// createConfigMap creates a configmap with name kube-dns in the joining cluster
// which stores the information about this federation zone name.
// If the configmap with this name already exists, its updated with this information.
func createConfigMap(hostClientSet internalclientset.Interface, config util.AdminConfig, fedSystemNamespace, targetClusterContext, kubeconfigPath string, dryRun bool) (*api.ConfigMap, error) {
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
			},
			Data: map[string]string{
				util.FedDomainMapKey: domainMap,
			},
		}

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
func clusterGenerator(clientConfig *clientcmdapi.Config, name, contextName, secretName string) (kubectl.StructuredGenerator, error) {
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
		Name:          name,
		ClientCIDR:    defaultClientCIDR,
		ServerAddress: serverAddress,
		SecretName:    secretName,
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

// createFederationSystemNamespace creates the federation-system namespace in the cluster
// associated with clusterClient, if it doesn't already exist.
func createFederationSystemNamespace(clusterClient internalclientset.Interface, federationNamespace string, dryRun bool) error {
	federationNS := api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: federationNamespace,
		},
	}

	if dryRun {
		return nil
	}

	_, err := clusterClient.Core().Namespaces().Create(&federationNS)
	if err != nil && !errors.IsAlreadyExists(err) {
		glog.V(2).Infof("Could not create federation-system namespace in client: %v", err)
		return err
	}
	return nil
}

// createServiceAccount creates a service account in the cluster associated with clusterClient with
// credentials that will be used by the host cluster to access its API server.
func createServiceAccount(clusterClient internalclientset.Interface, namespace, federationName, hostContext string, dryRun bool) (string, error) {
	saName := util.ClusterServiceAccountName(federationName, hostContext)
	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: namespace,
		},
	}

	if dryRun {
		return saName, nil
	}

	// Create a new service account.
	_, err := clusterClient.Core().ServiceAccounts(namespace).Create(sa)
	if err != nil {
		glog.V(2).Infof("Could not create service account in joining cluster: %v", err)
		return "", err
	}

	return saName, nil
}

// createClusterRoleBinding creates an RBAC role and binding that allows the
// service account identified by saName to access all resources in all namespaces
// in the cluster associated with clusterClient.
func createClusterRoleBinding(clusterClient internalclientset.Interface, saName, clusterContext, namespace string, dryRun bool) error {
	roleName := util.ClusterRoleName(saName)
	role := &rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleName,
			Namespace: namespace,
		},
		Rules: []rbac.PolicyRule{
			rbac.NewRule(rbac.VerbAll).Groups(rbac.APIGroupAll).Resources(rbac.ResourceAll).RuleOrDie(),
		},
	}

	// TODO: This should limit its access to only necessary resources.
	rolebinding, err := rbac.NewClusterBinding(roleName).SAs(namespace, saName).Binding()
	rolebinding.ObjectMeta.Namespace = namespace
	if err != nil {
		glog.V(2).Infof("Could not create role binding for service account: %v", err)
		return err
	}

	if dryRun {
		return nil
	}

	_, err = clusterClient.Rbac().ClusterRoles().Create(role)
	if err != nil {
		glog.V(2).Infof("Could not create role for service account in joining cluster: %v", err)
		return err
	}

	_, err = clusterClient.Rbac().ClusterRoleBindings().Create(&rolebinding)
	if err != nil {
		glog.V(2).Infof("Could not create role binding for service account in joining cluster: %v", err)
		return err
	}

	return nil
}

// populateSecretInHostCluster copies the service account secret for saName from the cluster
// referenced by clusterClient to the client referenced by hostClient, putting it in a secret
// named secretName in the provided namespace.
func populateSecretInHostCluster(clusterClient, hostClient internalclientset.Interface, saName, namespace, secretName string, dryRun bool) error {
	if dryRun {
		return nil
	}
	// Get the secret from the joining cluster.
	var sa *api.ServiceAccount
	err := wait.PollInfinite(1*time.Second, func() (bool, error) {
		var err error
		sa, err = clusterClient.Core().ServiceAccounts(namespace).Get(saName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		return err
	}

	if len(sa.Secrets) != 1 {
		return fmt.Errorf("Expected 1 secret for service account %s, but got %v", sa.Name, sa.Secrets)
	}

	glog.V(2).Info("Getting secret named: %s", sa.Secrets[0].Name)
	secret, err := clusterClient.Core().Secrets(namespace).Get(sa.Secrets[0].Name, metav1.GetOptions{})
	if err != nil {
		glog.V(2).Infof("Could not get service account secret from joining cluster: %v", err)
		return err
	}

	// Create a parallel secret in the host cluster.
	v1Secret := api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
		Data: secret.Data,
	}

	glog.V(2).Infof("Creating secret in host cluster named: %s", v1Secret.Name)
	_, err = hostClient.Core().Secrets(namespace).Create(&v1Secret)
	if err != nil {
		glog.V(2).Infof("Could not create secret in host cluster: %v", err)
		return err
	}
	return nil
}
