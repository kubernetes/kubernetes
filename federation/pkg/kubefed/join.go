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
)

const (
	// defaultClusterCIDR is the default CIDR range accepted by the
	// joining API server. See `apis/federation.ClusterSpec` for
	// details.
	// TODO(madhusudancs): Make this value customizable.
	defaultClientCIDR = "0.0.0.0/0"
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
			cmdutil.CheckErr(opts.Complete(cmd, args))
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
func (j *joinFederation) Complete(cmd *cobra.Command, args []string) error {
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

	return nil
}

// Run is the implementation of the `join federation` command.
func (j *joinFederation) Run(f cmdutil.Factory, cmdOut io.Writer, config util.AdminConfig, cmd *cobra.Command) error {
	po := config.PathOptions()
	po.LoadingRules.ExplicitPath = j.commonOptions.Kubeconfig
	clientConfig, err := po.GetStartingConfig()
	if err != nil {
		return err
	}

	joiningClusterFactory := config.ClusterFactory(j.options.clusterContext, j.commonOptions.Kubeconfig)
	joiningClusterClient, err := joiningClusterFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not create client for joining cluster: %v", err)
		return err
	}

	// TODO: This needs to check if the API server for the joining cluster supports RBAC.
	// If the cluster does not, this should fall back to the old secret creation strategy.
	glog.V(2).Info("Creating federation system namespace")
	err = createFederationSystemNamespace(joiningClusterClient, j.commonOptions.FederationSystemNamespace)
	if err != nil {
		glog.V(2).Info("Error creating federation system namespace: %v", err)
		return err
	}

	glog.V(2).Info("Creating service account")
	saName, err := createServiceAccount(joiningClusterClient, j.commonOptions.FederationSystemNamespace, j.commonOptions.Host)
	if err != nil {
		glog.V(2).Info("Error creating service account: %v", err)
		return err
	}

	glog.V(2).Info("Creating role binding for service account")
	err = createClusterRoleBinding(joiningClusterClient, saName, j.options.clusterContext, j.commonOptions.FederationSystemNamespace)
	if err != nil {
		glog.V(2).Info("Error creating role binding for service account: %v", err)
		return err
	}

	hostFactory := config.ClusterFactory(j.commonOptions.Host, j.commonOptions.Kubeconfig)
	hostClient, err := hostFactory.ClientSet()
	if err != nil {
		glog.V(2).Infof("Could not get host clientset: %v", err)
		return err
	}

	glog.V(2).Info("Creating secret")
	err = populateSecretInHostCluster(joiningClusterClient, hostClient, saName, j.commonOptions.FederationSystemNamespace, j.options.secretName)
	if err != nil {
		glog.V(2).Info("Error creating secret in host cluster: %v", err)
		return err
	}

	glog.V(2).Info("Generating client")
	generator, err := clusterGenerator(clientConfig, j.commonOptions.Name, j.options.clusterContext, "")
	if err != nil {
		glog.V(2).Infof("Failed creating cluster generator: %v", err)
		return err
	}
	glog.V(2).Infof("Created cluster generator: %#v", generator)

	glog.V(2).Info("Running kubectl")
	return kubectlcmd.RunCreateSubcommand(f, cmd, cmdOut, &kubectlcmd.CreateSubcommandOptions{
		Name:                j.commonOptions.Name,
		StructuredGenerator: generator,
		DryRun:              j.options.dryRun,
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
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

// createFederationSystemNamespace creates the federation-system namespace in the cluster
// associated with clusterClient, if it doesn't exist.
func createFederationSystemNamespace(clusterClient *internalclientset.Clientset, federationNamespace string) error {
	federationNS := api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: federationNamespace,
		},
	}

	_, err := clusterClient.Core().Namespaces().Create(&federationNS)
	if err != nil && !errors.IsAlreadyExists(err) {
		glog.V(2).Infof("Could not create federation-system namespace in client: %v", err)
		return err
	}
	return nil
}

// createServiceAccount creates a service account in the cluster associated with clusterClienr with
// credentials that will be used by the host cluster to access its API server.
func createServiceAccount(clusterClient *internalclientset.Clientset, namespace, hostContext string) (string, error) {
	saName := util.ClusterServiceAccountName(hostContext)
	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: namespace,
		},
	}

	// Check for an existing service account with that name. Delete it if it exists.
	err := clusterClient.Core().ServiceAccounts(namespace).Delete(saName, &metav1.DeleteOptions{})
	if err != nil && !errors.IsNotFound(err) {
		glog.V(2).Infof("Could not delete existing service account: %v", err)
		return "", err
	}

	// Wait for the existing account to be deleted.
	err = wait.PollInfinite(1*time.Second, func() (bool, error) {
		_, err = clusterClient.Core().ServiceAccounts(namespace).Get(saName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		} else if err != nil {
			return false, err
		}

		return false, fmt.Errorf("service account still exists in cluster")
	})

	if err != nil {
		return "", err
	}

	// Create a new service account
	_, err = clusterClient.Core().ServiceAccounts(namespace).Create(sa)
	if err != nil {
		glog.V(2).Infof("Could not create service account in joining cluster: %v", err)
		return "", err
	}

	return saName, nil
}

// createClusterRoleBinding creates an RBAC role and binding that allows the
// service account identified by saName to access all resources in all namespaces
// in the cluster associated with clusterClient.
func createClusterRoleBinding(clusterClient *internalclientset.Clientset, saName, clusterContext, namespace string) error {
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
func populateSecretInHostCluster(clusterClient, hostClient *internalclientset.Clientset, saName, namespace, secretName string) error {
	// Get the secret from the joining cluster.
	var sa *api.ServiceAccount
	err := wait.PollInfinite(1*time.Second, func() (bool, error) {
		var err error
		sa, err = clusterClient.Core().ServiceAccounts(namespace).Get(saName, metav1.GetOptions{})
		if err != nil {
			return false, err
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
			Name: secretName,
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
