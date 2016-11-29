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
	"net/url"

	federationapi "k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime/schema"

	"github.com/spf13/cobra"
)

var (
	unjoin_long = templates.LongDesc(`
		Unjoin removes a cluster from a federation.

        Current context is assumed to be a federation endpoint.
        Please use the --context flag otherwise.`)
	unjoin_example = templates.Examples(`
		# Unjoin removes the specified cluster from a federation.
		# Federation control plane's host cluster context name
		# must be specified via the --host-cluster-context flag
		# to properly cleanup the credentials.
		kubectl unjoin foo --host-cluster-context=bar`)
)

// NewCmdUnjoin defines the `unjoin` command that removes a cluster
// from a federation.
func NewCmdUnjoin(f cmdutil.Factory, cmdOut, cmdErr io.Writer, config util.AdminConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "unjoin CLUSTER_NAME --host-cluster-context=HOST_CONTEXT",
		Short:   "Unjoins a cluster from a federation",
		Long:    unjoin_long,
		Example: unjoin_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := unjoinFederation(f, cmdOut, cmdErr, config, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	util.AddSubcommandFlags(cmd)
	return cmd
}

// unjoinFederation is the implementation of the `unjoin` command.
func unjoinFederation(f cmdutil.Factory, cmdOut, cmdErr io.Writer, config util.AdminConfig, cmd *cobra.Command, args []string) error {
	unjoinFlags, err := util.GetSubcommandFlags(cmd, args)
	if err != nil {
		return err
	}

	cluster, err := popCluster(f, unjoinFlags.Name)
	if err != nil {
		return err
	}
	if cluster == nil {
		fmt.Fprintf(cmdErr, "WARNING: cluster %q not found in federation, so its credentials' secret couldn't be deleted", unjoinFlags.Name)
		return nil
	}

	// We want a separate client factory to communicate with the
	// federation host cluster. See join_federation.go for details.
	hostFactory := config.HostFactory(unjoinFlags.Host, unjoinFlags.Kubeconfig)
	err = deleteSecret(hostFactory, cluster.Spec.SecretRef.Name, unjoinFlags.FederationSystemNamespace)
	if isNotFound(err) {
		fmt.Fprintf(cmdErr, "WARNING: secret %q not found in the host cluster, so it couldn't be deleted", cluster.Spec.SecretRef.Name)
	} else if err != nil {
		return err
	}
	_, err = fmt.Fprintf(cmdOut, "Successfully removed cluster %q from federation\n", unjoinFlags.Name)
	return err
}

// popCluster fetches the cluster object with the given name, deletes
// it and returns the deleted cluster object.
func popCluster(f cmdutil.Factory, name string) (*federationapi.Cluster, error) {
	// Boilerplate to create the secret in the host cluster.
	mapper, typer := f.Object()
	gvks, _, err := typer.ObjectKinds(&federationapi.Cluster{})
	if err != nil {
		return nil, err
	}
	gvk := gvks[0]
	mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
	if err != nil {
		return nil, err
	}
	client, err := f.ClientForMapping(mapping)
	if err != nil {
		return nil, err
	}

	rh := resource.NewHelper(client, mapping)
	obj, err := rh.Get("", name, false)

	if isNotFound(err) {
		// Cluster isn't registered, there isn't anything to be done here.
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	cluster, ok := obj.(*federationapi.Cluster)
	if !ok {
		return nil, fmt.Errorf("unexpected object type: expected \"federation/v1beta1.Cluster\", got %T: obj: %#v", obj, obj)
	}

	// Remove the cluster resource in the federation API server by
	// calling rh.Delete()
	return cluster, rh.Delete("", name)
}

// deleteSecret deletes the secret with the given name from the host
// cluster.
func deleteSecret(hostFactory cmdutil.Factory, name, namespace string) error {
	clientset, err := hostFactory.ClientSet()
	if err != nil {
		return err
	}
	return clientset.Core().Secrets(namespace).Delete(name, &api.DeleteOptions{})
}

// isNotFound checks if the given error is a NotFound status error.
func isNotFound(err error) bool {
	statusErr := err
	if urlErr, ok := err.(*url.Error); ok {
		statusErr = urlErr.Err
	}
	return errors.IsNotFound(statusErr)
}
