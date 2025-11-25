/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

var (
	describeApplySetLong = templates.LongDesc(i18n.T(`
		Show detailed information about an ApplySet.

		Displays detailed information about a Helm release that has been integrated with
		ApplySet metadata, including parent Secret metadata, managed resources, and
		resource counts by GroupKind.`))

	describeApplySetExample = templates.Examples(i18n.T(`
		# Describe an ApplySet
		kubectl describe applyset my-release

		# Describe an ApplySet in a specific namespace
		kubectl describe applyset my-release -n default`))
)

// DescribeApplySetOptions contains options for the describe applyset command
type DescribeApplySetOptions struct {
	Name      string
	Namespace string

	genericiooptions.IOStreams
}

// NewCmdDescribeApplySet creates the 'describe applyset' command
func NewCmdDescribeApplySet(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := &DescribeApplySetOptions{
		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:                   "applyset NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Show detailed information about an ApplySet"),
		Long:                  describeApplySetLong,
		Example:               describeApplySetExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run(f))
		},
	}

	cmd.Flags().StringVarP(&o.Namespace, "namespace", "n", o.Namespace, "If present, the namespace scope for this CLI request")

	return cmd
}

// Complete completes the options
func (o *DescribeApplySetOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageErrorf(cmd, "NAME is required")
	}
	if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "only one NAME may be specified")
	}

	o.Name = args[0]

	// Get namespace from factory if not explicitly set
	if o.Namespace == "" {
		namespace, _, err := f.ToRawKubeConfigLoader().Namespace()
		if err != nil {
			return err
		}
		o.Namespace = namespace
	}

	return nil
}

// Validate validates the options
func (o *DescribeApplySetOptions) Validate() error {
	if o.Name == "" {
		return fmt.Errorf("NAME is required")
	}
	return nil
}

// Run executes the command
func (o *DescribeApplySetOptions) Run(f cmdutil.Factory) error {
	client, err := f.KubernetesClientSet()
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	// Get the ApplySet parent Secret
	parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, o.Name)
	secret, err := client.CoreV1().Secrets(o.Namespace).Get(context.TODO(), parentName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get ApplySet parent Secret: %w", err)
	}

	// Verify it's an ApplySet parent Secret
	if !IsApplySetParentSecret(secret) {
		return fmt.Errorf("Secret %s/%s is not an ApplySet parent Secret", o.Namespace, parentName)
	}

	// Print detailed information
	return o.printApplySetDetails(secret, client)
}

// printApplySetDetails prints detailed ApplySet information
func (o *DescribeApplySetOptions) printApplySetDetails(secret *v1.Secret, client kubernetes.Interface) error {
	applySetID := secret.Labels[parent.ApplySetParentIDLabel]
	tooling := secret.Annotations[parent.ApplySetToolingAnnotation]
	groupKindsStr := secret.Annotations[parent.ApplySetGKsAnnotation]

	// Print header
	fmt.Fprintf(o.Out, "Name:         %s\n", o.Name)
	fmt.Fprintf(o.Out, "Namespace:    %s\n", secret.Namespace)
	fmt.Fprintf(o.Out, "ApplySet ID:  %s\n", applySetID)
	fmt.Fprintf(o.Out, "Tooling:      %s\n", tooling)
	fmt.Fprintf(o.Out, "Created:      %s\n", secret.CreationTimestamp.Format("2006-01-02 15:04:05 -0700 MST"))
	fmt.Fprintf(o.Out, "\n")

	// Print parent Secret metadata
	fmt.Fprintf(o.Out, "Parent Secret:\n")
	fmt.Fprintf(o.Out, "  Name:      %s\n", secret.Name)
	fmt.Fprintf(o.Out, "  Namespace: %s\n", secret.Namespace)
	fmt.Fprintf(o.Out, "  UID:       %s\n", secret.UID)
	fmt.Fprintf(o.Out, "\n")

	// Print GroupKinds
	if groupKindsStr != "" {
		groupKinds, err := parent.ParseGroupKinds(groupKindsStr)
		if err == nil {
			fmt.Fprintf(o.Out, "GroupKinds:\n")
			gks := make([]string, 0, groupKinds.Len())
			for gk := range groupKinds {
				if gk.Group == "" {
					gks = append(gks, gk.Kind)
				} else {
					gks = append(gks, fmt.Sprintf("%s.%s", gk.Kind, gk.Group))
				}
			}
			sort.Strings(gks)
			for _, gk := range gks {
				fmt.Fprintf(o.Out, "  - %s\n", gk)
			}
			fmt.Fprintf(o.Out, "\n")
		}
	}

	// Print managed resources
	fmt.Fprintf(o.Out, "Managed Resources:\n")
	resources, err := o.listManagedResources(secret, client)
	if err != nil {
		fmt.Fprintf(o.ErrOut, "Warning: failed to list managed resources: %v\n", err)
	} else {
		// Group by GroupKind
		byGK := make(map[schema.GroupKind][]string)
		for _, res := range resources {
			gk := schema.GroupKind{
				Group: res.Group,
				Kind:  res.Kind,
			}
			byGK[gk] = append(byGK[gk], res.Name)
		}

		// Sort GroupKinds
		gks := make([]schema.GroupKind, 0, len(byGK))
		for gk := range byGK {
			gks = append(gks, gk)
		}
		sort.Slice(gks, func(i, j int) bool {
			if gks[i].Group != gks[j].Group {
				return gks[i].Group < gks[j].Group
			}
			return gks[i].Kind < gks[j].Kind
		})

		// Print resources by GroupKind
		for _, gk := range gks {
			names := byGK[gk]
			sort.Strings(names)
			gkStr := gk.Kind
			if gk.Group != "" {
				gkStr = fmt.Sprintf("%s.%s", gk.Kind, gk.Group)
			}
			fmt.Fprintf(o.Out, "  %s (%d):\n", gkStr, len(names))
			for _, name := range names {
				fmt.Fprintf(o.Out, "    - %s\n", name)
			}
		}
		fmt.Fprintf(o.Out, "\n")
	}

	// Print resource counts summary
	fmt.Fprintf(o.Out, "Resource Counts:\n")
	counts, err := o.getResourceCounts(secret, client)
	if err != nil {
		fmt.Fprintf(o.ErrOut, "Warning: failed to get resource counts: %v\n", err)
	} else {
		total := 0
		for gk, count := range counts {
			total += count
			gkStr := gk.Kind
			if gk.Group != "" {
				gkStr = fmt.Sprintf("%s.%s", gk.Kind, gk.Group)
			}
			fmt.Fprintf(o.Out, "  %s: %d\n", gkStr, count)
		}
		fmt.Fprintf(o.Out, "  Total: %d\n", total)
	}

	return nil
}

// ResourceInfo represents a managed resource
type ResourceInfo struct {
	Name      string
	Namespace string
	Group     string
	Kind      string
}

// listManagedResources lists all resources managed by this ApplySet
func (o *DescribeApplySetOptions) listManagedResources(secret *v1.Secret, client kubernetes.Interface) ([]ResourceInfo, error) {
	releaseName := strings.TrimPrefix(secret.Name, parent.ParentSecretNamePrefix)
	if releaseName == "" {
		return nil, fmt.Errorf("invalid parent Secret name")
	}

	labelSelector := fmt.Sprintf("app.kubernetes.io/instance=%s", releaseName)
	resources := []ResourceInfo{}

	// List Deployments
	deployments, err := client.AppsV1().Deployments(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		for _, d := range deployments.Items {
			resources = append(resources, ResourceInfo{
				Name:      d.Name,
				Namespace: d.Namespace,
				Group:     "apps",
				Kind:      "Deployment",
			})
		}
	}

	// List Services
	services, err := client.CoreV1().Services(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		for _, s := range services.Items {
			resources = append(resources, ResourceInfo{
				Name:      s.Name,
				Namespace: s.Namespace,
				Group:     "",
				Kind:      "Service",
			})
		}
	}

	// List ConfigMaps
	configMaps, err := client.CoreV1().ConfigMaps(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		for _, cm := range configMaps.Items {
			resources = append(resources, ResourceInfo{
				Name:      cm.Name,
				Namespace: cm.Namespace,
				Group:     "",
				Kind:      "ConfigMap",
			})
		}
	}

	// List Secrets (excluding the parent Secret)
	secrets, err := client.CoreV1().Secrets(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		for _, s := range secrets.Items {
			if s.Name != secret.Name {
				resources = append(resources, ResourceInfo{
					Name:      s.Name,
					Namespace: s.Namespace,
					Group:     "",
					Kind:      "Secret",
				})
			}
		}
	}

	return resources, nil
}

// getResourceCounts gets resource counts by GroupKind
func (o *DescribeApplySetOptions) getResourceCounts(secret *v1.Secret, client kubernetes.Interface) (map[schema.GroupKind]int, error) {
	resources, err := o.listManagedResources(secret, client)
	if err != nil {
		return nil, err
	}

	counts := make(map[schema.GroupKind]int)
	for _, res := range resources {
		gk := schema.GroupKind{
			Group: res.Group,
			Kind:  res.Kind,
		}
		counts[gk]++
	}

	return counts, nil
}
