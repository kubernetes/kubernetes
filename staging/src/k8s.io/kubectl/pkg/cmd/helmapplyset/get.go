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
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

var (
	getApplySetsLong = templates.LongDesc(i18n.T(`
		List Helm releases with ApplySet metadata.

		Displays a table of ApplySets (Helm releases) with their tooling, resource counts,
		and age. ApplySets are identified by Secrets with the applyset.kubernetes.io/id label.`))

	getApplySetsExample = templates.Examples(i18n.T(`
		# List all ApplySets in the current namespace
		kubectl get applysets

		# List ApplySets in a specific namespace
		kubectl get applysets -n default

		# List ApplySets across all namespaces
		kubectl get applysets --all-namespaces

		# Filter by tooling
		kubectl get applysets -l applyset.kubernetes.io/tooling=helm/v3

		# Output in JSON format
		kubectl get applysets -o json

		# Output in YAML format
		kubectl get applysets -o yaml

		# Wide output with additional details
		kubectl get applysets -o wide`))
)

// ApplySetInfo represents an ApplySet for display
type ApplySetInfo struct {
	Name       string
	Namespace  string
	Tooling    string
	Resources  int
	GroupKinds []string
	Age        string
	ApplySetID string
}

// GetApplySetsOptions contains options for the get applysets command
type GetApplySetsOptions struct {
	Namespace     string
	AllNamespaces bool
	LabelSelector string
	Output        string

	genericiooptions.IOStreams
}

// NewCmdGetApplySets creates the 'get applysets' command
func NewCmdGetApplySets(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := &GetApplySetsOptions{
		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:     "applysets",
		Aliases: []string{"applyset"},
		Short:   i18n.T("List Helm releases with ApplySet metadata"),
		Long:    getApplySetsLong,
		Example: getApplySetsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run(f))
		},
	}

	cmd.Flags().StringVarP(&o.Namespace, "namespace", "n", o.Namespace, "If present, the namespace scope for this CLI request")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().StringVarP(&o.LabelSelector, "selector", "l", o.LabelSelector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().StringVarP(&o.Output, "output", "o", o.Output, "Output format. One of: (json, yaml, name, go-template, go-template-file, template, templatefile, jsonpath, jsonpath-as-json, jsonpath-file, custom-columns, custom-columns-file, wide)")

	return cmd
}

// Complete completes the options
func (o *GetApplySetsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	// Get namespace from factory if not explicitly set
	if !o.AllNamespaces {
		namespace, _, err := f.ToRawKubeConfigLoader().Namespace()
		if err != nil {
			return err
		}
		if o.Namespace == "" {
			o.Namespace = namespace
		}
	} else {
		o.Namespace = metav1.NamespaceAll
	}

	return nil
}

// Validate validates the options
func (o *GetApplySetsOptions) Validate() error {
	return nil
}

// Run executes the command
func (o *GetApplySetsOptions) Run(f cmdutil.Factory) error {
	client, err := f.KubernetesClientSet()
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	// Build label selector
	labelSelector := parent.ApplySetParentIDLabel
	if o.LabelSelector != "" {
		labelSelector = fmt.Sprintf("%s,%s", labelSelector, o.LabelSelector)
	}

	// List Secrets with ApplySet label
	secrets, err := client.CoreV1().Secrets(o.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return fmt.Errorf("failed to list ApplySet parent Secrets: %w", err)
	}

	// Convert to ApplySetInfo
	applySets := make([]ApplySetInfo, 0, len(secrets.Items))
	for _, secret := range secrets.Items {
		info, err := o.secretToApplySetInfo(&secret, client)
		if err != nil {
			// Skip invalid ApplySets but log the error
			fmt.Fprintf(o.ErrOut, "Warning: skipping invalid ApplySet %s/%s: %v\n", secret.Namespace, secret.Name, err)
			continue
		}
		applySets = append(applySets, *info)
	}

	// Sort by namespace and name
	sort.Slice(applySets, func(i, j int) bool {
		if applySets[i].Namespace != applySets[j].Namespace {
			return applySets[i].Namespace < applySets[j].Namespace
		}
		return applySets[i].Name < applySets[j].Name
	})

	// Print output
	return o.printApplySets(applySets)
}

// secretToApplySetInfo converts a Secret to ApplySetInfo
func (o *GetApplySetsOptions) secretToApplySetInfo(secret *v1.Secret, client kubernetes.Interface) (*ApplySetInfo, error) {
	// Extract ApplySet ID
	applySetID, ok := secret.Labels[parent.ApplySetParentIDLabel]
	if !ok {
		return nil, fmt.Errorf("missing ApplySet ID label")
	}

	// Extract tooling
	tooling := secret.Annotations[parent.ApplySetToolingAnnotation]
	if tooling == "" {
		tooling = "unknown"
	}

	// Extract GroupKinds
	groupKindsStr := secret.Annotations[parent.ApplySetGKsAnnotation]
	groupKinds := []string{}
	if groupKindsStr != "" {
		gks, err := parent.ParseGroupKinds(groupKindsStr)
		if err == nil {
			for gk := range gks {
				if gk.Group == "" {
					groupKinds = append(groupKinds, gk.Kind)
				} else {
					groupKinds = append(groupKinds, fmt.Sprintf("%s.%s", gk.Kind, gk.Group))
				}
			}
			sort.Strings(groupKinds)
		}
	}

	// Count resources (simplified - in production, query by label)
	resourceCount := o.countResources(secret, client)

	// Calculate age
	age := getAge(secret.CreationTimestamp.Time)

	// Extract release name from Secret name (format: applyset-<release-name>)
	releaseName := strings.TrimPrefix(secret.Name, parent.ParentSecretNamePrefix)

	return &ApplySetInfo{
		Name:       releaseName,
		Namespace:  secret.Namespace,
		Tooling:    tooling,
		Resources:  resourceCount,
		GroupKinds: groupKinds,
		Age:        age,
		ApplySetID: applySetID,
	}, nil
}

// countResources counts resources managed by this ApplySet
func (o *GetApplySetsOptions) countResources(secret *v1.Secret, client kubernetes.Interface) int {
	// Extract release name
	releaseName := strings.TrimPrefix(secret.Name, parent.ParentSecretNamePrefix)
	if releaseName == "" {
		return 0
	}

	// Query resources by Helm label
	labelSelector := fmt.Sprintf("app.kubernetes.io/instance=%s", releaseName)

	// Count resources (simplified - just count a few common types)
	// In production, this would query all GroupKinds
	count := 0

	// Count Deployments
	deployments, err := client.AppsV1().Deployments(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		count += len(deployments.Items)
	}

	// Count Services
	services, err := client.CoreV1().Services(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		count += len(services.Items)
	}

	// Count ConfigMaps
	configMaps, err := client.CoreV1().ConfigMaps(secret.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		count += len(configMaps.Items)
	}

	return count
}

// printApplySets prints ApplySets in the requested format
func (o *GetApplySetsOptions) printApplySets(applySets []ApplySetInfo) error {
	switch o.Output {
	case "json":
		return o.printJSON(applySets)
	case "yaml":
		return o.printYAML(applySets)
	case "name":
		return o.printNames(applySets)
	case "wide":
		return o.printWide(applySets)
	case "":
		return o.printTable(applySets)
	default:
		return fmt.Errorf("unsupported output format: %s", o.Output)
	}
}

// printTable prints ApplySets in table format
func (o *GetApplySetsOptions) printTable(applySets []ApplySetInfo) error {
	if len(applySets) == 0 {
		fmt.Fprintf(o.Out, "No resources found.\n")
		return nil
	}

	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	// Print headers
	showNamespace := o.AllNamespaces
	if showNamespace {
		fmt.Fprintf(w, "NAMESPACE\tNAME\tTOOLING\tRESOURCES\tAGE\n")
	} else {
		fmt.Fprintf(w, "NAME\tTOOLING\tRESOURCES\tAGE\n")
	}

	// Print rows
	for _, as := range applySets {
		if showNamespace {
			fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\n", as.Namespace, as.Name, as.Tooling, as.Resources, as.Age)
		} else {
			fmt.Fprintf(w, "%s\t%s\t%d\t%s\n", as.Name, as.Tooling, as.Resources, as.Age)
		}
	}

	return nil
}

// printWide prints ApplySets in wide format
func (o *GetApplySetsOptions) printWide(applySets []ApplySetInfo) error {
	if len(applySets) == 0 {
		fmt.Fprintf(o.Out, "No resources found.\n")
		return nil
	}

	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	// Print headers
	showNamespace := o.AllNamespaces
	if showNamespace {
		fmt.Fprintf(w, "NAMESPACE\tNAME\tTOOLING\tRESOURCES\tGROUP-KINDS\tAGE\tAPPLYSET-ID\n")
	} else {
		fmt.Fprintf(w, "NAME\tTOOLING\tRESOURCES\tGROUP-KINDS\tAGE\tAPPLYSET-ID\n")
	}

	// Print rows
	for _, as := range applySets {
		gksStr := strings.Join(as.GroupKinds, ",")
		if len(gksStr) > 50 {
			gksStr = gksStr[:47] + "..."
		}
		if showNamespace {
			fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\t%s\t%s\n", as.Namespace, as.Name, as.Tooling, as.Resources, gksStr, as.Age, as.ApplySetID)
		} else {
			fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\t%s\n", as.Name, as.Tooling, as.Resources, gksStr, as.Age, as.ApplySetID)
		}
	}

	return nil
}

// printJSON prints ApplySets in JSON format
func (o *GetApplySetsOptions) printJSON(applySets []ApplySetInfo) error {
	encoder := json.NewEncoder(o.Out)
	encoder.SetIndent("", "  ")
	if len(applySets) == 1 {
		return encoder.Encode(applySets[0])
	}
	return encoder.Encode(applySets)
}

// printYAML prints ApplySets in YAML format
func (o *GetApplySetsOptions) printYAML(applySets []ApplySetInfo) error {
	printer := &printers.YAMLPrinter{}
	// Create a list-like structure for YAML output
	list := &metav1.List{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "List",
		},
		Items: make([]runtime.RawExtension, len(applySets)),
	}

	for i, as := range applySets {
		// Convert to JSON bytes
		jsonBytes, err := json.Marshal(as)
		if err != nil {
			return err
		}
		list.Items[i] = runtime.RawExtension{Raw: jsonBytes}
	}

	return printer.PrintObj(list, o.Out)
}

// printNames prints only the names
func (o *GetApplySetsOptions) printNames(applySets []ApplySetInfo) error {
	for _, as := range applySets {
		if o.AllNamespaces {
			fmt.Fprintf(o.Out, "%s/%s\n", as.Namespace, as.Name)
		} else {
			fmt.Fprintf(o.Out, "%s\n", as.Name)
		}
	}
	return nil
}

// getAge returns a human-readable age string
func getAge(t time.Time) string {
	if t.IsZero() {
		return "<unknown>"
	}
	duration := time.Since(t)
	if duration < time.Minute {
		return fmt.Sprintf("%ds", int(duration.Seconds()))
	}
	if duration < time.Hour {
		return fmt.Sprintf("%dm", int(duration.Minutes()))
	}
	if duration < 24*time.Hour {
		return fmt.Sprintf("%dh", int(duration.Hours()))
	}
	days := int(duration.Hours() / 24)
	if days < 30 {
		return fmt.Sprintf("%dd", days)
	}
	months := days / 30
	if months < 12 {
		return fmt.Sprintf("%dmo", months)
	}
	years := months / 12
	return fmt.Sprintf("%dy", years)
}
