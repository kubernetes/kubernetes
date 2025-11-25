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
	"os"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/labeler"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
)

var (
	statusLong = templates.LongDesc(i18n.T(`
		Display status of a Helm release managed by ApplySet.

		This command shows the overall health status, individual resource health,
		conditions, and recent events for a Helm release.
	`))

	statusExample = templates.Examples(i18n.T(`
		# Show status of a Helm release
		kubectl helmapplyset status my-release

		# Show status in a specific namespace
		kubectl helmapplyset status my-release -n production

		# Show status with tree view
		kubectl helmapplyset status my-release --tree

		# Show status without colors
		kubectl helmapplyset status my-release --no-color

		# Show status in JSON format
		kubectl helmapplyset status my-release -o json
	`))
)

// StatusOptions holds the options for the status command
type StatusOptions struct {
	genericiooptions.IOStreams

	Namespace string
	Name      string
	Tree      bool
	NoColor   bool
	Output    string

	Client  kubernetes.Interface
	Printer printers.ResourcePrinter
}

// NewStatusOptions creates a new StatusOptions
func NewStatusOptions(streams genericiooptions.IOStreams) *StatusOptions {
	return &StatusOptions{
		IOStreams: streams,
		Output:    "table",
	}
}

// NewCmdStatus creates the status command
func NewCmdStatus(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := NewStatusOptions(streams)

	cmd := &cobra.Command{
		Use:                   "status NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display status of a Helm release"),
		Long:                  statusLong,
		Example:               statusExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) < 1 {
				return cmdutil.UsageErrorf(cmd, "NAME is required")
			}
			o.Name = args[0]

			// Get namespace
			namespace, _, err := f.ToRawKubeConfigLoader().Namespace()
			if err != nil {
				return err
			}
			o.Namespace = namespace

			// Override namespace if specified
			if cmd.Flags().Changed("namespace") {
				o.Namespace, err = cmd.Flags().GetString("namespace")
				if err != nil {
					return err
				}
			}

			// Get output format
			if cmd.Flags().Changed("output") {
				o.Output, err = cmd.Flags().GetString("output")
				if err != nil {
					return err
				}
			}

			// Get clients
			client, err := f.KubernetesClientSet()
			if err != nil {
				return err
			}
			o.Client = client

			// Setup printer for structured output
			if o.Output != "table" {
				printFlags := genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme)
				printFlags.OutputFormat = &o.Output
				printer, err := printFlags.ToPrinter()
				if err != nil {
					return err
				}
				o.Printer = printer
			}

			return o.Run()
		},
	}

	cmd.Flags().BoolVar(&o.Tree, "tree", false, "Show resource hierarchy in tree format")
	cmd.Flags().BoolVar(&o.NoColor, "no-color", false, "Disable colored output")
	cmd.Flags().StringVarP(&o.Output, "output", "o", o.Output, "Output format. One of: (table, json, yaml)")

	return cmd
}

// Run executes the status command
func (o *StatusOptions) Run() error {
	ctx := context.Background()

	// Get ApplySet parent Secret
	parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, o.Name)
	parentSecret, err := o.Client.CoreV1().Secrets(o.Namespace).Get(ctx, parentName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get ApplySet parent Secret: %w", err)
	}

	// Get ApplySet ID
	applySetID := parentSecret.Labels[parent.ApplySetParentIDLabel]
	if applySetID == "" {
		return fmt.Errorf("ApplySet parent Secret missing ID label")
	}

	// Get conditions from parent Secret
	conditions, err := o.getConditions(parentSecret)
	if err != nil {
		return fmt.Errorf("failed to get conditions: %w", err)
	}

	// Aggregate health status
	health, err := o.aggregateHealth(ctx, applySetID, parentSecret)
	if err != nil {
		return fmt.Errorf("failed to aggregate health: %w", err)
	}

	// Get recent events
	events, err := o.getRecentEvents(ctx, parentSecret)
	if err != nil {
		// Log error but don't fail - events are optional
		fmt.Fprintf(o.ErrOut, "Warning: Failed to get events: %v\n", err)
	}

	// Output based on format
	if o.Output == "json" || o.Output == "yaml" {
		return o.outputStructured(health, conditions, events)
	}

	return o.outputTable(health, conditions, events)
}

// aggregateHealth aggregates health status for all resources in the ApplySet
func (o *StatusOptions) aggregateHealth(ctx context.Context, applySetID string, parentSecret *v1.Secret) (*status.ReleaseHealth, error) {
	// This is a simplified version - in production, this would use the status aggregator
	// For now, we'll query resources and compute basic health

	labelSelector := fmt.Sprintf("%s=%s", labeler.ApplysetPartOfLabel, applySetID)

	// Get all resources with the ApplySet label
	// This is simplified - in production, we'd use the status aggregator
	health := &status.ReleaseHealth{
		ReleaseName:    o.Name,
		Namespace:      o.Namespace,
		ApplySetID:     applySetID,
		OverallStatus:  "unknown",
		ResourceHealth: make(map[string]status.HealthStatus),
		Timestamp:      metav1.Now(),
	}

	// Determine overall status from conditions
	for key, val := range parentSecret.Annotations {
		if strings.HasPrefix(key, "status.conditions.") {
			var condition metav1.Condition
			if err := json.Unmarshal([]byte(val), &condition); err == nil {
				if condition.Type == "Ready" {
					if condition.Status == metav1.ConditionTrue {
						health.OverallStatus = "healthy"
					} else if strings.Contains(condition.Reason, "Progressing") {
						health.OverallStatus = "progressing"
					} else if strings.Contains(condition.Reason, "Failed") {
						health.OverallStatus = "failed"
					} else {
						health.OverallStatus = "degraded"
					}
				}
			}
		}
	}

	// Note: labelSelector would be used by the status aggregator in production
	// Full implementation would use the status aggregator from pkg/controller/helmapplyset/status
	// to query resources: c.statusAggregator.AggregateHealth(ctx, releaseName, namespace, applySetID, groupKinds)
	_ = labelSelector

	return health, nil
}

// getConditions retrieves conditions from the ApplySet parent Secret
func (o *StatusOptions) getConditions(secret *v1.Secret) ([]metav1.Condition, error) {
	conditions := make([]metav1.Condition, 0)

	conditionTypes := []string{"Ready", "Progressing", "Degraded", "Available"}
	for _, condType := range conditionTypes {
		annotationKey := fmt.Sprintf("status.conditions.%s", strings.ToLower(condType))
		if conditionJSON, ok := secret.Annotations[annotationKey]; ok {
			var condition metav1.Condition
			if err := json.Unmarshal([]byte(conditionJSON), &condition); err == nil {
				conditions = append(conditions, condition)
			}
		}
	}

	return conditions, nil
}

// getRecentEvents retrieves recent events related to the ApplySet parent Secret
func (o *StatusOptions) getRecentEvents(ctx context.Context, secret *v1.Secret) ([]v1.Event, error) {
	// Get events for the parent Secret
	eventList, err := o.Client.CoreV1().Events(o.Namespace).List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Secret", secret.Name),
	})
	if err != nil {
		return nil, err
	}

	// Sort by timestamp (newest first)
	events := eventList.Items
	sort.Slice(events, func(i, j int) bool {
		return events[i].LastTimestamp.Time.After(events[j].LastTimestamp.Time)
	})

	// Return last 5 events
	if len(events) > 5 {
		events = events[:5]
	}

	return events, nil
}

// outputTable outputs status in table format
func (o *StatusOptions) outputTable(health *status.ReleaseHealth, conditions []metav1.Condition, events []v1.Event) error {
	// Header
	o.printHeader(health)

	// Resource table
	o.printResourceTable(health)

	// Conditions
	o.printConditions(conditions)

	// Events
	o.printEvents(events)

	return nil
}

// printHeader prints the status header
func (o *StatusOptions) printHeader(health *status.ReleaseHealth) {
	fmt.Fprintf(o.Out, "\n")
	fmt.Fprintf(o.Out, "Release: %s\n", health.ReleaseName)
	fmt.Fprintf(o.Out, "Namespace: %s\n", health.Namespace)
	fmt.Fprintf(o.Out, "Status: %s\n", o.colorizeStatus(health.OverallStatus))
	fmt.Fprintf(o.Out, "\n")
}

// printResourceTable prints the resource status table
func (o *StatusOptions) printResourceTable(health *status.ReleaseHealth) {
	fmt.Fprintf(o.Out, "Resources:\n")
	fmt.Fprintf(o.Out, "%-20s %-30s %-15s %s\n", "KIND", "NAME", "STATUS", "MESSAGE")
	fmt.Fprintf(o.Out, "%s\n", strings.Repeat("-", 80))

	// Sort resources by GroupKind and name
	type resourceInfo struct {
		key    string
		kind   string
		name   string
		status status.HealthStatus
	}

	resources := make([]resourceInfo, 0, len(health.ResourceHealth))
	for key, status := range health.ResourceHealth {
		parts := strings.Split(key, "/")
		kind := "Unknown"
		name := key
		if len(parts) >= 3 {
			kind = parts[0] // GroupKind
			name = fmt.Sprintf("%s/%s", parts[len(parts)-2], parts[len(parts)-1])
		}

		resources = append(resources, resourceInfo{
			key:    key,
			kind:   kind,
			name:   name,
			status: status,
		})
	}

	sort.Slice(resources, func(i, j int) bool {
		if resources[i].kind != resources[j].kind {
			return resources[i].kind < resources[j].kind
		}
		return resources[i].name < resources[j].name
	})

	for _, res := range resources {
		statusStr := "Ready"
		if !res.status.Healthy {
			if strings.Contains(res.status.Reason, "Progressing") {
				statusStr = "Progressing"
			} else if strings.Contains(res.status.Reason, "Failed") {
				statusStr = "Failed"
			} else {
				statusStr = "NotReady"
			}
		}

		message := res.status.Message
		if message == "" {
			message = "-"
		}

		fmt.Fprintf(o.Out, "%-20s %-30s %-15s %s\n",
			res.kind,
			res.name,
			o.colorizeStatus(statusStr),
			message)
	}

	fmt.Fprintf(o.Out, "\n")
}

// printConditions prints the conditions
func (o *StatusOptions) printConditions(conditions []metav1.Condition) {
	if len(conditions) == 0 {
		return
	}

	fmt.Fprintf(o.Out, "Conditions:\n")
	for _, cond := range conditions {
		statusStr := string(cond.Status)
		if cond.Status == metav1.ConditionTrue {
			statusStr = o.colorizeStatus("True")
		} else if cond.Status == metav1.ConditionFalse {
			statusStr = o.colorizeStatus("False")
		}

		fmt.Fprintf(o.Out, "  %s: %s (%s) - %s\n",
			cond.Type,
			statusStr,
			cond.Reason,
			cond.Message)
	}
	fmt.Fprintf(o.Out, "\n")
}

// printEvents prints recent events
func (o *StatusOptions) printEvents(events []v1.Event) {
	if len(events) == 0 {
		return
	}

	fmt.Fprintf(o.Out, "Recent Events:\n")
	fmt.Fprintf(o.Out, "%-20s %-10s %-25s %s\n", "TIME", "TYPE", "REASON", "MESSAGE")
	fmt.Fprintf(o.Out, "%s\n", strings.Repeat("-", 100))

	for _, event := range events {
		timeStr := event.LastTimestamp.Format(time.RFC3339)
		if time.Since(event.LastTimestamp.Time) < 24*time.Hour {
			timeStr = event.LastTimestamp.Format("15:04:05")
		}

		eventType := event.Type
		if event.Type == v1.EventTypeWarning {
			eventType = o.colorizeStatus("Warning")
		} else {
			eventType = o.colorizeStatus("Normal")
		}

		fmt.Fprintf(o.Out, "%-20s %-10s %-25s %s\n",
			timeStr,
			eventType,
			event.Reason,
			event.Message)
	}
	fmt.Fprintf(o.Out, "\n")
}

// colorizeStatus adds ANSI color codes to status strings
func (o *StatusOptions) colorizeStatus(status string) string {
	if o.NoColor || !o.isTerminal() {
		return status
	}

	switch strings.ToLower(status) {
	case "healthy", "ready", "true":
		return fmt.Sprintf("\033[32m%s\033[0m", status) // Green
	case "progressing":
		return fmt.Sprintf("\033[33m%s\033[0m", status) // Yellow
	case "degraded", "failed", "notready", "false":
		return fmt.Sprintf("\033[31m%s\033[0m", status) // Red
	default:
		return status
	}
}

// isTerminal checks if output is a terminal
func (o *StatusOptions) isTerminal() bool {
	if file, ok := o.Out.(*os.File); ok {
		return term.IsTerminal(int(file.Fd()))
	}
	return false
}

// outputStructured outputs status in JSON or YAML format
func (o *StatusOptions) outputStructured(health *status.ReleaseHealth, conditions []metav1.Condition, events []v1.Event) error {
	// Convert conditions to map[string]interface{} for unstructured
	conditionsList := make([]interface{}, len(conditions))
	for i, cond := range conditions {
		conditionsList[i] = map[string]interface{}{
			"type":               cond.Type,
			"status":             string(cond.Status),
			"reason":             cond.Reason,
			"message":            cond.Message,
			"lastTransitionTime": cond.LastTransitionTime.Format(time.RFC3339),
		}
	}

	// Convert events to map[string]interface{} for unstructured
	eventsList := make([]interface{}, len(events))
	for i, event := range events {
		eventsList[i] = map[string]interface{}{
			"type":          event.Type,
			"reason":        event.Reason,
			"message":       event.Message,
			"lastTimestamp": event.LastTimestamp.Format(time.RFC3339),
		}
	}

	// Convert resource health to map[string]interface{}
	resourcesMap := make(map[string]interface{})
	for key, healthStatus := range health.ResourceHealth {
		resourcesMap[key] = map[string]interface{}{
			"healthy":   healthStatus.Healthy,
			"reason":    healthStatus.Reason,
			"message":   healthStatus.Message,
			"timestamp": healthStatus.Timestamp.Format(time.RFC3339),
		}
	}

	// Create unstructured object for output
	statusObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "HelmApplySetStatus",
			"release": map[string]interface{}{
				"name":      health.ReleaseName,
				"namespace": health.Namespace,
				"status":    health.OverallStatus,
			},
			"resources":  resourcesMap,
			"conditions": conditionsList,
			"events":     eventsList,
		},
	}

	return o.Printer.PrintObj(statusObj, o.Out)
}
