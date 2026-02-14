/*
Copyright 2025 The Kubernetes Authors.

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

package lifecycle

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kubectl/pkg/describe"
	"sigs.k8s.io/yaml"
)

// Printer handles output formatting
type Printer struct {
	streams     genericiooptions.IOStreams
	format      string
	showDetails bool
}

// NewPrinter creates a new output printer
func NewPrinter(streams genericiooptions.IOStreams, format string, showDetails bool) *Printer {
	return &Printer{
		streams:     streams,
		format:      format,
		showDetails: showDetails,
	}
}

// Print outputs the diagnostic result
func (p *Printer) Print(result *DiagnosticResult) error {
	switch p.format {
	case "json":
		return p.printJSON(result)
	case "yaml":
		return p.printYAML(result)
	default:
		return p.printHuman(result)
	}
}

func (p *Printer) printJSON(result *DiagnosticResult) error {
	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(p.streams.Out, string(output))
	return nil
}

func (p *Printer) printYAML(result *DiagnosticResult) error {
	output, err := yaml.Marshal(result)
	if err != nil {
		return err
	}
	fmt.Fprintln(p.streams.Out, string(output))
	return nil
}

func (p *Printer) printHuman(result *DiagnosticResult) error {
	w := describe.NewPrefixWriter(p.streams.Out)

	// Header
	w.Write(describe.LEVEL_0, "Pod Lifecycle Diagnostic Report\n")
	w.Write(describe.LEVEL_0, "================================\n\n")

	// Pod Info
	w.Write(describe.LEVEL_0, "Pod:\t%s/%s\n", result.Pod.Namespace, result.Pod.Name)
	w.Write(describe.LEVEL_0, "Phase:\t%s\n", result.Pod.Status.Phase)
	w.Write(describe.LEVEL_0, "Diagnostic State:\t%s\n", result.Phase)

	if result.Pod.Spec.NodeName != "" {
		w.Write(describe.LEVEL_0, "Node:\t%s\n", result.Pod.Spec.NodeName)
	} else {
		w.Write(describe.LEVEL_0, "Node:\t<not scheduled>\n")
	}
	w.Write(describe.LEVEL_0, "\n")

	// Root Cause (highlighted)
	w.Write(describe.LEVEL_0, "Root Cause Analysis\n")
	w.Write(describe.LEVEL_0, "-------------------\n")
	w.Write(describe.LEVEL_0, "Category:\t%s\n", result.RootCause.Category)
	w.Write(describe.LEVEL_0, "Summary:\t%s\n", result.RootCause.Summary)
	w.Write(describe.LEVEL_0, "Confidence:\t%.0f%%\n", result.RootCause.Confidence*100)
	if p.showDetails {
		w.Write(describe.LEVEL_0, "Details:\t%s\n", result.RootCause.Details)
	}
	w.Write(describe.LEVEL_0, "\n")

	// Scheduling Info
	if result.SchedulingInfo != nil && result.SchedulingInfo.Message != "" {
		w.Write(describe.LEVEL_0, "Scheduling Information\n")
		w.Write(describe.LEVEL_0, "----------------------\n")
		w.Write(describe.LEVEL_0, "Nodes:\t%s/%s available\n",
			result.SchedulingInfo.AvailableNodes, result.SchedulingInfo.TotalNodes)
		if len(result.SchedulingInfo.FailureReasons) > 0 {
			w.Write(describe.LEVEL_0, "Failure Reasons:\n")
			for _, reason := range result.SchedulingInfo.FailureReasons {
				w.Write(describe.LEVEL_1, "- %s\n", reason)
			}
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Container Issues
	if len(result.ContainerIssues) > 0 {
		w.Write(describe.LEVEL_0, "Container Issues\n")
		w.Write(describe.LEVEL_0, "----------------\n")
		for _, issue := range result.ContainerIssues {
			w.Write(describe.LEVEL_0, "Container:\t%s\n", issue.ContainerName)
			w.Write(describe.LEVEL_1, "State:\t%s\n", issue.State)
			if issue.Message != "" {
				w.Write(describe.LEVEL_1, "Message:\t%s\n", issue.Message)
			}
			if issue.RestartCount > 0 {
				w.Write(describe.LEVEL_1, "Restart Count:\t%d\n", issue.RestartCount)
			}
			if issue.ExitCode != 0 {
				w.Write(describe.LEVEL_1, "Exit Code:\t%d\n", issue.ExitCode)
			}
			if p.showDetails && issue.LastLogs != "" {
				w.Write(describe.LEVEL_1, "Previous Logs:\n")
				for _, line := range strings.Split(issue.LastLogs, "\n") {
					if line != "" {
						w.Write(describe.LEVEL_2, "%s\n", line)
					}
				}
			}
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Volume Issues
	if len(result.VolumeIssues) > 0 {
		w.Write(describe.LEVEL_0, "Volume Issues\n")
		w.Write(describe.LEVEL_0, "-------------\n")
		for _, issue := range result.VolumeIssues {
			w.Write(describe.LEVEL_0, "Volume:\t%s\n", issue.VolumeName)
			w.Write(describe.LEVEL_1, "Issue:\t%s\n", issue.Type)
			w.Write(describe.LEVEL_1, "Message:\t%s\n", issue.Message)
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Resource Issues
	if len(result.ResourceIssues) > 0 {
		w.Write(describe.LEVEL_0, "Resource Issues\n")
		w.Write(describe.LEVEL_0, "---------------\n")
		for _, issue := range result.ResourceIssues {
			w.Write(describe.LEVEL_0, "Resource:\t%s\n", issue.ResourceType)
			if issue.Limit != "" {
				w.Write(describe.LEVEL_1, "Limit:\t%s\n", issue.Limit)
			}
			if issue.Requested != "" {
				w.Write(describe.LEVEL_1, "Requested:\t%s\n", issue.Requested)
			}
			w.Write(describe.LEVEL_1, "Message:\t%s\n", issue.Message)
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Probe Issues
	if len(result.ProbeIssues) > 0 {
		w.Write(describe.LEVEL_0, "Probe Issues\n")
		w.Write(describe.LEVEL_0, "------------\n")
		for _, issue := range result.ProbeIssues {
			w.Write(describe.LEVEL_0, "Probe Type:\t%s\n", issue.ProbeType)
			if issue.ContainerName != "" {
				w.Write(describe.LEVEL_1, "Container:\t%s\n", issue.ContainerName)
			}
			if issue.FailureCount > 0 {
				w.Write(describe.LEVEL_1, "Failure Count:\t%d\n", issue.FailureCount)
			}
			if issue.Message != "" {
				w.Write(describe.LEVEL_1, "Message:\t%s\n", issue.Message)
			}
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Network Issues
	if len(result.NetworkIssues) > 0 && p.showDetails {
		w.Write(describe.LEVEL_0, "Network Information\n")
		w.Write(describe.LEVEL_0, "-------------------\n")
		for _, issue := range result.NetworkIssues {
			w.Write(describe.LEVEL_0, "%s:\t%s\n", issue.Type, issue.Message)
		}
		w.Write(describe.LEVEL_0, "\n")
	}

	// Recommendations
	w.Write(describe.LEVEL_0, "Recommendations\n")
	w.Write(describe.LEVEL_0, "---------------\n")
	for i, rec := range result.Recommendations {
		if i >= 3 && !p.showDetails {
			w.Write(describe.LEVEL_0, "(Use --show-details for more recommendations)\n")
			break
		}
		w.Write(describe.LEVEL_0, "%d. %s\n", rec.Priority, rec.Action)
		if rec.Command != "" {
			w.Write(describe.LEVEL_1, "Command: %s\n", rec.Command)
		}
		if p.showDetails && rec.Explanation != "" {
			w.Write(describe.LEVEL_1, "Why: %s\n", rec.Explanation)
		}
	}
	w.Write(describe.LEVEL_0, "\n")

	// Events (if showDetails)
	if p.showDetails && len(result.Events) > 0 {
		w.Write(describe.LEVEL_0, "Recent Events\n")
		w.Write(describe.LEVEL_0, "-------------\n")
		w.Write(describe.LEVEL_0, "Type\tReason\tAge\tMessage\n")
		for _, event := range result.Events {
			age := translateAge(event.Age)
			w.Write(describe.LEVEL_0, "%s\t%s\t%s\t%s\n",
				event.Type, event.Reason, age, truncate(event.Message, 60))
		}
	}

	w.Flush()
	return nil
}

func translateAge(t time.Time) string {
	if t.IsZero() {
		return "<unknown>"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd", int(d.Hours()/24))
	}
}
