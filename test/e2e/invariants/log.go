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

package invariants

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
)

// The code in this file grabs logs of containers in "kube-system" after
// a test run and checks them for certain problems:
// - DATA RACE (depends on enabling race detection in the binary, see https://github.com/kubernetes/kubernetes/pull/133834).
// - unhandled and potentially unexpected errors (https://github.com/kubernetes/kubernetes/issues/122005)
//
// Please speak to SIG-Testing leads before adding anything to this file.

const (
	logInvariantsSIG              = "testing"
	logInvariantsContextText      = "Invariant Logs"
	logInvariantsDataRaceLeafText = "should enable data race checking"
	logInvariantsErrorLeafText    = "should enable error checking"
)

type logCheck struct {
	dataRaces, errors bool
}

func (c logCheck) any() bool {
	return c.dataRaces || c.errors
}

var _ = framework.SIGDescribe(logInvariantsSIG)(logInvariantsContextText, func() {
	// This test is a dummy for selecting the report after suite logic
	// for data races.
	//
	// This allows us to run it by default in most jobs, but it can be opted-out,
	// does not run when selecting Conformance, and it can be tagged Flaky
	// if we encounter issues with it.
	ginkgo.It(logInvariantsDataRaceLeafText, func() {})

	// Same for error log entries. This is a separate dummy test because it's potentially more flaky.
	ginkgo.It(logInvariantsErrorLeafText, func() {})
})

var _ = ginkgo.ReportAfterSuite("Invariant Logs", func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	// Skip early without any logging if we are in dry-run mode and didn't really run any tests.
	if report.SuiteConfig.DryRun {
		return
	}

	check := logCheck{
		dataRaces: invariantsSelected(report, logInvariantsSIG, logInvariantsContextText, logInvariantsDataRaceLeafText),
		errors:    invariantsSelected(report, logInvariantsSIG, logInvariantsContextText, logInvariantsErrorLeafText),
	}
	if !check.any() {
		framework.Logf("No checks enabled.")
		return
	}
	framework.Logf("Checking... %+v", check)

	// Failed assertions are okay here.
	// They get included in the final Ginkgo report because that is generated later.
	config, err := framework.LoadConfig()
	framework.ExpectNoError(err, "loading client config")
	client, err := clientset.NewForConfig(config)
	framework.ExpectNoError(err, "creating client")

	// Collect problems for different kube-system components.
	pods, err := client.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "list system pods")

	// All problems need to be reported as a single failure
	// because Ginkgo only supports one failure per test.
	// The entire text is the final failure message, using some
	// minor Markdown markup (sections, lists of errors)
	// such that it can be copied-and-pasted into a GitHub issue.
	failed, _, message := checkLogs(ctx,
		func(ctx context.Context, namespace, podName, containerName string) (string, error) {
			return getLogs(ctx, client, namespace, podName, containerName)
		}, pods.Items, check)
	if failed {
		framework.Failf("Checking cluster component logs failed:\n\n%s", message)
	}
	framework.Logf("%s", message)
})

type getLogsFunc func(ctx context.Context, namespace, podName, containerName string) (string, error)

func getLogs(ctx context.Context, client clientset.Interface, namespace, podName, containerName string) (string, error) {
	data, err := client.CoreV1().RESTClient().Get().
		Resource("pods").
		Namespace(namespace).
		Name(podName).
		SubResource("log").
		Param("container", containerName).
		Do(ctx).
		Raw()
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func checkLogs(ctx context.Context, getLogs getLogsFunc, pods []v1.Pod, check logCheck) (bool, int, string) {
	var message strings.Builder
	failed := false

	numLogEntries := 0
	for _, pod := range pods {
		for _, container := range pod.Spec.Containers {
			header := "#### " + klog.KObj(&pod).String()
			if len(pod.Spec.Containers) > 1 {
				header += " " + container.Name
			}
			if message.Len() > 0 {
				message.WriteString("\n")
			}
			message.WriteString(header)
			message.WriteString("\n\n")
			msg, containerNumLogEntries, err := checkLogOutput(ctx, getLogs, pod.Namespace, pod.Name, container.Name, check)
			numLogEntries += containerNumLogEntries
			if msg != "" {
				message.WriteString(msg)
				message.WriteString("\n")
			}
			if errList, ok := err.(interface{ Unwrap() []error }); ok {
				// Format each error as a list, except when there is only one or even none.
				wrappedErrors := errList.Unwrap()
				switch len(wrappedErrors) {
				case 0:
					// Shouldn't happen, errors.Join(nil) returns nil.
					err = nil
				case 1:
					err = wrappedErrors[0]
				default:
					for i, err := range wrappedErrors {
						if err == nil {
							continue
						}
						if i > 0 {
							message.WriteString("\n")
						}
						message.WriteString("- ")
						message.WriteString(strings.ReplaceAll(strings.TrimSpace(err.Error()), "\n", "\n  "))
						message.WriteString("\n")
						failed = true
					}
					continue
				}
			}
			if err == nil {
				// Success!
				message.WriteString("Okay.\n")
				continue
			}
			message.WriteString(strings.TrimSpace(err.Error()))
			message.WriteString("\n")
			failed = true
		}
	}
	return failed, numLogEntries, message.String()
}

var dataRaceWarning = regexp.MustCompile(`(?m)^==================
WARNING: DATA RACE
((?:(?:.*)\n)*?)==================
`)

// logEntry matches both klog structured log entries (potentially multiline!) or
// JSON log entries (single line).
var logEntry = regexp.MustCompile(`(?m)^[IWE][[:digit:]]{4}.*?\.go:[[:digit:]]*\].*\n(?:^[[:space:]].*\n)*|^\{.*\}$`)

// checkLogOutput retrieves the log output of a container and returns all problems found in it.
// It must not call ginkgo.Fail, whether it's directly or indirectly!
//
// If non-empty, the returned message includes harmless information. It should be used
// also when there is an error. The returned error might be the result of errors.Join
// or a single error.
func checkLogOutput(ctx context.Context, getLogs getLogsFunc, namespace, podName, containerName string, check logCheck) (msg string, logEntryCount int, err error) {
	log, err := getLogs(ctx, namespace, podName, containerName)
	if err != nil {
		return "", 0, fmt.Errorf("get log output: %w", err)
	}

	var allErrs []error

	// Extract all DATA RACE warnings and capture the remaining log output for further processing.
	var remainigLog strings.Builder
	dataRaces := dataRaceWarning.FindAllStringSubmatchIndex(log, -1)
	last := 0
	for _, m := range dataRaces {
		// Add chunk before the data race.
		remainigLog.WriteString(log[last:m[0]])
		// Skip past the data race.
		last = m[1]
		if check.dataRaces {
			// The actual content of the data race is first (and only) submatch.
			dataRace := log[m[2]:m[3]]
			// Empty line and inndent by four spaces to let it render as verbatim text in Markdown.
			allErrs = append(allErrs, fmt.Errorf("DATA RACE:\n\n    %s", strings.ReplaceAll(strings.TrimSpace(dataRace), "\n", "\n    ")))
		}
	}
	// Add last chunk after the last data race.
	remainigLog.WriteString(log[last:])
	log = remainigLog.String()

	// Ideally we shouldn't have *any* error log entries. Admins should set up
	// alerting and somehow react to errors because they might indicate some real
	// problem. In practice we are far from that. Alerting solutions for Kubernetes
	// probably have a long list of known harmless errors, which is not precise and
	// (no pun intended) error-prone.
	//
	// TODO: Instead count how often certain error log entries occur and treat it as an
	// error if any error occurs more than a certain threshold.
	logEntries := logEntry.FindAllString(log, -1)

	return "", len(logEntries), errors.Join(allErrs...)
}
