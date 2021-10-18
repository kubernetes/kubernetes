/*
Copyright 2021 The Kubernetes Authors.

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

package framework

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type MemorySummary map[string]string

func getMemInfo() string {
	buf, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		Logf("Can't read meminfo %v", err)
		return ""
	}
	return string(buf)
}

//getMemorySummaries construct memory summary
func getMemorySummaries(f *Framework) *MemorySummary {
	var sb strings.Builder

	result := make(MemorySummary)
	result["meminfo"] = getMemInfo()

	if output, err := exec.Command("/bin/sh", "-c", "ps -aux --sort=-rss|head -n20").Output(); err == nil {
		result["topmemconsumer"] = string(output)
	} else {
		Logf("Can't get processes: %v\n", err)
	}

	pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
	ExpectNoError(err)
	for _, p := range pods.Items {
		fmt.Fprintf(&sb, "pod: %s\n", p.Name)
	}

	result["podsummary"] = sb.String()
	return &result
}

// PrintHumanReadable return human readable representation of MemorySummary
func (ms *MemorySummary) PrintHumanReadable() string {
	buf := bytes.Buffer{}
	for _, v := range *ms {
		buf.WriteString(fmt.Sprintf("%v\n", v))
	}
	return buf.String()
}

// SummaryKind returns the summary of memory usage.
func (f *MemorySummary) SummaryKind() string {
	return "MemorySummary"
}

func (f *MemorySummary) PrintJSON() string {
	return PrettyPrintJSON(f)
}
