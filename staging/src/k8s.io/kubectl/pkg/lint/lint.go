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

// Package lint checks Kubernetes resource configuration for common errors.
package lint

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

// Cmd command layout
type Cmd struct {
	Factory         cmdutil.Factory
	FileNameOptions *resource.FilenameOptions
	genericclioptions.IOStreams
}

// Linter interface for all Lint Operations
type Linter interface {
	Lint([]*resource.Info) (bool, error)
}

// SecurityContextLinter config for securitycontext linter
type SecurityContextLinter struct {
	Cmd
}

var _ Linter = SecurityContextLinter{}

func (c *Cmd) getLinters() []Linter {
	return []Linter{
		SecurityContextLinter{Cmd: *c},
	}
}

// Run read and process Lint command args
func (c *Cmd) Run() error {
	var err error
	r := c.Factory.NewBuilder().
		Unstructured().
		FilenameParam(false, c.FileNameOptions).
		ContinueOnError().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	items, err := r.Infos()
	if err != nil {
		return err
	}

	passed := true
	for _, l := range c.getLinters() {
		result, err := l.Lint(items)
		if err != nil {
			return err
		}
		passed = passed && result
	}
	if !passed {
		return fmt.Errorf("resources had linting errors")
	}
	return nil
}

//Lint Linter based upon SecurityContext
func (c SecurityContextLinter) Lint(items []*resource.Info) (bool, error) {
	passed := true
	for i := range items {
		u := items[i].Object.(*unstructured.Unstructured)
		unit := u.GetKind()
		if unit == "Pod" {
			pods, _, err := unstructured.NestedFieldCopy(u.Object, "spec") //Pod security context
			if err != nil {
				return false, err
			}
			result := checkPodSecurityContext(pods, unit, u.GetName())
			if result != "" {
				fmt.Fprintf(c.Out, "%v: %v\n", items[i].Source, result)
				passed = false
			}

			containers, _, err := unstructured.NestedSlice(u.Object, "spec", "containers") //containers security context
			if err != nil {
				return false, err
			}
			result = checkContainerSecurityContext(containers, unit, u.GetName())
			if result != "" {
				fmt.Fprintf(c.Out, "%v: %v\n", items[i].Source, result)
				passed = false
			}
		} else {
			pods, _, err := unstructured.NestedFieldCopy(u.Object, "spec", "template", "spec") //Pod security context for Deployment, DaemonSet etc.
			if err != nil {
				return false, err
			}
			result := checkPodSecurityContext(pods, unit, u.GetName())
			if result != "" {
				fmt.Fprintf(c.Out, "%v: %v\n", items[i].Source, result)
				passed = false
			}
			containers, _, err := unstructured.NestedSlice(u.Object, "spec", "template", "spec", "containers") //containers security context ina pod for Deployment, DaemonSet etc.
			if err != nil {
				return false, err
			}
			result = checkContainerSecurityContext(containers, unit, u.GetName())
			if result != "" {
				fmt.Fprintf(c.Out, "%v: %v\n", items[i].Source, result)
				passed = false
			}
		}
	}
	return passed, nil
}

func isRunningAsRoot(runAsUser interface{}, unit, name string) string {
	var linterWarning string
	switch runAsUser := runAsUser.(type) {
	case int64:
		isRoot := runAsUser
		if isRoot == 0 {
			linterWarning = unit + " " + name + " is running as root"
		}
	case int32:
		isRoot := runAsUser
		if isRoot == 0 {
			linterWarning = unit + " " + name + " is running as root"
		}
	case int:
		isRoot := runAsUser
		if isRoot == 0 {
			linterWarning = unit + " " + name + " is running as root"
		}
	}
	return strings.TrimSpace(linterWarning)
}

func checkPodSecurityContext(pods interface{}, unit, name string) string {
	if pods.(map[string]interface{})["securityContext"] != nil {
		securityContext, ok := pods.(map[string]interface{})["securityContext"]
		if ok {
			runAsUser, ok := securityContext.(map[string]interface{})["runAsUser"]
			if ok {
				return isRunningAsRoot(runAsUser, unit, name)
			}
		}
	}
	return ""
}

func checkContainerSecurityContext(containers []interface{}, unit, name string) string {
	for _, elem := range containers {
		securityContext, ok := elem.(map[string]interface{})["securityContext"]
		if ok {
			runAsUser, ok := securityContext.(map[string]interface{})["runAsUser"]
			if ok {
				containerName := elem.(map[string]interface{})["name"]
				item := unit + "/" + name + "/container"
				return isRunningAsRoot(runAsUser, item, containerName.(string))
			}
		}
	}
	return ""
}
