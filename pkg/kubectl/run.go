/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

type BasicReplicationController struct{}

func (BasicReplicationController) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
		{"port", false},
		{"hostport", false},
		{"stdin", false},
		{"tty", false},
		{"command", false},
		{"args", false},
		{"env", false},
		{"requests", false},
		{"limits", false},
	}
}

// populateResourceList takes strings of form <resourceName1>=<value1>,<resourceName1>=<value2>
func populateResourceList(spec string) (api.ResourceList, error) {
	// empty input gets a nil response to preserve generator test expected behaviors
	if spec == "" {
		return nil, nil
	}

	result := api.ResourceList{}
	resourceStatements := strings.Split(spec, ",")
	for _, resourceStatement := range resourceStatements {
		parts := strings.Split(resourceStatement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("Invalid argument syntax %v, expected <resource>=<value>", resourceStatement)
		}
		resourceName := api.ResourceName(parts[0])
		resourceQuantity, err := resource.ParseQuantity(parts[1])
		if err != nil {
			return nil, err
		}
		result[resourceName] = *resourceQuantity
	}
	return result, nil
}

// HandleResourceRequirements parses the limits and requests parameters if specified
func HandleResourceRequirements(params map[string]string) (api.ResourceRequirements, error) {
	result := api.ResourceRequirements{}
	limits, err := populateResourceList(params["limits"])
	if err != nil {
		return result, err
	}
	result.Limits = limits
	requests, err := populateResourceList(params["requests"])
	if err != nil {
		return result, err
	}
	result.Requests = requests
	return result, nil
}

func makePodSpec(params map[string]string, name string) (*api.PodSpec, error) {
	stdin, err := GetBool(params, "stdin", false)
	if err != nil {
		return nil, err
	}

	tty, err := GetBool(params, "tty", false)
	if err != nil {
		return nil, err
	}

	resourceRequirements, err := HandleResourceRequirements(params)
	if err != nil {
		return nil, err
	}

	spec := api.PodSpec{
		Containers: []api.Container{
			{
				Name:      name,
				Image:     params["image"],
				Stdin:     stdin,
				TTY:       tty,
				Resources: resourceRequirements,
			},
		},
	}
	return &spec, nil
}

func (BasicReplicationController) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	args := []string{}
	val, found := genericParams["args"]
	if found {
		var isArray bool
		args, isArray = val.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found: %v", val)
		}
		delete(genericParams, "args")
	}

	// TODO: abstract this logic so that multiple generators can handle env in the same way. Same for parse envs.
	var envs []api.EnvVar
	envStrings, found := genericParams["env"]
	if found {
		if envStringArray, isArray := envStrings.([]string); isArray {
			var err error
			envs, err = parseEnvs(envStringArray)
			if err != nil {
				return nil, err
			}
			delete(genericParams, "env")
		} else {
			return nil, fmt.Errorf("expected []string, found: %v", envStrings)
		}
	}

	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return nil, fmt.Errorf("'name' is a required parameter.")
		}
	}
	// TODO: extract this flag to a central location.
	labelString, found := params["labels"]
	var labels map[string]string
	var err error
	if found && len(labelString) > 0 {
		labels, err = ParseLabels(labelString)
		if err != nil {
			return nil, err
		}
	} else {
		labels = map[string]string{
			"run": name,
		}
	}
	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}
	if len(args) > 0 {
		command, err := GetBool(params, "command", false)
		if err != nil {
			return nil, err
		}
		if command {
			podSpec.Containers[0].Command = args
		} else {
			podSpec.Containers[0].Args = args
		}
	}

	if len(envs) > 0 {
		podSpec.Containers[0].Env = envs
	}

	controller := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: count,
			Selector: labels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: *podSpec,
			},
		},
	}
	if err := updatePodPorts(params, &controller.Spec.Template.Spec); err != nil {
		return nil, err
	}
	return &controller, nil
}

func updatePodPorts(params map[string]string, podSpec *api.PodSpec) (err error) {
	port := -1
	hostPort := -1
	if len(params["port"]) > 0 {
		port, err = strconv.Atoi(params["port"])
		if err != nil {
			return err
		}
	}

	if len(params["hostport"]) > 0 {
		hostPort, err = strconv.Atoi(params["hostport"])
		if err != nil {
			return err
		}
		if hostPort > 0 && port < 0 {
			return fmt.Errorf("--hostport requires --port to be specified")
		}
	}

	// Don't include the port if it was not specified.
	if port > 0 {
		podSpec.Containers[0].Ports = []api.ContainerPort{
			{
				ContainerPort: port,
			},
		}
		if hostPort > 0 {
			podSpec.Containers[0].Ports[0].HostPort = hostPort
		}
	}
	return nil
}

type BasicPod struct{}

func (BasicPod) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"image", true},
		{"port", false},
		{"hostport", false},
		{"stdin", false},
		{"tty", false},
		{"restart", false},
		{"command", false},
		{"args", false},
		{"env", false},
		{"requests", false},
		{"limits", false},
	}
}

func (BasicPod) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	args := []string{}
	val, found := genericParams["args"]
	if found {
		var isArray bool
		args, isArray = val.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found: %v", val)
		}
		delete(genericParams, "args")
	}
	// TODO: abstract this logic so that multiple generators can handle env in the same way. Same for parse envs.
	var envs []api.EnvVar
	envStrings, found := genericParams["env"]
	if found {
		if envStringArray, isArray := envStrings.([]string); isArray {
			var err error
			envs, err = parseEnvs(envStringArray)
			if err != nil {
				return nil, err
			}
			delete(genericParams, "env")
		} else {
			return nil, fmt.Errorf("expected []string, found: %v", envStrings)
		}
	}

	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return nil, fmt.Errorf("'name' is a required parameter.")
		}
	}
	// TODO: extract this flag to a central location.
	labelString, found := params["labels"]
	var labels map[string]string
	var err error
	if found && len(labelString) > 0 {
		labels, err = ParseLabels(labelString)
		if err != nil {
			return nil, err
		}
	}
	stdin, err := GetBool(params, "stdin", false)
	if err != nil {
		return nil, err
	}

	tty, err := GetBool(params, "tty", false)
	if err != nil {
		return nil, err
	}

	resourceRequirements, err := HandleResourceRequirements(params)
	if err != nil {
		return nil, err
	}

	restartPolicy := api.RestartPolicy(params["restart"])
	if len(restartPolicy) == 0 {
		restartPolicy = api.RestartPolicyAlways
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            name,
					Image:           params["image"],
					ImagePullPolicy: api.PullIfNotPresent,
					Stdin:           stdin,
					TTY:             tty,
					Resources:       resourceRequirements,
				},
			},
			DNSPolicy:     api.DNSClusterFirst,
			RestartPolicy: restartPolicy,
		},
	}
	if len(args) > 0 {
		command, err := GetBool(params, "command", false)
		if err != nil {
			return nil, err
		}
		if command {
			pod.Spec.Containers[0].Command = args
		} else {
			pod.Spec.Containers[0].Args = args
		}
	}

	if len(envs) > 0 {
		pod.Spec.Containers[0].Env = envs
	}

	if err := updatePodPorts(params, &pod.Spec); err != nil {
		return nil, err
	}
	return &pod, nil
}

func parseEnvs(envArray []string) ([]api.EnvVar, error) {
	envs := []api.EnvVar{}
	for _, env := range envArray {
		parts := strings.Split(env, "=")
		if len(parts) != 2 || !util.IsCIdentifier(parts[0]) || len(parts[1]) == 0 {
			return nil, fmt.Errorf("invalid env: %v", env)
		}
		envVar := api.EnvVar{Name: parts[0], Value: parts[1]}
		envs = append(envs, envVar)
	}
	return envs, nil
}
