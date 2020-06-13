/*
Copyright 2014 The Kubernetes Authors.

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

package versioned

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubectl/pkg/generate"
)

// getLabels returns map of labels.
func getLabels(params map[string]string, name string) (map[string]string, error) {
	labelString, found := params["labels"]
	var labels map[string]string
	var err error
	if found && len(labelString) > 0 {
		labels, err = generate.ParseLabels(labelString)
		if err != nil {
			return nil, err
		}
	} else {
		labels = map[string]string{
			"run": name,
		}
	}
	return labels, nil
}

// getName returns the name of newly created resource.
func getName(params map[string]string) (string, error) {
	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return "", fmt.Errorf("'name' is a required parameter")
		}
	}
	return name, nil
}

// getParams returns map of generic parameters.
func getParams(genericParams map[string]interface{}) (map[string]string, error) {
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	return params, nil
}

// getArgs returns arguments for the container command.
func getArgs(genericParams map[string]interface{}) ([]string, error) {
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
	return args, nil
}

// getEnvs returns environment variables.
func getEnvs(genericParams map[string]interface{}) ([]v1.EnvVar, error) {
	var envs []v1.EnvVar
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
	return envs, nil
}

// populateResourceListV1 takes strings of form <resourceName1>=<value1>,<resourceName1>=<value2>
// and returns ResourceList.
func populateResourceListV1(spec string) (v1.ResourceList, error) {
	// empty input gets a nil response to preserve generator test expected behaviors
	if spec == "" {
		return nil, nil
	}

	result := v1.ResourceList{}
	resourceStatements := strings.Split(spec, ",")
	for _, resourceStatement := range resourceStatements {
		parts := strings.Split(resourceStatement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("Invalid argument syntax %v, expected <resource>=<value>", resourceStatement)
		}
		resourceName := v1.ResourceName(parts[0])
		resourceQuantity, err := resource.ParseQuantity(parts[1])
		if err != nil {
			return nil, err
		}
		result[resourceName] = resourceQuantity
	}
	return result, nil
}

// HandleResourceRequirementsV1 parses the limits and requests parameters if specified
// and returns ResourceRequirements.
func HandleResourceRequirementsV1(params map[string]string) (v1.ResourceRequirements, error) {
	result := v1.ResourceRequirements{}
	limits, err := populateResourceListV1(params["limits"])
	if err != nil {
		return result, err
	}
	result.Limits = limits
	requests, err := populateResourceListV1(params["requests"])
	if err != nil {
		return result, err
	}
	result.Requests = requests
	return result, nil
}

// updatePodContainers updates PodSpec.Containers with passed parameters.
func updatePodContainers(params map[string]string, args []string, envs []v1.EnvVar, imagePullPolicy v1.PullPolicy, podSpec *v1.PodSpec) error {
	if len(args) > 0 {
		command, err := generate.GetBool(params, "command", false)
		if err != nil {
			return err
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

	if len(imagePullPolicy) > 0 {
		// imagePullPolicy should be valid here since we have verified it before.
		podSpec.Containers[0].ImagePullPolicy = imagePullPolicy
	}
	return nil
}

// updatePodContainers updates PodSpec.Containers.Ports with passed parameters.
func updatePodPorts(params map[string]string, podSpec *v1.PodSpec) (err error) {
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
	if len(params["port"]) > 0 {
		podSpec.Containers[0].Ports = []v1.ContainerPort{
			{
				ContainerPort: int32(port),
			},
		}
		if hostPort > 0 {
			podSpec.Containers[0].Ports[0].HostPort = int32(hostPort)
		}
	}
	return nil
}

type BasicPod struct{}

func (BasicPod) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "labels", Required: false},
		{Name: "default-name", Required: false},
		{Name: "name", Required: true},
		{Name: "image", Required: true},
		{Name: "image-pull-policy", Required: false},
		{Name: "port", Required: false},
		{Name: "hostport", Required: false},
		{Name: "stdin", Required: false},
		{Name: "leave-stdin-open", Required: false},
		{Name: "tty", Required: false},
		{Name: "restart", Required: false},
		{Name: "command", Required: false},
		{Name: "args", Required: false},
		{Name: "env", Required: false},
		{Name: "requests", Required: false},
		{Name: "limits", Required: false},
		{Name: "serviceaccount", Required: false},
	}
}

func (BasicPod) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	args, err := getArgs(genericParams)
	if err != nil {
		return nil, err
	}

	envs, err := getEnvs(genericParams)
	if err != nil {
		return nil, err
	}

	params, err := getParams(genericParams)
	if err != nil {
		return nil, err
	}

	name, err := getName(params)
	if err != nil {
		return nil, err
	}

	labels, err := getLabels(params, name)
	if err != nil {
		return nil, err
	}

	stdin, err := generate.GetBool(params, "stdin", false)
	if err != nil {
		return nil, err
	}
	leaveStdinOpen, err := generate.GetBool(params, "leave-stdin-open", false)
	if err != nil {
		return nil, err
	}

	tty, err := generate.GetBool(params, "tty", false)
	if err != nil {
		return nil, err
	}

	resourceRequirements, err := HandleResourceRequirementsV1(params)
	if err != nil {
		return nil, err
	}

	restartPolicy := v1.RestartPolicy(params["restart"])
	if len(restartPolicy) == 0 {
		restartPolicy = v1.RestartPolicyAlways
	}
	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.PodSpec{
			ServiceAccountName: params["serviceaccount"],
			Containers: []v1.Container{
				{
					Name:      name,
					Image:     params["image"],
					Stdin:     stdin,
					StdinOnce: !leaveStdinOpen && stdin,
					TTY:       tty,
					Resources: resourceRequirements,
				},
			},
			DNSPolicy:     v1.DNSClusterFirst,
			RestartPolicy: restartPolicy,
		},
	}
	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, &pod.Spec); err != nil {
		return nil, err
	}

	if err := updatePodPorts(params, &pod.Spec); err != nil {
		return nil, err
	}
	return &pod, nil
}

// parseEnvs converts string into EnvVar objects.
func parseEnvs(envArray []string) ([]v1.EnvVar, error) {
	envs := make([]v1.EnvVar, 0, len(envArray))
	for _, env := range envArray {
		pos := strings.Index(env, "=")
		if pos == -1 {
			return nil, fmt.Errorf("invalid env: %v", env)
		}
		name := env[:pos]
		value := env[pos+1:]
		if len(name) == 0 {
			return nil, fmt.Errorf("invalid env: %v", env)
		}
		if len(validation.IsEnvVarName(name)) != 0 {
			return nil, fmt.Errorf("invalid env: %v", env)
		}
		envVar := v1.EnvVar{Name: name, Value: value}
		envs = append(envs, envVar)
	}
	return envs, nil
}
