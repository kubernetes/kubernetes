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

package kubectl

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

type DeploymentV1Beta1 struct{}

func (DeploymentV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
		{"image-pull-policy", false},
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

func (DeploymentV1Beta1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
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

	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}

	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, podSpec); err != nil {
		return nil, err
	}

	if err := updatePodPorts(params, podSpec); err != nil {
		return nil, err
	}

	// TODO: use versioned types for generators so that we don't need to
	// set default values manually (see issue #17384)
	count32 := int32(count)
	deployment := extensionsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: extensionsv1beta1.DeploymentSpec{
			Replicas: &count32,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: *podSpec,
			},
		},
	}
	return &deployment, nil
}

type DeploymentAppsV1Beta1 struct{}

func (DeploymentAppsV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
		{"image-pull-policy", false},
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

func (DeploymentAppsV1Beta1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
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

	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}

	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, podSpec); err != nil {
		return nil, err
	}

	if err := updatePodPorts(params, podSpec); err != nil {
		return nil, err
	}

	count32 := int32(count)
	deployment := appsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: &count32,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: *podSpec,
			},
		},
	}
	return &deployment, nil
}

// getLabels returns map of labels.
func getLabels(params map[string]string, name string) (map[string]string, error) {
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
	return labels, nil
}

// getName returns the name of newly created resource.
func getName(params map[string]string) (string, error) {
	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return "", fmt.Errorf("'name' is a required parameter.")
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

type JobV1 struct{}

func (JobV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"image", true},
		{"image-pull-policy", false},
		{"port", false},
		{"hostport", false},
		{"stdin", false},
		{"leave-stdin-open", false},
		{"tty", false},
		{"command", false},
		{"args", false},
		{"env", false},
		{"requests", false},
		{"limits", false},
		{"restart", false},
	}
}

func (JobV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
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

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}

	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, podSpec); err != nil {
		return nil, err
	}

	leaveStdinOpen, err := GetBool(params, "leave-stdin-open", false)
	if err != nil {
		return nil, err
	}
	podSpec.Containers[0].StdinOnce = !leaveStdinOpen && podSpec.Containers[0].Stdin

	if err := updatePodPorts(params, podSpec); err != nil {
		return nil, err
	}

	restartPolicy := v1.RestartPolicy(params["restart"])
	if len(restartPolicy) == 0 {
		restartPolicy = v1.RestartPolicyNever
	}
	podSpec.RestartPolicy = restartPolicy

	job := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: batchv1.JobSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: *podSpec,
			},
		},
	}

	return &job, nil
}

type CronJobV2Alpha1 struct{}

func (CronJobV2Alpha1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"image", true},
		{"image-pull-policy", false},
		{"port", false},
		{"hostport", false},
		{"stdin", false},
		{"leave-stdin-open", false},
		{"tty", false},
		{"command", false},
		{"args", false},
		{"env", false},
		{"requests", false},
		{"limits", false},
		{"restart", false},
		{"schedule", true},
	}
}

func (CronJobV2Alpha1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
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

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}

	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, podSpec); err != nil {
		return nil, err
	}

	leaveStdinOpen, err := GetBool(params, "leave-stdin-open", false)
	if err != nil {
		return nil, err
	}
	podSpec.Containers[0].StdinOnce = !leaveStdinOpen && podSpec.Containers[0].Stdin

	if err := updatePodPorts(params, podSpec); err != nil {
		return nil, err
	}

	restartPolicy := v1.RestartPolicy(params["restart"])
	if len(restartPolicy) == 0 {
		restartPolicy = v1.RestartPolicyNever
	}
	podSpec.RestartPolicy = restartPolicy

	cronJob := batchv2alpha1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: batchv2alpha1.CronJobSpec{
			Schedule:          params["schedule"],
			ConcurrencyPolicy: batchv2alpha1.AllowConcurrent,
			JobTemplate: batchv2alpha1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labels,
						},
						Spec: *podSpec,
					},
				},
			},
		},
	}

	return &cronJob, nil
}

type BasicReplicationController struct{}

func (BasicReplicationController) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
		{"image-pull-policy", false},
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
// and returns ResourceList.
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
		result[resourceName] = resourceQuantity
	}
	return result, nil
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

// HandleResourceRequirements parses the limits and requests parameters if specified
// and returns ResourceRequirements.
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

// makePodSpec returns PodSpec filled with passed parameters.
func makePodSpec(params map[string]string, name string) (*v1.PodSpec, error) {
	stdin, err := GetBool(params, "stdin", false)
	if err != nil {
		return nil, err
	}

	tty, err := GetBool(params, "tty", false)
	if err != nil {
		return nil, err
	}

	resourceRequirements, err := HandleResourceRequirementsV1(params)
	if err != nil {
		return nil, err
	}

	spec := v1.PodSpec{
		Containers: []v1.Container{
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

	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}

	podSpec, err := makePodSpec(params, name)
	if err != nil {
		return nil, err
	}

	imagePullPolicy := v1.PullPolicy(params["image-pull-policy"])
	if err = updatePodContainers(params, args, envs, imagePullPolicy, podSpec); err != nil {
		return nil, err
	}

	if err := updatePodPorts(params, podSpec); err != nil {
		return nil, err
	}

	count32 := int32(count)
	controller := v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &count32,
			Selector: labels,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: *podSpec,
			},
		},
	}
	return &controller, nil
}

// updatePodContainers updates PodSpec.Containers with passed parameters.
func updatePodContainers(params map[string]string, args []string, envs []v1.EnvVar, imagePullPolicy v1.PullPolicy, podSpec *v1.PodSpec) error {
	if len(args) > 0 {
		command, err := GetBool(params, "command", false)
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

func (BasicPod) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"image", true},
		{"image-pull-policy", false},
		{"port", false},
		{"hostport", false},
		{"stdin", false},
		{"leave-stdin-open", false},
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

	stdin, err := GetBool(params, "stdin", false)
	if err != nil {
		return nil, err
	}
	leaveStdinOpen, err := GetBool(params, "leave-stdin-open", false)
	if err != nil {
		return nil, err
	}

	tty, err := GetBool(params, "tty", false)
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
	// TODO: Figure out why we set ImagePullPolicy here, whether we can make it
	// consistent with the other places imagePullPolicy is set using flag.
	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            name,
					Image:           params["image"],
					ImagePullPolicy: v1.PullIfNotPresent,
					Stdin:           stdin,
					StdinOnce:       !leaveStdinOpen && stdin,
					TTY:             tty,
					Resources:       resourceRequirements,
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
		if len(validation.IsCIdentifier(name)) != 0 {
			return nil, fmt.Errorf("invalid env: %v", env)
		}
		envVar := v1.EnvVar{Name: name, Value: value}
		envs = append(envs, envVar)
	}
	return envs, nil
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
