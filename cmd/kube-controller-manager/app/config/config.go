/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	kyaml "k8s.io/apimachinery/pkg/util/yaml"
	apiserver "k8s.io/apiserver/pkg/server"
	clientset "k8s.io/client-go/kubernetes"
	coreclientv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"
	utilpointer "k8s.io/utils/pointer"

	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	kubectrlmgrconfigscheme "k8s.io/kubernetes/pkg/controller/apis/config/scheme"
)

const (
	// KubeControllerManagerConfigMapName is the configMap name used to store controller manager config.
	KubeControllerManagerConfigMapName = "deployment-kube-controller-manager-config"
)

// Config is the main context object for the controller manager.
type Config struct {
	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration

	SecureServing *apiserver.SecureServingInfo
	// LoopbackClientConfig is a config for a privileged loopback connection
	LoopbackClientConfig *restclient.Config

	// TODO: remove deprecated insecure serving
	InsecureServing *apiserver.DeprecatedInsecureServingInfo
	Authentication  apiserver.AuthenticationInfo
	Authorization   apiserver.AuthorizationInfo

	// the general kube client
	Client clientset.Interface

	// the client only used for leader election
	LeaderElectionClient *clientset.Clientset

	// the rest config for the master
	Kubeconfig *restclient.Config

	// the event sink
	EventRecorder record.EventRecorder

	// ConfigFile is the location of the kube-controller manager server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the current kube-controller manager server's configuration will be written.
	WriteConfigTo string
}

type completedConfig struct {
	*Config
}

// CompletedConfig same as Config, just to swap private object.
type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() *CompletedConfig {
	cc := completedConfig{c}

	apiserver.AuthorizeClientBearerToken(c.LoopbackClientConfig, &c.Authentication, &c.Authorization)

	return &CompletedConfig{&cc}
}

// Sync should run once we detect the local disk config file changed.
func (c *Config) Sync() error {
	_, _, err := c.manageKubeControllerManagerConfigMaptoLatest(c.Client.CoreV1())
	if err != nil {
		return fmt.Errorf("update the kube-controller manager configmap %s/%s to latest error: %v", metav1.NamespaceDefault, KubeControllerManagerConfigMapName, err)
	}

	cg, err := c.Client.CoreV1().ConfigMaps(metav1.NamespaceDefault).Get(KubeControllerManagerConfigMapName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get configmap %s/%s error: %v", metav1.NamespaceDefault, KubeControllerManagerConfigMapName, err)
	}

	if err := ioutil.WriteFile(c.WriteConfigTo, []byte(cg.Data["config.yaml"]), 0644); err != nil {
		klog.Fatalf("Failed writing cluster config to file: %v", err)
	}
	klog.Infof("Wrote cluster config to: %s successfully\n", c.WriteConfigTo)

	return nil
}

// manageKubeControllerManagerConfigMaptoLatest update the kube-controller manager configmap to latest.
func (c *Config) manageKubeControllerManagerConfigMaptoLatest(client coreclientv1.ConfigMapsGetter) (*corev1.ConfigMap, bool, error) {
	configMap := makeConfigMap(metav1.NamespaceDefault, KubeControllerManagerConfigMapName, "config.yaml", "")
	clusterConfig, _ := ioutil.ReadFile(c.ConfigFile)
	localDiskConfig, _ := ioutil.ReadFile(c.WriteConfigTo)

	requiredConfigMap, _, err := MergeConfigMap(configMap, "config.yaml", clusterConfig, localDiskConfig)
	if err != nil {
		return nil, false, err
	}
	return ApplyConfigMap(client, requiredConfigMap)
}

// MergeConfigMap takes a configmap, the target key and a list of config configs to overlay on top of each other
// It returns the resultant configmap and a bool indicating if any changes were made to the configmap
func MergeConfigMap(configMap *corev1.ConfigMap, configKey string, configYAMLs ...[]byte) (*corev1.ConfigMap, bool, error) {
	configBytes, err := MergeProcessConfig(configYAMLs...)
	if err != nil {
		return nil, false, err
	}

	if reflect.DeepEqual(configMap.Data[configKey], configBytes) {
		return configMap, false, nil
	}

	ret := configMap.DeepCopy()
	ret.Data[configKey] = string(configBytes)

	return ret, true, nil
}

// MergeProcessConfig merges a series of config yaml files together with each later one overlaying all previous
func MergeProcessConfig(configYAMLs ...[]byte) ([]byte, error) {
	currentConfigYAML := configYAMLs[0]

	for _, currConfigYAML := range configYAMLs[1:] {
		prevConfigJSON, err := kyaml.ToJSON(currentConfigYAML)
		if err != nil {
			klog.Warning(err)
			// maybe it's just json
			prevConfigJSON = currentConfigYAML
		}
		prevConfig := map[string]interface{}{}
		if err := json.NewDecoder(bytes.NewBuffer(prevConfigJSON)).Decode(&prevConfig); err != nil {
			return nil, err
		}
		if len(currConfigYAML) > 0 {
			currConfigJSON, err := kyaml.ToJSON(currConfigYAML)
			if err != nil {
				klog.Warning(err)
				// maybe it's just json
				currConfigJSON = currConfigYAML
			}
			currConfig := map[string]interface{}{}
			if err := json.NewDecoder(bytes.NewBuffer(currConfigJSON)).Decode(&currConfig); err != nil {
				return nil, err
			}

			// protected against mismatched typemeta
			prevAPIVersion, _, _ := unstructured.NestedString(prevConfig, "apiVersion")
			prevKind, _, _ := unstructured.NestedString(prevConfig, "kind")
			currAPIVersion, _, _ := unstructured.NestedString(currConfig, "apiVersion")
			currKind, _, _ := unstructured.NestedString(currConfig, "kind")
			currGVKSet := len(currAPIVersion) > 0 || len(currKind) > 0
			gvkMismatched := currAPIVersion != prevAPIVersion || currKind != prevKind
			if currGVKSet && gvkMismatched {
				return nil, fmt.Errorf("%v/%v does not equal %v/%v", currAPIVersion, currKind, prevAPIVersion, prevKind)
			}

			if err := mergeConfig(prevConfig, currConfig, ""); err != nil {
				return nil, err
			}
		}

		currentConfigYAML, err = runtime.Encode(unstructured.UnstructuredJSONScheme, &unstructured.Unstructured{Object: prevConfig})
		if err != nil {
			return nil, err
		}
	}

	return currentConfigYAML, nil
}

// mergeConfig overwrites entries in curr by additional.  It modifies curr.
func mergeConfig(curr, additional map[string]interface{}, currentPath string) error {
	for additionalKey, additionalVal := range additional {
		fullKey := currentPath + "." + additionalKey

		// new added
		currVal, ok := curr[additionalKey]
		if !ok {
			curr[additionalKey] = additionalVal
			continue
		}

		// only some scalars are accepted
		switch castVal := additionalVal.(type) {
		case map[string]interface{}:
			currValAsMap, ok := currVal.(map[string]interface{})
			if !ok {
				currValAsMap = map[string]interface{}{}
				curr[additionalKey] = currValAsMap
			}
			err := mergeConfig(currValAsMap, castVal, fullKey)
			if err != nil {
				return err
			}
			continue

		default:
			if err := unstructured.SetNestedField(curr, castVal, additionalKey); err != nil {
				return err
			}
		}

	}

	return nil
}

// ApplyConfigMap merges objectmeta, requires data
func ApplyConfigMap(client coreclientv1.ConfigMapsGetter, required *corev1.ConfigMap) (*corev1.ConfigMap, bool, error) {
	existing, err := client.ConfigMaps(required.Namespace).Get(required.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		actual, err := client.ConfigMaps(required.Namespace).Create(required)
		return actual, true, err
	}
	if err != nil {
		return nil, false, err
	}

	modified := utilpointer.BoolPtr(false)
	EnsureObjectMeta(modified, &existing.ObjectMeta, required.ObjectMeta)
	dataSame := equality.Semantic.DeepEqual(existing.Data, required.Data)
	if dataSame && !*modified {
		return existing, false, nil
	}
	existing.Data = required.Data

	actual, err := client.ConfigMaps(required.Namespace).Update(existing)
	return actual, true, err
}

// makeConfigMap construct a new ConfigMap object.
func makeConfigMap(namespace, name, key, value string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string]string{
			key: value,
		},
	}
}

// WriteConfigFile writes the config into the given file name as YAML.
func WriteConfigFile(fileName string, cfg *kubectrlmgrconfig.KubeControllerManagerConfiguration) error {
	var encoder runtime.Encoder
	mediaTypes := kubectrlmgrconfigscheme.Codecs.SupportedMediaTypes()
	for _, info := range mediaTypes {
		if info.MediaType == "application/yaml" {
			encoder = info.Serializer
			break
		}
	}
	if encoder == nil {
		return fmt.Errorf("unable to locate yaml encoder")
	}
	encoder = runtimejson.NewYAMLSerializer(runtimejson.DefaultMetaFactory, kubectrlmgrconfigscheme.Scheme, kubectrlmgrconfigscheme.Scheme)
	encoder = kubectrlmgrconfigscheme.Codecs.EncoderForVersion(encoder, kubectrlmgrconfigv1alpha1.SchemeGroupVersion)

	configFile, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer configFile.Close()
	if err := encoder.Encode(cfg, configFile); err != nil {
		return err
	}

	return nil
}

// EnsureObjectMeta writes namespace, name, labels, and annotations.  Don't set other things here.
// TODO finalizer support maybe?
func EnsureObjectMeta(modified *bool, existing *metav1.ObjectMeta, required metav1.ObjectMeta) {
	SetStringIfSet(modified, &existing.Namespace, required.Namespace)
	SetStringIfSet(modified, &existing.Name, required.Name)
	MergeMap(modified, &existing.Labels, required.Labels)
	MergeMap(modified, &existing.Annotations, required.Annotations)
}

// SetStringIfSet modify the string field if set.
func SetStringIfSet(modified *bool, existing *string, required string) {
	if len(required) == 0 {
		return
	}
	if required != *existing {
		*existing = required
		*modified = true
	}
}

// MergeMap modify the map field if set.
func MergeMap(modified *bool, existing *map[string]string, required map[string]string) {
	if *existing == nil {
		*existing = map[string]string{}
	}
	for k, v := range required {
		if existingV, ok := (*existing)[k]; !ok || v != existingV {
			*modified = true
			(*existing)[k] = v
		}
	}
}
