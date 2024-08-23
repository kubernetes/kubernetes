/*
Copyright 2023 The Kubernetes Authors.

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

package cel

import (
	"bytes"
	"context"
	"encoding/csv"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/admission/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	admissionregistrationv1alpha1apis "k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1"
	admissionregistrationv1beta1apis "k8s.io/kubernetes/pkg/apis/admissionregistration/v1beta1"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
)

const (
	beginSentinel   = "###___BEGIN_SENTINEL___###"
	recordSeparator = `###$$$###`
)

// Policy registration helpers
var testSpec admissionregistration.ValidatingAdmissionPolicy = admissionregistration.ValidatingAdmissionPolicy{
	Spec: admissionregistration.ValidatingAdmissionPolicySpec{
		ParamKind: &admissionregistration.ParamKind{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		Variables: []admissionregistration.Variable{
			{
				Name:       "shouldFail",
				Expression: `true`,
			},
			{
				Name:       "resourceGroup",
				Expression: `has(request.resource.group) ? request.resource.group : ""`,
			},
			{
				Name:       "resourceVersion",
				Expression: `has(request.resource.version) ? request.resource.version : ""`,
			},
			{
				Name:       "resourceResource",
				Expression: `has(request.resource.resource) ? request.resource.resource : ""`,
			},
			{
				Name:       "subresource",
				Expression: `has(request.subResource) ? request.subResource : ""`,
			},
			{
				Name:       "operation",
				Expression: `has(request.operation) ? request.operation : ""`,
			},
			{
				Name:       "name",
				Expression: `has(request.name) ? request.name : ""`,
			},
			{
				Name:       "namespaceName",
				Expression: `has(request.namespace) ? request.namespace : ""`,
			},
			{
				Name:       "objectExists",
				Expression: `object != null ? "true" : "false"`,
			},
			{
				Name:       "objectAPIVersion",
				Expression: `(object != null && has(object.apiVersion)) ? object.apiVersion : ""`,
			},
			{
				Name:       "objectKind",
				Expression: `(object != null && has(object.kind)) ? object.kind : ""`,
			},
			{
				Name:       "oldObjectExists",
				Expression: `oldObject != null ? "true" : "false"`,
			},
			{
				Name:       "oldObjectAPIVersion",
				Expression: `(oldObject != null && has(oldObject.apiVersion)) ? oldObject.apiVersion : ""`,
			},
			{
				Name:       "oldObjectKind",
				Expression: `(oldObject != null && has(oldObject.kind)) ? oldObject.kind : ""`,
			},
			{
				Name:       "optionsExists",
				Expression: `(has(request.options) && request.options != null) ? "true" : "false"`,
			},
			{
				Name:       "optionsKind",
				Expression: `(has(request.options) && has(request.options.kind)) ? request.options.kind : ""`,
			},
			{
				Name:       "optionsAPIVersion",
				Expression: `(has(request.options) && has(request.options.apiVersion)) ? request.options.apiVersion : ""`,
			},
			{
				Name:       "paramsPhase",
				Expression: `params.data.phase`,
			},
			{
				Name:       "paramsVersion",
				Expression: `params.data.version`,
			},
			{
				Name:       "paramsConvert",
				Expression: `params.data.convert`,
			},
		},
		// Would be nice to use CEL to create a single map
		// and stringify it. Unfortunately those library functions
		// are not yet available, so we must create a map
		// like so
		Validations: []admissionregistration.Validation{
			{
				// newlines forbidden so use recordSeparator
				Expression:        "!variables.shouldFail",
				MessageExpression: `"` + beginSentinel + `resourceGroup,resourceVersion,resourceResource,subresource,operation,name,namespace,objectExists,objectKind,objectAPIVersion,oldObjectExists,oldObjectKind,oldObjectAPIVersion,optionsExists,optionsKind,optionsAPIVersion,paramsPhase,paramsVersion,paramsConvert` + recordSeparator + `"+variables.resourceGroup + "," + variables.resourceVersion + "," + variables.resourceResource + "," + variables.subresource + "," + variables.operation + "," + variables.name + "," + variables.namespaceName + "," + variables.objectExists + "," + variables.objectKind + "," + variables.objectAPIVersion + "," + variables.oldObjectExists + "," + variables.oldObjectKind + "," + variables.oldObjectAPIVersion + "," + variables.optionsExists + "," + variables.optionsKind + "," + variables.optionsAPIVersion + "," + variables.paramsPhase + "," + variables.paramsVersion + "," + variables.paramsConvert`,
			},
		},
		MatchConditions: []admissionregistration.MatchCondition{
			{
				Name:       "testclient-only",
				Expression: `request.userInfo.username == "` + testClientUsername + `"`,
			},
			{
				Name:       "ignore-test-config",
				Expression: `object == null || !has(object.metadata) || !has(object.metadata.annotations) || !has(object.metadata.annotations.skipMatch) || object.metadata.annotations.skipMatch != "yes"`,
			},
		},
	},
}

func createV1beta1ValidatingPolicyAndBinding(client clientset.Interface, convertedRules []admissionregistrationv1beta1.NamedRuleWithOperations) error {
	denyAction := admissionregistrationv1beta1.DenyAction
	exact := admissionregistrationv1beta1.Exact
	equivalent := admissionregistrationv1beta1.Equivalent

	var outSpec admissionregistrationv1beta1.ValidatingAdmissionPolicy
	if err := admissionregistrationv1beta1apis.Convert_admissionregistration_ValidatingAdmissionPolicy_To_v1beta1_ValidatingAdmissionPolicy(&testSpec, &outSpec, nil); err != nil {
		return err
	}

	exactPolicyTemplate := outSpec.DeepCopy()
	convertedPolicyTemplate := outSpec.DeepCopy()

	exactPolicyTemplate.SetName("test-policy-v1beta1")
	exactPolicyTemplate.Spec.MatchConstraints = &admissionregistrationv1beta1.MatchResources{
		ResourceRules: []admissionregistrationv1beta1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*/*"}},
				},
			},
		},
		MatchPolicy: &exact,
	}

	convertedPolicyTemplate.SetName("test-policy-v1beta1-convert")
	convertedPolicyTemplate.Spec.MatchConstraints = &admissionregistrationv1beta1.MatchResources{
		ResourceRules: convertedRules,
		MatchPolicy:   &equivalent,
	}

	exactPolicy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(context.TODO(), exactPolicyTemplate, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	convertPolicy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(context.TODO(), convertedPolicyTemplate, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	// Create a param that holds the options for this
	configuration, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-policy-v1beta1-param",
			Namespace: "default",
			Annotations: map[string]string{
				"skipMatch": "yes",
			},
		},
		Data: map[string]string{
			"version": "v1beta1",
			"phase":   validation,
			"convert": "false",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	configurationConvert, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-policy-v1beta1-convert-param",
			Namespace: "default",
			Annotations: map[string]string{
				"skipMatch": "yes",
			},
		},
		Data: map[string]string{
			"version": "v1beta1",
			"phase":   validation,
			"convert": "true",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	_, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), &admissionregistrationv1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy-v1beta1-binding",
		},
		Spec: admissionregistrationv1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        exactPolicy.GetName(),
			ValidationActions: []admissionregistrationv1beta1.ValidationAction{admissionregistrationv1beta1.Warn},
			ParamRef: &admissionregistrationv1beta1.ParamRef{
				Name:                    configuration.GetName(),
				Namespace:               configuration.GetNamespace(),
				ParameterNotFoundAction: &denyAction,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	_, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), &admissionregistrationv1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy-v1beta1-convert-binding",
		},
		Spec: admissionregistrationv1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        convertPolicy.GetName(),
			ValidationActions: []admissionregistrationv1beta1.ValidationAction{admissionregistrationv1beta1.Warn},
			ParamRef: &admissionregistrationv1beta1.ParamRef{
				Name:                    configurationConvert.GetName(),
				Namespace:               configurationConvert.GetNamespace(),
				ParameterNotFoundAction: &denyAction,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	return nil
}

func createV1alpha1ValidatingPolicyAndBinding(client clientset.Interface, convertedRules []admissionregistrationv1alpha1.NamedRuleWithOperations) error {
	exact := admissionregistrationv1alpha1.Exact
	equivalent := admissionregistrationv1alpha1.Equivalent
	denyAction := admissionregistrationv1alpha1.DenyAction

	var outSpec admissionregistrationv1alpha1.ValidatingAdmissionPolicy
	if err := admissionregistrationv1alpha1apis.Convert_admissionregistration_ValidatingAdmissionPolicy_To_v1alpha1_ValidatingAdmissionPolicy(&testSpec, &outSpec, nil); err != nil {
		return err
	}

	exactPolicyTemplate := outSpec.DeepCopy()
	convertedPolicyTemplate := outSpec.DeepCopy()

	exactPolicyTemplate.SetName("test-policy-v1alpha1")
	exactPolicyTemplate.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*/*"}},
				},
			},
		},
		MatchPolicy: &exact,
	}

	convertedPolicyTemplate.SetName("test-policy-v1alpha1-convert")
	convertedPolicyTemplate.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		ResourceRules: convertedRules,
		MatchPolicy:   &equivalent,
	}

	exactPolicy, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), exactPolicyTemplate, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	convertPolicy, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), convertedPolicyTemplate, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	// Create a param that holds the options for this
	configuration, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-policy-v1alpha1-param",
			Namespace: "default",
			Annotations: map[string]string{
				"skipMatch": "yes",
			},
		},
		Data: map[string]string{
			"version": "v1alpha1",
			"phase":   validation,
			"convert": "false",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	configurationConvert, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-policy-v1alpha1-convert-param",
			Namespace: "default",
			Annotations: map[string]string{
				"skipMatch": "yes",
			},
		},
		Data: map[string]string{
			"version": "v1alpha1",
			"phase":   validation,
			"convert": "true",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), &admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy-v1alpha1-binding",
		},
		Spec: admissionregistrationv1alpha1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        exactPolicy.GetName(),
			ValidationActions: []admissionregistrationv1alpha1.ValidationAction{admissionregistrationv1alpha1.Warn},
			ParamRef: &admissionregistrationv1alpha1.ParamRef{
				Name:                    configuration.GetName(),
				Namespace:               configuration.GetNamespace(),
				ParameterNotFoundAction: &denyAction,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), &admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy-v1alpha1-convert-binding",
		},
		Spec: admissionregistrationv1alpha1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        convertPolicy.GetName(),
			ValidationActions: []admissionregistrationv1alpha1.ValidationAction{admissionregistrationv1alpha1.Warn},
			ParamRef: &admissionregistrationv1alpha1.ParamRef{
				Name:                    configurationConvert.GetName(),
				Namespace:               configurationConvert.GetNamespace(),
				ParameterNotFoundAction: &denyAction,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	return nil
}

// This test shows that policy intercepts all requests for all resources,
// subresources, verbs, and input versions of policy/binding.
//
// This test tries to mirror very closely the same test for webhook admission
// test/integration/apiserver/admissionwebhook/admission_test.go testWebhookAdmission
func TestPolicyAdmission(t *testing.T) {
	holder := &policyExpectationHolder{
		holder: holder{
			t:                 t,
			gvrToConvertedGVR: map[metav1.GroupVersionResource]metav1.GroupVersionResource{},
			gvrToConvertedGVK: map[metav1.GroupVersionResource]schema.GroupVersionKind{},
		},
	}

	server := apiservertesting.StartTestServerOrDie(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
		// turn off admission plugins that add finalizers
		"--disable-admission-plugins=ServiceAccount,StorageObjectInUseProtection",
		// force enable all resources so we can check storage.
		"--runtime-config=api/all=true",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// Create admission policy & binding that match everything
	clientConfig := server.ClientConfig
	clientConfig.Impersonate.UserName = testClientUsername
	clientConfig.Impersonate.Groups = []string{"system:masters", "system:authenticated"}
	clientConfig.WarningHandler = holder
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// create CRDs
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// gather resources to test
	dynamicClient, err := dynamic.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	_, resources, err := client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources with error: %+v", err)
	}

	gvrsToTest := []schema.GroupVersionResource{}
	resourcesByGVR := map[schema.GroupVersionResource]metav1.APIResource{}

	for _, list := range resources {
		defaultGroupVersion, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			t.Errorf("Failed to get GroupVersion for: %+v", list)
			continue
		}
		for _, resource := range list.APIResources {
			if resource.Group == "" {
				resource.Group = defaultGroupVersion.Group
			}
			if resource.Version == "" {
				resource.Version = defaultGroupVersion.Version
			}
			gvr := defaultGroupVersion.WithResource(resource.Name)
			resourcesByGVR[gvr] = resource
			if shouldTestResource(gvr, resource) {
				gvrsToTest = append(gvrsToTest, gvr)
			}
		}
	}

	sort.SliceStable(gvrsToTest, func(i, j int) bool {
		if gvrsToTest[i].Group < gvrsToTest[j].Group {
			return true
		}
		if gvrsToTest[i].Group > gvrsToTest[j].Group {
			return false
		}
		if gvrsToTest[i].Version < gvrsToTest[j].Version {
			return true
		}
		if gvrsToTest[i].Version > gvrsToTest[j].Version {
			return false
		}
		if gvrsToTest[i].Resource < gvrsToTest[j].Resource {
			return true
		}
		if gvrsToTest[i].Resource > gvrsToTest[j].Resource {
			return false
		}
		return true
	})

	// map unqualified resource names to the fully qualified resource we will expect to be converted to
	// Note: this only works because there are no overlapping resource names in-process that are not co-located
	convertedResources := map[string]schema.GroupVersionResource{}
	// build the webhook rules enumerating the specific group/version/resources we want
	convertedV1beta1Rules := []admissionregistrationv1beta1.NamedRuleWithOperations{}
	convertedV1alpha1Rules := []admissionregistrationv1alpha1.NamedRuleWithOperations{}
	for _, gvr := range gvrsToTest {
		metaGVR := metav1.GroupVersionResource{Group: gvr.Group, Version: gvr.Version, Resource: gvr.Resource}

		convertedGVR, ok := convertedResources[gvr.Resource]
		if !ok {
			// this is the first time we've seen this resource
			// record the fully qualified resource we expect
			convertedGVR = gvr
			convertedResources[gvr.Resource] = gvr
			// add an admission rule indicating we can receive this version
			convertedV1beta1Rules = append(convertedV1beta1Rules, admissionregistrationv1beta1.NamedRuleWithOperations{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1beta1.OperationType{admissionregistrationv1beta1.OperationAll},
					Rule:       admissionregistrationv1beta1.Rule{APIGroups: []string{gvr.Group}, APIVersions: []string{gvr.Version}, Resources: []string{gvr.Resource}},
				},
			})
			convertedV1alpha1Rules = append(convertedV1alpha1Rules, admissionregistrationv1alpha1.NamedRuleWithOperations{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1alpha1.OperationType{admissionregistrationv1alpha1.OperationAll},
					Rule:       admissionregistrationv1alpha1.Rule{APIGroups: []string{gvr.Group}, APIVersions: []string{gvr.Version}, Resources: []string{gvr.Resource}},
				},
			})
		}

		// record the expected resource and kind
		holder.gvrToConvertedGVR[metaGVR] = metav1.GroupVersionResource{Group: convertedGVR.Group, Version: convertedGVR.Version, Resource: convertedGVR.Resource}
		holder.gvrToConvertedGVK[metaGVR] = schema.GroupVersionKind{Group: resourcesByGVR[convertedGVR].Group, Version: resourcesByGVR[convertedGVR].Version, Kind: resourcesByGVR[convertedGVR].Kind}
	}

	if err := createV1alpha1ValidatingPolicyAndBinding(client, convertedV1alpha1Rules); err != nil {
		t.Fatal(err)
	}

	if err := createV1beta1ValidatingPolicyAndBinding(client, convertedV1beta1Rules); err != nil {
		t.Fatal(err)
	}

	// Allow the policy & binding to establish
	time.Sleep(1 * time.Second)

	start := time.Now()
	count := 0

	// Test admission on all resources, subresources, and verbs
	for _, gvr := range gvrsToTest {
		resource := resourcesByGVR[gvr]
		t.Run(gvr.Group+"."+gvr.Version+"."+strings.ReplaceAll(resource.Name, "/", "."), func(t *testing.T) {
			for _, verb := range []string{"create", "update", "patch", "connect", "delete", "deletecollection"} {
				if shouldTestResourceVerb(gvr, resource, verb) {
					t.Run(verb, func(t *testing.T) {
						count++
						holder.reset(t)
						testFunc := getTestFunc(gvr, verb)
						testFunc(&testContext{
							t:               t,
							admissionHolder: holder,
							client:          dynamicClient,
							clientset:       client,
							verb:            verb,
							gvr:             gvr,
							resource:        resource,
							resources:       resourcesByGVR,
						})
						holder.verify(t)
					})
				}
			}
		})
	}

	if count >= 10 {
		duration := time.Since(start)
		perResourceDuration := time.Duration(int(duration) / count)
		if perResourceDuration >= 150*time.Millisecond {
			t.Errorf("expected resources to process in < 150ms, average was %v", perResourceDuration)
		}
	}
}

// Policy admission holder for test framework

type policyExpectationHolder struct {
	holder
	warningLock sync.Mutex
	warnings    []string
}

func (p *policyExpectationHolder) reset(t *testing.T) {
	p.warningLock.Lock()
	defer p.warningLock.Unlock()
	p.warnings = nil

	p.holder.reset(t)

}
func (p *policyExpectationHolder) expect(gvr schema.GroupVersionResource, gvk, optionsGVK schema.GroupVersionKind, operation v1beta1.Operation, name, namespace string, object, oldObject, options bool) {
	p.holder.expect(gvr, gvk, optionsGVK, operation, name, namespace, object, oldObject, options)

	p.lock.Lock()
	defer p.lock.Unlock()
	// Set up the recorded map with nil records for all combinations
	p.recorded = map[webhookOptions]*admissionRequest{}
	for _, phase := range []string{validation} {
		for _, converted := range []bool{true, false} {
			for _, version := range []string{"v1alpha1", "v1beta1"} {
				p.recorded[webhookOptions{version: version, phase: phase, converted: converted}] = nil
			}
		}
	}
}

func (p *policyExpectationHolder) verify(t *testing.T) {
	p.warningLock.Lock()
	defer p.warningLock.Unlock()

	// Process all detected warnings and record in the nested handler
	for _, w := range p.warnings {
		var currentRequest *admissionRequest
		var currentParams webhookOptions
		if idx := strings.Index(w, beginSentinel); idx >= 0 {

			csvData := strings.ReplaceAll(w[idx+len(beginSentinel):], recordSeparator, "\n")

			b := bytes.Buffer{}
			b.WriteString(csvData)
			reader := csv.NewReader(&b)
			csvRecords, err := reader.ReadAll()
			if err != nil {
				t.Fatal(err)
				return
			}

			mappedCSV := []map[string]string{}
			var header []string
			for line, record := range csvRecords {
				if line == 0 {
					header = record
				} else {
					line := map[string]string{}
					for i := 0; i < len(record); i++ {
						line[header[i]] = record[i]
					}
					mappedCSV = append(mappedCSV, line)
				}
			}

			if len(mappedCSV) != 1 {
				t.Fatal("incorrect # CSV elements in parsed warning")
				return
			}

			data := mappedCSV[0]
			currentRequest = &admissionRequest{
				Operation: data["operation"],
				Name:      data["name"],
				Namespace: data["namespace"],
				Resource: metav1.GroupVersionResource{
					Group:    data["resourceGroup"],
					Version:  data["resourceVersion"],
					Resource: data["resourceResource"],
				},
				SubResource: data["subresource"],
			}
			currentParams = webhookOptions{
				version:   data["paramsVersion"],
				phase:     data["paramsPhase"],
				converted: data["paramsConvert"] == "true",
			}

			if e, ok := data["objectExists"]; ok && e == "true" {
				currentRequest.Object.Object = &unstructured.Unstructured{}
				currentRequest.Object.Object.(*unstructured.Unstructured).SetAPIVersion(data["objectAPIVersion"])
				currentRequest.Object.Object.(*unstructured.Unstructured).SetKind(data["objectKind"])
			}

			if e, ok := data["oldObjectExists"]; ok && e == "true" {
				currentRequest.OldObject.Object = &unstructured.Unstructured{}
				currentRequest.OldObject.Object.(*unstructured.Unstructured).SetAPIVersion(data["oldObjectAPIVersion"])
				currentRequest.OldObject.Object.(*unstructured.Unstructured).SetKind(data["oldObjectKind"])
			}

			if e, ok := data["optionsExists"]; ok && e == "true" {
				currentRequest.Options.Object = &unstructured.Unstructured{}
				currentRequest.Options.Object.(*unstructured.Unstructured).SetAPIVersion(data["optionsAPIVersion"])
				currentRequest.Options.Object.(*unstructured.Unstructured).SetKind(data["optionsKind"])
			}

			p.holder.record(currentParams.version, currentParams.phase, currentParams.converted, currentRequest)
		}
	}

	p.holder.verify(t)
}

func (p *policyExpectationHolder) HandleWarningHeader(code int, agent string, message string) {
	if code != 299 || len(message) == 0 {
		return
	}
	p.warningLock.Lock()
	defer p.warningLock.Unlock()
	p.warnings = append(p.warnings, message)
}
