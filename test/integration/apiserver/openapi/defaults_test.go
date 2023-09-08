package openapi

import (
	"errors"
	"fmt"
	"math/rand"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi3"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
	k8stest "k8s.io/kubernetes/pkg/api/testing"

	// Initialize install packages
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/test/integration/framework"

	// Not included in testing package. Should it be added?
	_ "k8s.io/kubernetes/pkg/apis/apiserverinternal/install"
)

func gvk(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   group,
		Version: version,
		Kind:    kind,
	}
}

func exception(path string, detail string) []string {
	return strings.Split(path, ".")
}

// !TODO: make test to check that not specifying a required field results in error
// !TODO: fuzz tests?
var exceptions map[schema.GroupVersionKind][][]string = map[schema.GroupVersionKind][][]string{
	//
	gvk("", "v1", "Pod"): {
		exception("Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),

		exception("Spec.Containers.ResizePolicy", "requires conditional defaults (cel)"),
		exception("Spec.EphemeralContainers.ResizePolicy", "requires conditional defaults (cel)"),

		exception("Spec.InitContainers.LivenessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Spec.InitContainers.ReadinessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Spec.InitContainers.StartupProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),

		exception("Spec.Containers.LivenessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Spec.Containers.ReadinessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Spec.Containers.StartupProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
	},

	gvk("", "v1", "PodTemplate"): {
		exception("Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),

		exception("Template.Spec.InitContainers.LivenessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Template.Spec.InitContainers.ReadinessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Template.Spec.InitContainers.StartupProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),

		exception("Template.Spec.Containers.LivenessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Template.Spec.Containers.ReadinessProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
		exception("Template.Spec.Containers.StartupProbe.TerminationGracePeriodSeconds", "requires conditional defaults (cel)"),
	},
	gvk("", "v1", "Service"): {
		exception("Spec.ExternalTrafficPolicy", "requires conditional defaults (cel)"),
		exception("Spec.SessionAffinityConfig", "requires conditional defaults (cel)"),
		exception("Status.LoadBalancer.Ingress.IPMode", "requires conditional defaults (cel)"),
	},
	gvk("apps", "v1", "Deployment"): {
		exception("Spec.Strategy.RollingUpdate", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},
	gvk("apps", "v1", "DaemonSet"): {
		exception("Spec.UpdateStrategy.RollingUpdate", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},
	gvk("apps", "v1", "StatefulSet"): {
		exception("Spec.PersistentVolumeClaimRetentionPolicy", "requires conditional defaults (feature-gated)"),
		exception("Spec.UpdateStrategy.RollingUpdate", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},
	gvk("autoscaling", "v2", "HorizontalPodAutoscaler"): {
		exception("Spec.Behavior", "requires conditional defaults (cel)"),
	},
	// Unechecked default: if HostNetwork enabled for container, InitContainers and Containers ContainerPort defaults to HostPort

	// Types which are used in more than one place cannot be defaulted with
	// our declarative marker comments + code generation system
	gvk("batch", "v1", "CronJob"): {
		exception("Spec.JobTemplate.Spec", "shared template type"),
	},
	gvk("batch", "v1", "Job"): {
		exception("Spec", "shared template type"),
	},
	gvk("apps", "v1", "ReplicaSet"): {
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},

	gvk("apps", "v1", "ReplicaSet"): {
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},

	gvk("", "v1", "ReplicationController"): {
		exception("Spec.Template.Spec.Volumes.VolumeSource.EmptyDir", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.InitContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.Containers.ImagePullPolicy", "requires conditional defaults (cel)"),
		exception("Spec.Template.Spec.EphemeralContainers.ImagePullPolicy", "requires conditional defaults (cel)"),
	},

	gvk("", "v1", "PersistentVolume"): {
		exception("Spec.VolumeMode", "shared template type"),
	},

	// Exceptions limited by code-generator must-default
	// Defaults cannot be set onto non-pointers unless they are omit-empty
	// Need to remove limitation from code-generator
	gvk("flowcontrol.apiserver.k8s.io", "v1beta1", "FlowSchema"): {
		exception("Spec.MatchingPrecedence", "code-generator mustEnforceDefault"),
	},
	gvk("flowcontrol.apiserver.k8s.io", "v1beta2", "FlowSchema"): {
		exception("Spec.MatchingPrecedence", "code-generator mustEnforceDefault"),
	},
	gvk("flowcontrol.apiserver.k8s.io", "v1beta3", "FlowSchema"): {
		exception("Spec.MatchingPrecedence", "code-generator mustEnforceDefault"),
	},
	gvk("flowcontrol.apiserver.k8s.io", "v1beta1", "PriorityLevelConfiguration"): {
		exception("Spec.Limited.AssuredConcurrencyShares", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.Queues", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.HandSize", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.QueueLengthLimit", "code-generator mustEnforceDefault"),
	},
	gvk("flowcontrol.apiserver.k8s.io", "v1beta2", "PriorityLevelConfiguration"): {
		exception("Spec.Limited.AssuredConcurrencyShares", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.Queues", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.HandSize", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.QueueLengthLimit", "code-generator mustEnforceDefault"),
	},
	gvk("flowcontrol.apiserver.k8s.io", "v1beta3", "PriorityLevelConfiguration"): {
		exception("Spec.Limited.NominalConcurrencyShares", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.Queues", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.HandSize", "code-generator mustEnforceDefault"),
		exception("Spec.Limited.LimitResponse.Queuing.QueueLengthLimit", "code-generator mustEnforceDefault"),
	},
	gvk("rbac.authorization.k8s.io", "v1", "RoleBinding"): {
		exception("RoleRef.APIGroup", "code-generator mustEnforceDefault (non-omitempty)"),
	},
	gvk("rbac.authorization.k8s.io", "v1", "ClusterRoleBinding"): {
		exception("RoleRef.APIGroup", "code-generator mustEnforceDefault (non-omitempty)"),
	},
	gvk("apiextensions.k8s.io", "v1", "CustomResourceDefinition"): {
		exception("Spec.Conversion.Strategy", "code-generator mustEnforceDefault (non-omitempty)"),
	},
}

// Finds all unconditional and some conditional defaults that are being applied
// with native defaulting but are not published in the schema.
//
// There may be some conditional defaults which are not detected by this method
// due to testing for an enum value or other conditions before setting the default.
// Those can be snuffed out by adding a fuzz test here, and ensuring 100% default
// line coverage for this test.
func TestDefaulting(t *testing.T) {
	//!TODO: run test both with and without feature gates
	//!TODO: maybe use resolver instead of server: 	resolver2 "k8s.io/apiserver/pkg/cel/openapi/resolver"
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature("AllAlpha"), true)()
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), []string{
		"--runtime-config=api/all=true",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	clientset, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	openAPIClient := openapi3.NewRoot(openapi.NewClient(clientset.RESTClient()))
	gvs, err := openAPIClient.GroupVersions()
	if err != nil {
		t.Fatal(err)
	}

	funcs := fuzzer.MergeFuzzerFuncs(k8stest.FuzzerFuncs, apiextensionsfuzzer.Funcs)

	// Given a list of gvks, returns the first element that is registered to
	// a scheme, and that scheme.
	//!TODO: is it possible to merge the schemes so we can just use one?
	findSchemeForGVKAliases := func(gvks []metav1.GroupVersionKind) (*runtime.Scheme, *fuzz.Fuzzer, schema.GroupVersionKind) {
		for _, gvk := range gvks {
			gvk := schema.GroupVersionKind(gvk)
			for _, currentScheme := range []*runtime.Scheme{
				legacyscheme.Scheme, aggregatorscheme.Scheme, extensionsapiserver.Scheme,
			} {
				if _, contains := currentScheme.AllKnownTypes()[gvk]; contains {
					return currentScheme, fuzzer.FuzzerFor(funcs, rand.NewSource(rand.Int63()), serializer.NewCodecFactory(currentScheme)), gvk
				}
			}
		}
		return nil, nil, schema.GroupVersionKind{}
	}

	specs := map[schema.GroupVersion]*spec3.OpenAPI{}
	for _, gv := range gvs {
		doc, err := openAPIClient.GVSpec(gv)
		if err != nil {
			t.Fatal(err)
		}

		// Temporary patch until we add support into code generator to mark these
		// properly as nullable
		if t, ok := doc.Components.Schemas["io.k8s.apimachinery.pkg.apis.meta.v1.Time"]; ok {
			t.Nullable = true
		}

		if t, ok := doc.Components.Schemas["io.k8s.apimachinery.pkg.apis.meta.v1.MicroTime"]; ok {
			t.Nullable = true
		}

		if t, ok := doc.Components.Schemas["io.k8s.apimachinery.pkg.runtime.RawExtension"]; ok {
			t.Nullable = true
		}

		if t, ok := doc.Components.Schemas["io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1.JSON"]; ok {
			t.Nullable = true
		}

		if t, ok := doc.Components.Schemas["io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1.JSONSchemaPropsOrStringArray"]; ok {
			t.Nullable = true
		}

		specs[gv] = doc
	}

	for _, gv := range gvs {
		gv := gv
		t.Run(gv.Group+":"+gv.Version, func(t *testing.T) {
			t.Parallel()
			doc := specs[gv]

			for _, sch := range doc.Components.Schemas {
				sch := sch
				var gvks []metav1.GroupVersionKind
				if err := sch.Extensions.GetObject("x-kubernetes-group-version-kind", &gvks); err != nil {
					t.Error(err)
					continue
				} else if len(gvks) == 0 {
					// Skip intermediate types. They will be checked if referered
					// to by a top-level type.
					continue
				}

				scheme, fuzzer, schemeKnownGVK := findSchemeForGVKAliases(gvks)
				if scheme == nil {
					t.Errorf("Found no scheme associated with any of the OpenAPI GVKs: %v", gvks)
					continue
				} else if strings.HasSuffix(schemeKnownGVK.Kind, "List") {
					// Skip lists since they typically yield duplicate errors
					// and do not have any specific defaulting of their own
					continue
				} else if strings.HasSuffix(schemeKnownGVK.Kind, "Options") {
					// Skip options since they do not have an internal type
					continue
				}

				t.Run(schemeKnownGVK.Kind, func(t *testing.T) {
					t.Parallel()
					example, err := scheme.New(schemeKnownGVK)
					if err != nil {
						t.Fatal(err)
					}

					if err := verify(schemeKnownGVK, example, sch, doc.Components.Schemas, scheme); err != nil {
						t.Fatal(err)
					}

					if err := verifyWithFuzzer(schemeKnownGVK, sch, doc.Components.Schemas, scheme, fuzzer); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func fillEmptyPointers(val reflect.Value, limit int) {
	if limit <= 0 {
		return
	} else if val.Kind() == reflect.Struct && val.CanAddr() {
		val = val.Addr()
	}

	typ := val.Type()
	if typ.Kind() != reflect.Pointer || typ.Elem().Kind() != reflect.Struct {

		// Set non-zero values for primitive fields
		switch typ.Kind() {
		case reflect.Int:
			val.Set(reflect.ValueOf(rand.Int()))
		case reflect.Bool:
			val.Set(reflect.ValueOf(true))
		case reflect.String:
			//!TODO: would be nice if we could evaluate enum cases here
		}

		return
	}
	structType := typ.Elem()

	for i := 0; i < structType.NumField(); i++ {
		fld := val.Elem().Field(i)
		fldType := structType.Field(i)
		if !fldType.IsExported() {
			continue
		}

		var elemIsStruct func(t reflect.Type) bool
		elemIsStruct = func(t reflect.Type) bool {
			switch t.Kind() {
			case reflect.Struct:
				return true
			case reflect.Pointer:
				return elemIsStruct(t.Elem())
			case reflect.Slice:
				return elemIsStruct(t.Elem())
			case reflect.Map:
				return elemIsStruct(t.Elem())
			default:
				return false
			}
		}

		switch fldType.Type.Kind() {
		case reflect.Struct:
			fillEmptyPointers(fld.Addr(), limit-1)
		case reflect.Slice:
			if elemIsStruct(fldType.Type.Elem()) {
				if fld.Len() == 0 {
					fld.Set(reflect.MakeSlice(fldType.Type, 1, 1))
				}
				fillEmptyPointers(fld.Index(0), limit-1)
			}
		case reflect.Map:
			if fldType.Type.Key().Kind() == reflect.Pointer {
				continue
			} else if elemIsStruct(fldType.Type.Elem()) {
				mapKey := reflect.New(fldType.Type.Key()).Elem()
				if fld.Len() == 0 {
					mapInstance := reflect.MakeMap(fldType.Type)
					mapElem := reflect.New(fldType.Type.Elem())
					fillEmptyPointers(mapElem, limit-1)
					mapInstance.SetMapIndex(mapKey, mapElem.Elem())
					fld.Set(mapInstance)
				} else {
					fillEmptyPointers(fld.MapIndex(mapKey), limit-1)
				}
			}
		case reflect.Pointer:
			fld.Set(reflect.New(fldType.Type.Elem()))
			fillEmptyPointers(fld, limit-1)
		default:
			// Nothing to expand
		}
	}
}

func verify(gvk schema.GroupVersionKind, example runtime.Object, sch *spec.Schema, definitions map[string]*spec.Schema, scheme *runtime.Scheme) error {
	example = example.DeepCopyObject()

	for limit := 1; limit < 10; limit++ {
		if err := compareDefaultingStrategies(gvk, example, sch, definitions, scheme); err != nil {
			return err
		}

		scheme.Default(example)
		fillEmptyPointers(reflect.ValueOf(example), limit)
	}

	return nil
}

func verifyWithFuzzer(gvk schema.GroupVersionKind, sch *spec.Schema, definitions map[string]*spec.Schema, scheme *runtime.Scheme, fuzzer *fuzz.Fuzzer) error {
	// Create an empty instance of the internal version of this type
	internalExample, err := scheme.New(gvk.GroupKind().WithVersion(runtime.APIVersionInternal))
	if err != nil {
		return err
	}

	for i := 0; i < 100000; i++ {
		// Fuzzers are for internal types
		internal := internalExample.DeepCopyObject()
		fuzzer.Fuzz(internal)

		// Convert to versioned type
		converted, err := scheme.ConvertToVersion(internal, gvk.GroupVersion())
		if err != nil {
			return err
		}

		if err := compareDefaultingStrategies(gvk, converted, sch, definitions, scheme); err != nil {
			return err
		}
	}

	return nil
}

func compareDefaultingStrategies(gvk schema.GroupVersionKind, example runtime.Object, sch *spec.Schema, definitions map[string]*spec.Schema, scheme *runtime.Scheme) error {
	nativeDefaulted := example.DeepCopyObject()
	schemaDefaulted := example.DeepCopyObject()

	nativeUnstructured, err := runtime.DefaultUnstructuredConverter.ToUnstructured(nativeDefaulted)
	if err != nil {
		return err
	}

	schemaUnstructured, err := runtime.DefaultUnstructuredConverter.ToUnstructured(schemaDefaulted)
	if err != nil {
		return err
	}

	if err := Default(schemaUnstructured, sch, definitions); err != nil {
		return err
	}

	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(schemaUnstructured, schemaDefaulted); err != nil {
		return err
	}

	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(nativeUnstructured, nativeDefaulted); err != nil {
		return err
	}

	scheme.Default(nativeDefaulted)

	if gvkExceptions, ok := exceptions[gvk]; ok {
		// .Elem() to remove the runtime.Object interface from reflect consideration
		currentNative := reflect.ValueOf(&nativeDefaulted).Elem().Elem()
		currentSchema := reflect.ValueOf(&schemaDefaulted).Elem().Elem()

		for _, path := range gvkExceptions {
			// Set the "schemaDefaulted" to nativeDefaulted for the path
			// to ignore in the diff
			// Or set both to empty of their type
			if err := makeFieldPathEqual(currentNative, currentSchema, path...); err != nil {
				return fmt.Errorf("error whitelisting type %v for path %v: %w", gvk, strings.Join(path, "."), err)
			}
		}
	}

	diff := cmp.Diff(nativeDefaulted, schemaDefaulted)
	if len(diff) != 0 {
		return fmt.Errorf("```diff\n%s\n```", diff)
	}

	return nil
}

// Default does defaulting of x depending on default values in s.
// Default values from s are deep-copied.
//
// PruneNonNullableNullsWithoutDefaults has left the non-nullable nulls
// that have a default here.
//
// Code copied from structural schema defaulting, adapted for use with OpenAPI
// (not all native schemas are structural. e.g. CRDs are recursive)
//
// Assumes the provided schema has the following properties:
//  1. `properties` are only be defined on the root schema of any definition. No inline objects are allowed.
//  2. Refs are either defined directly on the schema, or as the lone member of the schema's `allOf`
//  3. Defaults are either defined on an inline property/arrayelement schema, or
//     directly on a root object schema. The inline default takes precedence.
//     All other defaults are ignored.
func Default(x interface{}, s *spec.Schema, definitions map[string]*spec.Schema) error {
	if s == nil {
		return nil
	}

	var errs []error
	addErr := func(e error) {
		errs = append(errs, e)
	}

	// If s is an intermediate schema (like object property or array items)
	// then resolve the schema to the referred schema if there is one.
	s, err := resolveProperty(s, definitions)
	if err != nil {
		return nil
	}

	switch x := x.(type) {
	case map[string]interface{}:
		for k, prop := range s.Properties {
			def, err := defaultValue(&prop, definitions)
			if err != nil {
				addErr(err)
				continue
			} else if def == nil {
				continue
			}

			if _, found := x[k]; !found || isNonNullableNull(x[k], &prop, definitions) {
				x[k] = runtime.DeepCopyJSONValue(def)
			}
		}
		for k := range x {
			if prop, found := s.Properties[k]; found {
				if err := Default(x[k], &prop, definitions); err != nil {
					addErr(err)
					continue
				}
			} else if s.AdditionalProperties != nil {
				if isNonNullableNull(x[k], s.AdditionalProperties.Schema, definitions) {
					if def, err := defaultValue(s.AdditionalProperties.Schema, definitions); err != nil {
						addErr(err)
						continue
					} else if def != nil {
						x[k] = runtime.DeepCopyJSONValue(def)
					}
				}

				if err := Default(x[k], s.AdditionalProperties.Schema, definitions); err != nil {
					addErr(err)
					continue
				}
			}
		}
	case []interface{}:
		if s.Items == nil || s.Items.Schema == nil {
			break
		}

		def, err := defaultValue(s.Items.Schema, definitions)
		if err != nil {
			addErr(err)
			break
		}

		for i := range x {
			if isNonNullableNull(x[i], s.Items.Schema, definitions) {
				if def != nil {
					x[i] = runtime.DeepCopyJSONValue(def)
				}
			}
			Default(x[i], s.Items.Schema, definitions)
		}
	default:
		// scalars, do nothing
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}

	return nil
}

func makeFieldPathEqual(source, destination reflect.Value, path ...string) error {
	if source.Type() != destination.Type() {
		return fmt.Errorf("source and destination must be the same type")
	} else if len(path) == 0 {
		destination.Set(source)
		return nil
	} else if source.Kind() == reflect.Slice {
		// Run for each member of the slice
		destination.SetCap(source.Cap())
		destination.SetLen(source.Len())

		var errs []error

		for i := 0; i < source.Len(); i++ {
			makeFieldPathEqual(source.Index(i), destination.Index(i), path...)
		}

		if len(errs) > 0 {
			return errors.Join(errs...)
		}
		return nil
	}

	for destination.Kind() == reflect.Pointer {
		if source.IsNil() {
			// Nothing to do. Source has nothing on this path.
			return nil
		} else if destination.IsNil() {
			destination.Set(reflect.New(destination.Type().Elem()))
		}
		source = source.Elem()
		destination = destination.Elem()
	}

	if k := destination.Kind(); k != reflect.Struct {
		return fmt.Errorf("expected struct. found %v", k)
	}

	sourceField := source.FieldByName(path[0])
	destField := destination.FieldByName(path[0])

	if (sourceField == reflect.Value{}) {
		return fmt.Errorf("field %v not found on %v", path[0], source.Type().String())
	}

	return makeFieldPathEqual(sourceField, destField, path[1:]...)
}

// isNonNullalbeNull returns true if the item is nil AND it's nullable
func isNonNullableNull(x interface{}, s *spec.Schema, defs map[string]*spec.Schema) bool {
	// Not possible to tell with spec.Schema if nullable is unset or not
	// (which would mean it is affirmatively overridden)
	resolved, _ := resolveProperty(s, defs)
	return x == nil && s != nil && s.Nullable == false && (resolved == nil || resolved.Nullable == false)
}

// returns the default value of a schema, or follows references until one is found
func defaultValue(sch *spec.Schema, defs map[string]*spec.Schema) (interface{}, error) {
	if sch == nil {
		return nil, nil
	}

	if sch.Default != nil {
		return sch.Default, nil
	}
	resolved, err := resolveProperty(sch, defs)
	if err != nil {
		return nil, err
	} else if resolved == sch {
		return nil, nil
	}
	return defaultValue(resolved, defs)
}

// checks all allowed locations for a ref in a schema
func refOfSchema(sch *spec.Schema) string {
	if res := sch.Ref.String(); len(res) > 0 {
		return res
	}

	// For OpenAPI V3 some refs were stuffed inside allOf.
	// These refs are guaranteed to be the only element inside an allOf attached
	// to property due to construction of OpenAPI schema on server
	if len(sch.AllOf) == 1 && len(sch.AllOf[0].Ref.String()) > 0 {
		return sch.AllOf[0].Ref.String()
	}

	return ""
}

// If provided schema contains a reference, returns the schema for the reference
// Otherwise returns input schema
func resolveProperty(sch *spec.Schema, defs map[string]*spec.Schema) (*spec.Schema, error) {
	if sch == nil {
		return nil, nil
	}
	ref := refOfSchema(sch)
	if len(ref) == 0 {
		return sch, nil
	}

	lastPart := filepath.Base(ref)
	if existing, exists := defs[lastPart]; exists {
		return existing, nil
	}

	// unresolved reference
	// error?
	return nil, fmt.Errorf("unresolved reference in schema: %s", lastPart)
}
