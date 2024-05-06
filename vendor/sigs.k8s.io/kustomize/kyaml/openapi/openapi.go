// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package openapi

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
	"google.golang.org/protobuf/proto"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/openapi/kubernetesapi"
	"sigs.k8s.io/kustomize/kyaml/openapi/kustomizationapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
	k8syaml "sigs.k8s.io/yaml"
)

var (
	// schemaLock is the lock for schema related globals.
	//
	// NOTE: This lock helps with preventing panics that might occur due to the data
	// race that concurrent access on this variable might cause but it doesn't
	// fully fix the issue described in https://github.com/kubernetes-sigs/kustomize/issues/4824.
	// For instance concurrently running goroutines where each of them calls SetSchema()
	// and/or GetSchemaVersion might end up received nil errors (success) whereas the
	// seconds one would overwrite the global variable that has been written by the
	// first one.
	schemaLock sync.RWMutex //nolint:gochecknoglobals

	// kubernetesOpenAPIVersion specifies which builtin kubernetes schema to use.
	kubernetesOpenAPIVersion string //nolint:gochecknoglobals

	// globalSchema contains global state information about the openapi
	globalSchema openapiData //nolint:gochecknoglobals

	// customSchemaFile stores the custom OpenApi schema if it is provided
	customSchema []byte //nolint:gochecknoglobals
)

// openapiData contains the parsed openapi state.  this is in a struct rather than
// a list of vars so that it can be reset from tests.
type openapiData struct {
	// schema holds the OpenAPI schema data
	schema spec.Schema

	// schemaForResourceType is a map of Resource types to their schemas
	schemaByResourceType map[yaml.TypeMeta]*spec.Schema

	// namespaceabilityByResourceType stores whether a given Resource type
	// is namespaceable or not
	namespaceabilityByResourceType map[yaml.TypeMeta]bool

	// noUseBuiltInSchema stores whether we want to prevent using the built-n
	// Kubernetes schema as part of the global schema
	noUseBuiltInSchema bool

	// schemaInit stores whether or not we've parsed the schema already,
	// so that we only reparse the when necessary (to speed up performance)
	schemaInit bool
}

type format string

const (
	JsonOrYaml format = "jsonOrYaml"
	Proto      format = "proto"
)

// precomputedIsNamespaceScoped precomputes IsNamespaceScoped for known types. This avoids Schema creation,
// which is expensive
// The test output from TestIsNamespaceScopedPrecompute shows the expected map in go syntax,and can be copy and pasted
// from the failure if it changes.
var precomputedIsNamespaceScoped = map[yaml.TypeMeta]bool{
	{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingWebhookConfiguration"}:        false,
	{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingWebhookConfiguration"}:      false,
	{APIVersion: "admissionregistration.k8s.io/v1beta1", Kind: "MutatingWebhookConfiguration"}:   false,
	{APIVersion: "admissionregistration.k8s.io/v1beta1", Kind: "ValidatingWebhookConfiguration"}: false,
	{APIVersion: "apiextensions.k8s.io/v1", Kind: "CustomResourceDefinition"}:                    false,
	{APIVersion: "apiextensions.k8s.io/v1beta1", Kind: "CustomResourceDefinition"}:               false,
	{APIVersion: "apiregistration.k8s.io/v1", Kind: "APIService"}:                                false,
	{APIVersion: "apiregistration.k8s.io/v1beta1", Kind: "APIService"}:                           false,
	{APIVersion: "apps/v1", Kind: "ControllerRevision"}:                                          true,
	{APIVersion: "apps/v1", Kind: "DaemonSet"}:                                                   true,
	{APIVersion: "apps/v1", Kind: "Deployment"}:                                                  true,
	{APIVersion: "apps/v1", Kind: "ReplicaSet"}:                                                  true,
	{APIVersion: "apps/v1", Kind: "StatefulSet"}:                                                 true,
	{APIVersion: "autoscaling/v1", Kind: "HorizontalPodAutoscaler"}:                              true,
	{APIVersion: "autoscaling/v1", Kind: "Scale"}:                                                true,
	{APIVersion: "autoscaling/v2beta1", Kind: "HorizontalPodAutoscaler"}:                         true,
	{APIVersion: "autoscaling/v2beta2", Kind: "HorizontalPodAutoscaler"}:                         true,
	{APIVersion: "batch/v1", Kind: "CronJob"}:                                                    true,
	{APIVersion: "batch/v1", Kind: "Job"}:                                                        true,
	{APIVersion: "batch/v1beta1", Kind: "CronJob"}:                                               true,
	{APIVersion: "certificates.k8s.io/v1", Kind: "CertificateSigningRequest"}:                    false,
	{APIVersion: "certificates.k8s.io/v1beta1", Kind: "CertificateSigningRequest"}:               false,
	{APIVersion: "coordination.k8s.io/v1", Kind: "Lease"}:                                        true,
	{APIVersion: "coordination.k8s.io/v1beta1", Kind: "Lease"}:                                   true,
	{APIVersion: "discovery.k8s.io/v1", Kind: "EndpointSlice"}:                                   true,
	{APIVersion: "discovery.k8s.io/v1beta1", Kind: "EndpointSlice"}:                              true,
	{APIVersion: "events.k8s.io/v1", Kind: "Event"}:                                              true,
	{APIVersion: "events.k8s.io/v1beta1", Kind: "Event"}:                                         true,
	{APIVersion: "extensions/v1beta1", Kind: "Ingress"}:                                          true,
	{APIVersion: "flowcontrol.apiserver.k8s.io/v1beta1", Kind: "FlowSchema"}:                     false,
	{APIVersion: "flowcontrol.apiserver.k8s.io/v1beta1", Kind: "PriorityLevelConfiguration"}:     false,
	{APIVersion: "networking.k8s.io/v1", Kind: "Ingress"}:                                        true,
	{APIVersion: "networking.k8s.io/v1", Kind: "IngressClass"}:                                   false,
	{APIVersion: "networking.k8s.io/v1", Kind: "NetworkPolicy"}:                                  true,
	{APIVersion: "networking.k8s.io/v1beta1", Kind: "Ingress"}:                                   true,
	{APIVersion: "networking.k8s.io/v1beta1", Kind: "IngressClass"}:                              false,
	{APIVersion: "node.k8s.io/v1", Kind: "RuntimeClass"}:                                         false,
	{APIVersion: "node.k8s.io/v1beta1", Kind: "RuntimeClass"}:                                    false,
	{APIVersion: "policy/v1", Kind: "PodDisruptionBudget"}:                                       true,
	{APIVersion: "policy/v1beta1", Kind: "PodDisruptionBudget"}:                                  true,
	{APIVersion: "policy/v1beta1", Kind: "PodSecurityPolicy"}:                                    false, // remove after openapi upgrades to v1.25.
	{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "ClusterRole"}:                            false,
	{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "ClusterRoleBinding"}:                     false,
	{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"}:                                   true,
	{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "RoleBinding"}:                            true,
	{APIVersion: "rbac.authorization.k8s.io/v1beta1", Kind: "ClusterRole"}:                       false,
	{APIVersion: "rbac.authorization.k8s.io/v1beta1", Kind: "ClusterRoleBinding"}:                false,
	{APIVersion: "rbac.authorization.k8s.io/v1beta1", Kind: "Role"}:                              true,
	{APIVersion: "rbac.authorization.k8s.io/v1beta1", Kind: "RoleBinding"}:                       true,
	{APIVersion: "scheduling.k8s.io/v1", Kind: "PriorityClass"}:                                  false,
	{APIVersion: "scheduling.k8s.io/v1beta1", Kind: "PriorityClass"}:                             false,
	{APIVersion: "storage.k8s.io/v1", Kind: "CSIDriver"}:                                         false,
	{APIVersion: "storage.k8s.io/v1", Kind: "CSINode"}:                                           false,
	{APIVersion: "storage.k8s.io/v1", Kind: "StorageClass"}:                                      false,
	{APIVersion: "storage.k8s.io/v1", Kind: "VolumeAttachment"}:                                  false,
	{APIVersion: "storage.k8s.io/v1beta1", Kind: "CSIDriver"}:                                    false,
	{APIVersion: "storage.k8s.io/v1beta1", Kind: "CSINode"}:                                      false,
	{APIVersion: "storage.k8s.io/v1beta1", Kind: "CSIStorageCapacity"}:                           true,
	{APIVersion: "storage.k8s.io/v1beta1", Kind: "StorageClass"}:                                 false,
	{APIVersion: "storage.k8s.io/v1beta1", Kind: "VolumeAttachment"}:                             false,
	{APIVersion: "v1", Kind: "ComponentStatus"}:                                                  false,
	{APIVersion: "v1", Kind: "ConfigMap"}:                                                        true,
	{APIVersion: "v1", Kind: "Endpoints"}:                                                        true,
	{APIVersion: "v1", Kind: "Event"}:                                                            true,
	{APIVersion: "v1", Kind: "LimitRange"}:                                                       true,
	{APIVersion: "v1", Kind: "Namespace"}:                                                        false,
	{APIVersion: "v1", Kind: "Node"}:                                                             false,
	{APIVersion: "v1", Kind: "NodeProxyOptions"}:                                                 false,
	{APIVersion: "v1", Kind: "PersistentVolume"}:                                                 false,
	{APIVersion: "v1", Kind: "PersistentVolumeClaim"}:                                            true,
	{APIVersion: "v1", Kind: "Pod"}:                                                              true,
	{APIVersion: "v1", Kind: "PodAttachOptions"}:                                                 true,
	{APIVersion: "v1", Kind: "PodExecOptions"}:                                                   true,
	{APIVersion: "v1", Kind: "PodPortForwardOptions"}:                                            true,
	{APIVersion: "v1", Kind: "PodProxyOptions"}:                                                  true,
	{APIVersion: "v1", Kind: "PodTemplate"}:                                                      true,
	{APIVersion: "v1", Kind: "ReplicationController"}:                                            true,
	{APIVersion: "v1", Kind: "ResourceQuota"}:                                                    true,
	{APIVersion: "v1", Kind: "Secret"}:                                                           true,
	{APIVersion: "v1", Kind: "Service"}:                                                          true,
	{APIVersion: "v1", Kind: "ServiceAccount"}:                                                   true,
	{APIVersion: "v1", Kind: "ServiceProxyOptions"}:                                              true,
}

// ResourceSchema wraps the OpenAPI Schema.
type ResourceSchema struct {
	// Schema is the OpenAPI schema for a Resource or field
	Schema *spec.Schema
}

// IsEmpty returns true if the ResourceSchema is empty
func (rs *ResourceSchema) IsMissingOrNull() bool {
	if rs == nil || rs.Schema == nil {
		return true
	}
	return reflect.DeepEqual(*rs.Schema, spec.Schema{})
}

// SchemaForResourceType returns the Schema for the given Resource
// TODO(pwittrock): create a version of this function that will return a schema
// which can be used for duck-typed Resources -- e.g. contains common fields such
// as metadata, replicas and spec.template.spec
func SchemaForResourceType(t yaml.TypeMeta) *ResourceSchema {
	initSchema()
	rs, found := globalSchema.schemaByResourceType[t]
	if !found {
		return nil
	}
	return &ResourceSchema{Schema: rs}
}

// SupplementaryOpenAPIFieldName is the conventional field name (JSON/YAML) containing
// supplementary OpenAPI definitions.
const SupplementaryOpenAPIFieldName = "openAPI"

const Definitions = "definitions"

// AddSchemaFromFile reads the file at path and parses the OpenAPI definitions
// from the field "openAPI", also returns a function to clean the added definitions
// The returned clean function is a no-op on error, or else it's a function
// that the caller should use to remove the added openAPI definitions from
// global schema
func SchemaFromFile(path string) (*spec.Schema, error) {
	object, err := parseOpenAPI(path)
	if err != nil {
		return nil, err
	}

	return schemaUsingField(object, SupplementaryOpenAPIFieldName)
}

// DefinitionRefs returns the list of openAPI definition references present in the
// input openAPIPath
func DefinitionRefs(openAPIPath string) ([]string, error) {
	object, err := parseOpenAPI(openAPIPath)
	if err != nil {
		return nil, err
	}
	return definitionRefsFromRNode(object)
}

// definitionRefsFromRNode returns the list of openAPI definitions keys from input
// yaml RNode
func definitionRefsFromRNode(object *yaml.RNode) ([]string, error) {
	definitions, err := object.Pipe(yaml.Lookup(SupplementaryOpenAPIFieldName, Definitions))
	if definitions == nil {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	return definitions.Fields()
}

// parseOpenAPI reads openAPIPath yaml and converts it to RNode
func parseOpenAPI(openAPIPath string) (*yaml.RNode, error) {
	b, err := os.ReadFile(openAPIPath)
	if err != nil {
		return nil, err
	}

	object, err := yaml.Parse(string(b))
	if err != nil {
		return nil, errors.Errorf("invalid file %q: %v", openAPIPath, err)
	}
	return object, nil
}

// addSchemaUsingField parses the OpenAPI definitions from the specified field.
// If field is the empty string, use the whole document as OpenAPI.
func schemaUsingField(object *yaml.RNode, field string) (*spec.Schema, error) {
	if field != "" {
		// get the field containing the openAPI
		m := object.Field(field)
		if m.IsNilOrEmpty() {
			// doesn't contain openAPI definitions
			return nil, nil
		}
		object = m.Value
	}

	oAPI, err := object.String()
	if err != nil {
		return nil, err
	}

	// convert the yaml openAPI to a JSON string by unmarshalling it to an
	// interface{} and the marshalling it to a string
	var o interface{}
	err = yaml.Unmarshal([]byte(oAPI), &o)
	if err != nil {
		return nil, err
	}
	j, err := json.Marshal(o)
	if err != nil {
		return nil, err
	}

	var sc spec.Schema
	err = sc.UnmarshalJSON(j)
	if err != nil {
		return nil, err
	}

	return &sc, nil
}

// AddSchema parses s, and adds definitions from s to the global schema.
func AddSchema(s []byte) error {
	return parse(s, JsonOrYaml)
}

// ResetOpenAPI resets the openapi data to empty
func ResetOpenAPI() {
	schemaLock.Lock()
	defer schemaLock.Unlock()

	globalSchema = openapiData{}
	customSchema = nil
	kubernetesOpenAPIVersion = ""
}

// AddDefinitions adds the definitions to the global schema.
func AddDefinitions(definitions spec.Definitions) {
	// initialize values if they have not yet been set
	if globalSchema.schemaByResourceType == nil {
		globalSchema.schemaByResourceType = map[yaml.TypeMeta]*spec.Schema{}
	}
	if globalSchema.schema.Definitions == nil {
		globalSchema.schema.Definitions = spec.Definitions{}
	}

	// index the schema definitions so we can lookup them up for Resources
	for k := range definitions {
		// index by GVK, if no GVK is found then it is the schema for a subfield
		// of a Resource
		d := definitions[k]

		// copy definitions to the schema
		globalSchema.schema.Definitions[k] = d
		gvk, found := d.VendorExtensible.Extensions[kubernetesGVKExtensionKey]
		if !found {
			continue
		}
		// cast the extension to a []map[string]string
		exts, ok := gvk.([]interface{})
		if !ok {
			continue
		}

		for i := range exts {
			typeMeta, ok := toTypeMeta(exts[i])
			if !ok {
				continue
			}
			globalSchema.schemaByResourceType[typeMeta] = &d
		}
	}
}

func toTypeMeta(ext interface{}) (yaml.TypeMeta, bool) {
	m, ok := ext.(map[string]interface{})
	if !ok {
		return yaml.TypeMeta{}, false
	}

	apiVersion := m[versionKey].(string)
	if g, ok := m[groupKey].(string); ok && g != "" {
		apiVersion = g + "/" + apiVersion
	}
	return yaml.TypeMeta{Kind: m[kindKey].(string), APIVersion: apiVersion}, true
}

// Resolve resolves the reference against the global schema
func Resolve(ref *spec.Ref, schema *spec.Schema) (*spec.Schema, error) {
	return resolve(schema, ref)
}

// Schema returns the global schema
func Schema() *spec.Schema {
	return rootSchema()
}

// GetSchema parses s into a ResourceSchema, resolving References within the
// global schema.
func GetSchema(s string, schema *spec.Schema) (*ResourceSchema, error) {
	var sc spec.Schema
	if err := sc.UnmarshalJSON([]byte(s)); err != nil {
		return nil, errors.Wrap(err)
	}
	if sc.Ref.String() != "" {
		r, err := Resolve(&sc.Ref, schema)
		if err != nil {
			return nil, errors.Wrap(err)
		}
		sc = *r
	}

	return &ResourceSchema{Schema: &sc}, nil
}

// IsNamespaceScoped determines whether a resource is namespace or
// cluster-scoped by looking at the information in the openapi schema.
// The second return value tells whether the provided type could be found
// in the openapi schema. If the value is false here, the scope of the
// resource is not known. If the type is found, the first return value will
// be true if the resource is namespace-scoped, and false if the type is
// cluster-scoped.
func IsNamespaceScoped(typeMeta yaml.TypeMeta) (bool, bool) {
	if res, f := precomputedIsNamespaceScoped[typeMeta]; f {
		return res, true
	}
	return isNamespaceScopedFromSchema(typeMeta)
}

func isNamespaceScopedFromSchema(typeMeta yaml.TypeMeta) (bool, bool) {
	initSchema()
	isNamespaceScoped, found := globalSchema.namespaceabilityByResourceType[typeMeta]
	return isNamespaceScoped, found
}

// IsCertainlyClusterScoped returns true for Node, Namespace, etc. and
// false for Pod, Deployment, etc. and kinds that aren't recognized in the
// openapi data. See:
// https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces
func IsCertainlyClusterScoped(typeMeta yaml.TypeMeta) bool {
	nsScoped, found := IsNamespaceScoped(typeMeta)
	return found && !nsScoped
}

// SuppressBuiltInSchemaUse can be called to prevent using the built-in Kubernetes
// schema as part of the global schema.
// Must be called before the schema is used.
func SuppressBuiltInSchemaUse() {
	globalSchema.noUseBuiltInSchema = true
}

// Elements returns the Schema for the elements of an array.
func (rs *ResourceSchema) Elements() *ResourceSchema {
	// load the schema from swagger files
	initSchema()

	if len(rs.Schema.Type) != 1 || rs.Schema.Type[0] != "array" {
		// either not an array, or array has multiple types
		return nil
	}
	if rs == nil || rs.Schema == nil || rs.Schema.Items == nil {
		// no-scheme for the items
		return nil
	}
	s := *rs.Schema.Items.Schema
	for s.Ref.String() != "" {
		sc, e := Resolve(&s.Ref, Schema())
		if e != nil {
			return nil
		}
		s = *sc
	}
	return &ResourceSchema{Schema: &s}
}

const Elements = "[]"

// Lookup calls either Field or Elements for each item in the path.
// If the path item is "[]", then Elements is called, otherwise
// Field is called.
// If any Field or Elements call returns nil, then Lookup returns
// nil immediately.
func (rs *ResourceSchema) Lookup(path ...string) *ResourceSchema {
	s := rs
	for _, p := range path {
		if s == nil {
			break
		}
		if p == Elements {
			s = s.Elements()
			continue
		}
		s = s.Field(p)
	}
	return s
}

// Field returns the Schema for a field.
func (rs *ResourceSchema) Field(field string) *ResourceSchema {
	// load the schema from swagger files
	initSchema()

	// locate the Schema
	s, found := rs.Schema.Properties[field]
	switch {
	case found:
		// no-op, continue with s as the schema
	case rs.Schema.AdditionalProperties != nil && rs.Schema.AdditionalProperties.Schema != nil:
		// map field type -- use Schema of the value
		// (the key doesn't matter, they all have the same value type)
		s = *rs.Schema.AdditionalProperties.Schema
	default:
		// no Schema found from either swagger files or line comments
		return nil
	}

	// resolve the reference to the Schema if the Schema has one
	for s.Ref.String() != "" {
		sc, e := Resolve(&s.Ref, Schema())
		if e != nil {
			return nil
		}
		s = *sc
	}

	// return the merged Schema
	return &ResourceSchema{Schema: &s}
}

// PatchStrategyAndKeyList returns the patch strategy and complete merge key list
func (rs *ResourceSchema) PatchStrategyAndKeyList() (string, []string) {
	ps, found := rs.Schema.Extensions[kubernetesPatchStrategyExtensionKey]
	if !found {
		// empty patch strategy
		return "", []string{}
	}
	mkList, found := rs.Schema.Extensions[kubernetesMergeKeyMapList]
	if found {
		// mkList is []interface, convert to []string
		mkListStr := make([]string, len(mkList.([]interface{})))
		for i, v := range mkList.([]interface{}) {
			mkListStr[i] = v.(string)
		}
		return ps.(string), mkListStr
	}
	mk, found := rs.Schema.Extensions[kubernetesMergeKeyExtensionKey]
	if !found {
		// no mergeKey -- may be a primitive associative list (e.g. finalizers)
		return ps.(string), []string{}
	}
	return ps.(string), []string{mk.(string)}
}

// PatchStrategyAndKey returns the patch strategy and merge key extensions
func (rs *ResourceSchema) PatchStrategyAndKey() (string, string) {
	ps, found := rs.Schema.Extensions[kubernetesPatchStrategyExtensionKey]
	if !found {
		// empty patch strategy
		return "", ""
	}

	mk, found := rs.Schema.Extensions[kubernetesMergeKeyExtensionKey]
	if !found {
		// no mergeKey -- may be a primitive associative list (e.g. finalizers)
		mk = ""
	}
	return ps.(string), mk.(string)
}

const (
	// kubernetesOpenAPIDefaultVersion is the latest version number of the statically compiled in
	// OpenAPI schema for kubernetes built-in types
	kubernetesOpenAPIDefaultVersion = kubernetesapi.DefaultOpenAPI

	// kustomizationAPIAssetName is the name of the asset containing the statically compiled in
	// OpenAPI definitions for Kustomization built-in types
	kustomizationAPIAssetName = "kustomizationapi/swagger.json"

	// kubernetesGVKExtensionKey is the key to lookup the kubernetes group version kind extension
	// -- the extension is an array of objects containing a gvk
	kubernetesGVKExtensionKey = "x-kubernetes-group-version-kind"

	// kubernetesMergeKeyExtensionKey is the key to lookup the kubernetes merge key extension
	// -- the extension is a string
	kubernetesMergeKeyExtensionKey = "x-kubernetes-patch-merge-key"

	// kubernetesPatchStrategyExtensionKey is the key to lookup the kubernetes patch strategy
	// extension -- the extension is a string
	kubernetesPatchStrategyExtensionKey = "x-kubernetes-patch-strategy"

	// kubernetesMergeKeyMapList is the list of merge keys when there needs to be multiple
	// -- the extension is an array of strings
	kubernetesMergeKeyMapList = "x-kubernetes-list-map-keys"

	// groupKey is the key to lookup the group from the GVK extension
	groupKey = "group"
	// versionKey is the key to lookup the version from the GVK extension
	versionKey = "version"
	// kindKey is the to lookup the kind from the GVK extension
	kindKey = "kind"
)

// SetSchema sets the kubernetes OpenAPI schema version to use
func SetSchema(openAPIField map[string]string, schema []byte, reset bool) error {
	schemaLock.Lock()
	defer schemaLock.Unlock()

	// this should only be set once
	schemaIsSet := (kubernetesOpenAPIVersion != "") || customSchema != nil
	if schemaIsSet && !reset {
		return nil
	}

	version, versionProvided := openAPIField["version"]

	// use custom schema
	if schema != nil {
		if versionProvided {
			return fmt.Errorf("builtin version and custom schema provided, cannot use both")
		}
		customSchema = schema
		kubernetesOpenAPIVersion = "custom"
		// if the schema is changed, initSchema should parse the new schema
		globalSchema.schemaInit = false
		return nil
	}

	// use builtin version
	kubernetesOpenAPIVersion = version
	if kubernetesOpenAPIVersion == "" {
		return nil
	}
	if _, ok := kubernetesapi.OpenAPIMustAsset[kubernetesOpenAPIVersion]; !ok {
		return fmt.Errorf("the specified OpenAPI version is not built in")
	}

	customSchema = nil
	// if the schema is changed, initSchema should parse the new schema
	globalSchema.schemaInit = false
	return nil
}

// GetSchemaVersion returns what kubernetes OpenAPI version is being used
func GetSchemaVersion() string {
	schemaLock.RLock()
	defer schemaLock.RUnlock()

	switch {
	case kubernetesOpenAPIVersion == "" && customSchema == nil:
		return kubernetesOpenAPIDefaultVersion
	case customSchema != nil:
		return "using custom schema from file provided"
	default:
		return kubernetesOpenAPIVersion
	}
}

// initSchema parses the json schema
func initSchema() {
	schemaLock.Lock()
	defer schemaLock.Unlock()

	if globalSchema.schemaInit {
		return
	}
	globalSchema.schemaInit = true

	// TODO(natasha41575): Accept proto-formatted schema files
	if customSchema != nil {
		err := parse(customSchema, JsonOrYaml)
		if err != nil {
			panic(fmt.Errorf("invalid schema file: %w", err))
		}
	} else {
		if kubernetesOpenAPIVersion == "" {
			parseBuiltinSchema(kubernetesOpenAPIDefaultVersion)
		} else {
			parseBuiltinSchema(kubernetesOpenAPIVersion)
		}
	}

	if err := parse(kustomizationapi.MustAsset(kustomizationAPIAssetName), JsonOrYaml); err != nil {
		// this should never happen
		panic(err)
	}
}

// parseBuiltinSchema calls parse to parse the json or proto schemas
func parseBuiltinSchema(version string) {
	if globalSchema.noUseBuiltInSchema {
		// don't parse the built in schema
		return
	}
	// parse the swagger, this should never fail
	assetName := filepath.Join(
		"kubernetesapi",
		strings.ReplaceAll(version, ".", "_"),
		"swagger.pb")

	if err := parse(kubernetesapi.OpenAPIMustAsset[version](assetName), Proto); err != nil {
		// this should never happen
		panic(err)
	}
}

// parse parses and indexes a single json or proto schema
func parse(b []byte, format format) error {
	var swagger spec.Swagger
	switch {
	case format == Proto:
		doc := &openapi_v2.Document{}
		// We parse protobuf and get an openapi_v2.Document here.
		if err := proto.Unmarshal(b, doc); err != nil {
			return fmt.Errorf("openapi proto unmarshalling failed: %w", err)
		}
		// convert the openapi_v2.Document back to Swagger
		_, err := swagger.FromGnostic(doc)
		if err != nil {
			return errors.Wrap(err)
		}

	case format == JsonOrYaml:
		if len(b) > 0 && b[0] != byte('{') {
			var err error
			b, err = k8syaml.YAMLToJSON(b)
			if err != nil {
				return errors.Wrap(err)
			}
		}
		if err := swagger.UnmarshalJSON(b); err != nil {
			return errors.Wrap(err)
		}
	}

	AddDefinitions(swagger.Definitions)
	findNamespaceability(swagger.Paths)
	return nil
}

// findNamespaceability looks at the api paths for the resource to determine
// if it is cluster-scoped or namespace-scoped. The gvk of the resource
// for each path is found by looking at the x-kubernetes-group-version-kind
// extension. If a path exists for the resource that contains a namespace path
// parameter, the resource is namespace-scoped.
func findNamespaceability(paths *spec.Paths) {
	if globalSchema.namespaceabilityByResourceType == nil {
		globalSchema.namespaceabilityByResourceType = make(map[yaml.TypeMeta]bool)
	}

	if paths == nil {
		return
	}

	for path, pathInfo := range paths.Paths {
		if pathInfo.Get == nil {
			continue
		}
		gvk, found := pathInfo.Get.VendorExtensible.Extensions[kubernetesGVKExtensionKey]
		if !found {
			continue
		}
		typeMeta, found := toTypeMeta(gvk)
		if !found {
			continue
		}

		if strings.Contains(path, "namespaces/{namespace}") {
			// if we find a namespace path parameter, we just update the map
			// directly
			globalSchema.namespaceabilityByResourceType[typeMeta] = true
		} else if _, found := globalSchema.namespaceabilityByResourceType[typeMeta]; !found {
			// if the resource doesn't have the namespace path parameter, we
			// only add it to the map if it doesn't already exist.
			globalSchema.namespaceabilityByResourceType[typeMeta] = false
		}
	}
}

func resolve(root interface{}, ref *spec.Ref) (*spec.Schema, error) {
	if s, ok := root.(*spec.Schema); ok && s == nil {
		return nil, nil
	}
	res, _, err := ref.GetPointer().Get(root)
	if err != nil {
		return nil, errors.Wrap(err)
	}
	switch sch := res.(type) {
	case spec.Schema:
		return &sch, nil
	case *spec.Schema:
		return sch, nil
	case map[string]interface{}:
		b, err := json.Marshal(sch)
		if err != nil {
			return nil, err
		}
		newSch := new(spec.Schema)
		if err = json.Unmarshal(b, newSch); err != nil {
			return nil, err
		}
		return newSch, nil
	default:
		return nil, errors.Wrap(fmt.Errorf("unknown type for the resolved reference"))
	}
}

func rootSchema() *spec.Schema {
	initSchema()
	return &globalSchema.schema
}
