package openshiftrestmapper

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// defaultRESTMappings contains enough RESTMappings to have enough of the kube-controller-manager succeed when running
// against a kube-apiserver that cannot reach aggregated APIs to do a full mapping.  This happens when the OwnerReferencesPermissionEnforcement
// admission plugin runs to confirm permissions.  Don't add things just because you don't want to fail.  These are here so that
// we can start enough back up to get the rest of the system working correctly.
var defaultRESTMappings = []meta.RESTMapping{
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ReplicationController"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "replicationcontrollers"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Secret"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "secrets"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ServiceAccount"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "ControllerRevision"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "controllerrevisions"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "DaemonSet"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "daemonsets"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "ReplicaSet"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "replicasets"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "StatefulSet"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "statefulsets"},
	},
	// This is created so that cluster-bootstrap can always map securitycontextconstraints since the CRD doesn't have
	// discovery. Discovery is delegated to the openshift-apiserver which doesn't not exist early in the bootstrapping
	// phase.  This leads to SCC related failures that we don't need to have.
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "security.openshift.io", Version: "v1", Kind: "SecurityContextConstraints"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "security.openshift.io", Version: "v1", Resource: "securitycontextconstraints"},
	},
	// This is created so that cluster-bootstrap can always map customresourcedefinitions, RBAC, machine resources so that CRDs and
	// permissions are always created quickly.  We observed discovery not including these on AWS OVN installations and
	// the lack of CRDs and permissions blocked additional aspects of cluster bootstrapping.
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "v1", Kind: "CustomResourceDefinition"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "v1", Resource: "customresourcedefinitions"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "ClusterRole"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "ClusterRoleBinding"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterrolebindings"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "roles"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "rolebindings"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "machine.openshift.io", Version: "v1beta1", Kind: "Machine"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "machine.openshift.io", Version: "v1beta1", Resource: "machines"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "machine.openshift.io", Version: "v1beta1", Kind: "MachineSet"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "machine.openshift.io", Version: "v1beta1", Resource: "machinesets"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "machineconfiguration.openshift.io", Version: "v1", Kind: "MachineConfig"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "machineconfiguration.openshift.io", Version: "v1", Resource: "machineconfigs"},
	},
	// This is here so cluster-bootstrap can always create the config instances that are used to drive our operators to avoid the
	// excessive bootstrap wait that prevents installer from completing on AWS OVN
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "DNS"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "dnses"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "Infrastructure"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "infrastructures"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "Network"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "networks"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "Ingress"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "ingresses"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "Proxy"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "proxies"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "Scheduler"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "schedulers"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "config.openshift.io", Version: "v1", Kind: "ClusterVersion"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "clusterversions"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "operator.openshift.io", Version: "v1", Kind: "CloudCredential"},
		Scope:            meta.RESTScopeRoot,
		Resource:         schema.GroupVersionResource{Group: "operator.openshift.io", Version: "v1", Resource: "cloudcredentials"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "monitoring.coreos.com", Version: "v1", Kind: "ServiceMonitor"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "monitoring.coreos.com", Version: "v1", Resource: "servicemonitors"},
	},
	{
		GroupVersionKind: schema.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
		Scope:            meta.RESTScopeNamespace,
		Resource:         schema.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
	},
}

func NewOpenShiftHardcodedRESTMapper(delegate meta.RESTMapper) meta.RESTMapper {
	ret := HardCodedFirstRESTMapper{
		Mapping:    map[schema.GroupVersionKind]meta.RESTMapping{},
		RESTMapper: delegate,
	}
	for i := range defaultRESTMappings {
		curr := defaultRESTMappings[i]
		ret.Mapping[curr.GroupVersionKind] = curr
	}
	return ret
}

// HardCodedFirstRESTMapper is a RESTMapper that will look for hardcoded mappings first, then delegate.
// This is done in service to `OwnerReferencesPermissionEnforcement` and for cluster-bootstrap.
type HardCodedFirstRESTMapper struct {
	Mapping map[schema.GroupVersionKind]meta.RESTMapping
	meta.RESTMapper
}

var _ meta.RESTMapper = HardCodedFirstRESTMapper{}

func (m HardCodedFirstRESTMapper) String() string {
	return fmt.Sprintf("HardCodedRESTMapper{\n\t%v\n%v\n}", m.Mapping, m.RESTMapper)
}

// RESTMapping is the only function called today.  The first hit openshiftrestmapper ought to make this work right.  OwnerReferencesPermissionEnforcement
// only ever calls with one version.
func (m HardCodedFirstRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	// not exactly one version, delegate
	if len(versions) != 1 {
		return m.RESTMapper.RESTMapping(gk, versions...)
	}
	gvk := gk.WithVersion(versions[0])

	single, ok := m.Mapping[gvk]
	// not handled, delegate
	if !ok {
		return m.RESTMapper.RESTMapping(gk, versions...)
	}

	return &single, nil
}

// RESTMapping is the only function called today.  The firsthit openshiftrestmapper ought to make this work right.  OwnerReferencesPermissionEnforcement
// only ever calls with one version.
func (m HardCodedFirstRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	// not exactly one version, delegate
	if len(versions) != 1 {
		return m.RESTMapper.RESTMappings(gk, versions...)
	}
	gvk := gk.WithVersion(versions[0])

	single, ok := m.Mapping[gvk]
	// not handled, delegate
	if !ok {
		return m.RESTMapper.RESTMappings(gk, versions...)
	}

	return []*meta.RESTMapping{&single}, nil
}
