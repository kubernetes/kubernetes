package securitycontextconstraints

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/admission"

	securityv1 "github.com/openshift/api/security/v1"
	sccutil "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util"
)

func TestDefaultingHappens(t *testing.T) {
	inputSCC := `{
	"allowHostDirVolumePlugin": true,
	"allowHostNetwork": true,
	"allowHostPID": true,
	"allowHostPorts": true,
	"apiVersion": "security.openshift.io/v1",
	"kind": "SecurityContextConstraints",
	"metadata": {
		"annotations": {
			"kubernetes.io/description": "node-exporter scc is used for the Prometheus node exporter"
		},
		"name": "node-exporter"
	},
	"readOnlyRootFilesystem": false,
	"runAsUser": {
		"type": "RunAsAny"
	},
	"seLinuxContext": {
		"type": "RunAsAny"
	},
	"users": []
}`

	inputUnstructured := &unstructured.Unstructured{}
	_, _, err := unstructured.UnstructuredJSONScheme.Decode([]byte(inputSCC), nil, inputUnstructured)
	if err != nil {
		t.Fatal(err)
	}

	attributes := admission.NewAttributesRecord(inputUnstructured, nil, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{Group: "security.openshift.io", Resource: "securitycontextconstraints"}, "", admission.Create, nil, false, nil)
	defaulter := NewDefaulter()
	if err := defaulter.(*defaultSCC).Admit(context.TODO(), attributes, nil); err != nil {
		t.Fatal(err)
	}

	buf := &bytes.Buffer{}
	if err := unstructured.UnstructuredJSONScheme.Encode(inputUnstructured, buf); err != nil {
		t.Fatal(err)
	}

	expectedSCC := `{
	"allowHostDirVolumePlugin": true,
	"allowHostIPC": false,
	"allowHostNetwork": true,
	"allowHostPID": true,
	"allowHostPorts": true,
	"allowPrivilegeEscalation": true,
	"allowPrivilegedContainer": false,
	"allowedCapabilities": null,
	"apiVersion": "security.openshift.io/v1",
	"defaultAddCapabilities": null,
	"fsGroup": {
		"type": "RunAsAny"
	},
	"groups": [],
	"kind": "SecurityContextConstraints",
	"metadata": {
		"annotations": {
			"kubernetes.io/description": "node-exporter scc is used for the Prometheus node exporter"
		},
		"name": "node-exporter",
		"creationTimestamp":null
	},
	"priority": null,
	"readOnlyRootFilesystem": false,
	"requiredDropCapabilities": null,
	"runAsUser": {
		"type": "RunAsAny"
	},
	"seLinuxContext": {
		"type": "RunAsAny"
	},
	"supplementalGroups": {
		"type": "RunAsAny"
	},
	"users": [],
	"volumes": [
		"*"
	]
}`
	expectedUnstructured := &unstructured.Unstructured{}
	if _, _, err := unstructured.UnstructuredJSONScheme.Decode([]byte(expectedSCC), nil, expectedUnstructured); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(expectedUnstructured.Object, inputUnstructured.Object) {
		t.Fatal(diff.ObjectDiff(expectedUnstructured.Object, inputUnstructured.Object))
	}
}

func TestDefaultSecurityContextConstraints(t *testing.T) {
	tests := map[string]struct {
		scc              *securityv1.SecurityContextConstraints
		expectedFSGroup  securityv1.FSGroupStrategyType
		expectedSupGroup securityv1.SupplementalGroupsStrategyType
	}{
		"shouldn't default": {
			scc: &securityv1.SecurityContextConstraints{
				FSGroup: securityv1.FSGroupStrategyOptions{
					Type: securityv1.FSGroupStrategyMustRunAs,
				},
				SupplementalGroups: securityv1.SupplementalGroupsStrategyOptions{
					Type: securityv1.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  securityv1.FSGroupStrategyMustRunAs,
			expectedSupGroup: securityv1.SupplementalGroupsStrategyMustRunAs,
		},
		"default fsgroup runAsAny": {
			scc: &securityv1.SecurityContextConstraints{
				RunAsUser: securityv1.RunAsUserStrategyOptions{
					Type: securityv1.RunAsUserStrategyRunAsAny,
				},
				SupplementalGroups: securityv1.SupplementalGroupsStrategyOptions{
					Type: securityv1.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  securityv1.FSGroupStrategyRunAsAny,
			expectedSupGroup: securityv1.SupplementalGroupsStrategyMustRunAs,
		},
		"default sup group runAsAny": {
			scc: &securityv1.SecurityContextConstraints{
				RunAsUser: securityv1.RunAsUserStrategyOptions{
					Type: securityv1.RunAsUserStrategyRunAsAny,
				},
				FSGroup: securityv1.FSGroupStrategyOptions{
					Type: securityv1.FSGroupStrategyMustRunAs,
				},
			},
			expectedFSGroup:  securityv1.FSGroupStrategyMustRunAs,
			expectedSupGroup: securityv1.SupplementalGroupsStrategyRunAsAny,
		},
		"default fsgroup runAsAny with mustRunAs UID strategy": {
			scc: &securityv1.SecurityContextConstraints{
				RunAsUser: securityv1.RunAsUserStrategyOptions{
					Type: securityv1.RunAsUserStrategyMustRunAsRange,
				},
				SupplementalGroups: securityv1.SupplementalGroupsStrategyOptions{
					Type: securityv1.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  securityv1.FSGroupStrategyRunAsAny,
			expectedSupGroup: securityv1.SupplementalGroupsStrategyMustRunAs,
		},
		"default sup group runAsAny with mustRunAs UID strategy": {
			scc: &securityv1.SecurityContextConstraints{
				RunAsUser: securityv1.RunAsUserStrategyOptions{
					Type: securityv1.RunAsUserStrategyMustRunAsRange,
				},
				FSGroup: securityv1.FSGroupStrategyOptions{
					Type: securityv1.FSGroupStrategyMustRunAs,
				},
			},
			expectedFSGroup:  securityv1.FSGroupStrategyMustRunAs,
			expectedSupGroup: securityv1.SupplementalGroupsStrategyRunAsAny,
		},
	}
	for k, v := range tests {
		SetDefaults_SCC(v.scc)
		if v.scc.FSGroup.Type != v.expectedFSGroup {
			t.Errorf("%s has invalid fsgroup.  Expected: %v got: %v", k, v.expectedFSGroup, v.scc.FSGroup.Type)
		}
		if v.scc.SupplementalGroups.Type != v.expectedSupGroup {
			t.Errorf("%s has invalid supplemental group.  Expected: %v got: %v", k, v.expectedSupGroup, v.scc.SupplementalGroups.Type)
		}
	}
}

func TestDefaultSCCVolumes(t *testing.T) {
	tests := map[string]struct {
		scc             *securityv1.SecurityContextConstraints
		expectedVolumes []securityv1.FSType
		expectedHostDir bool
	}{
		// this expects the volumes to default to all for an empty volume slice
		// but since the host dir setting is false it should be all - host dir
		"old client - default allow* fields, no volumes slice": {
			scc:             &securityv1.SecurityContextConstraints{},
			expectedVolumes: StringSetToFSType(sccutil.GetAllFSTypesExcept(string(securityv1.FSTypeHostPath))),
			expectedHostDir: false,
		},
		// this expects the volumes to default to all for an empty volume slice
		"old client - set allowHostDir true fields, no volumes slice": {
			scc: &securityv1.SecurityContextConstraints{
				AllowHostDirVolumePlugin: true,
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeAll},
			expectedHostDir: true,
		},
		"new client - allow* fields set with matching volume slice": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes:                  []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeHostPath},
				AllowHostDirVolumePlugin: true,
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeHostPath},
			expectedHostDir: true,
		},
		"new client - allow* fields set with mismatch host dir volume slice": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes:                  []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeHostPath},
				AllowHostDirVolumePlugin: false,
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeEmptyDir},
			expectedHostDir: false,
		},
		"new client - allow* fields set with mismatch FSTypeAll volume slice": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes:                  []securityv1.FSType{securityv1.FSTypeAll},
				AllowHostDirVolumePlugin: false,
			},
			expectedVolumes: StringSetToFSType(sccutil.GetAllFSTypesExcept(string(securityv1.FSTypeHostPath))),
			expectedHostDir: false,
		},
		"new client - allow* fields unset with volume slice": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes: []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeHostPath},
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeEmptyDir},
			expectedHostDir: false,
		},
		"new client - extra volume params retained": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes: []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeHostPath, securityv1.FSTypeGitRepo},
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeEmptyDir, securityv1.FSTypeGitRepo},
			expectedHostDir: false,
		},
		"new client - empty volume slice, host dir true": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes:                  []securityv1.FSType{},
				AllowHostDirVolumePlugin: true,
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeHostPath},
			expectedHostDir: true,
		},
		"new client - empty volume slice, host dir false": {
			scc: &securityv1.SecurityContextConstraints{
				Volumes:                  []securityv1.FSType{},
				AllowHostDirVolumePlugin: false,
			},
			expectedVolumes: []securityv1.FSType{securityv1.FSTypeNone},
			expectedHostDir: false,
		},
	}
	for k, v := range tests {
		SetDefaults_SCC(v.scc)

		if !reflect.DeepEqual(v.scc.Volumes, v.expectedVolumes) {
			t.Errorf("%s has invalid volumes.  Expected: %v got: %v", k, v.expectedVolumes, v.scc.Volumes)
		}

		if v.scc.AllowHostDirVolumePlugin != v.expectedHostDir {
			t.Errorf("%s has invalid host dir.  Expected: %v got: %v", k, v.expectedHostDir, v.scc.AllowHostDirVolumePlugin)
		}
	}
}
