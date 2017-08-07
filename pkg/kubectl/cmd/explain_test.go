package cmd

import (
	"bytes"
	"testing"

	clientcmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func TestValidResources(t *testing.T) {
	out := &bytes.Buffer{}
	errOut := &bytes.Buffer{}
	f := clientcmdutil.NewFactory(nil)
	cmd := NewCmdExplain(f, out, errOut)

	resources := getValidResources()

	for _, resource := range resources {
		args := []string{resource}
		err := RunExplain(f, out, errOut, cmd, args)
		if err != nil {
			t.Errorf("kubectl explain fails for resource: %s", resource)
		}
	}
}

// TODO: add this information to kubectl and generate validResources programmatically
func getValidResources() []string {

	return []string{
		"all",
		"certificatesigningrequests",
		"clusterrolebindings",
		"clusterroles",
		"clusters",
		"componentstatuses",
		"configmaps",
		"controllerrevisions",
		"cronjobs",
		"customresourcedefinition",
		"daemonsets",
		"deployments",
		"endpoints",
		"events",
		"horizontalpodautoscalers",
		"ingresses",
		"jobs",
		"limitranges",
		"namespaces",
		"networkpolicies",
		"nodes",
		"persistentvolumeclaims",
		"persistentvolumes",
		"poddisruptionbudgets",
		"podpreset",
		"pods",
		"podsecuritypolicies",
		"podtemplates",
		"replicasets",
		"replicationcontrollers",
		"resourcequotas",
		"rolebindings",
		"roles",
		"secrets",
		"serviceaccounts",
		"services",
		"statefulsets",
		"storageclasses",
	}
}
