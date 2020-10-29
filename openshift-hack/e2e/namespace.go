package e2e

import (
	"context"
	"fmt"
	"runtime/debug"
	"strings"

	"github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kclientset "k8s.io/client-go/kubernetes"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"

	projectv1 "github.com/openshift/api/project/v1"
)

// CreateTestingNS ensures that kubernetes e2e tests have their service accounts in the privileged and anyuid SCCs
func CreateTestingNS(ctx context.Context, baseName string, c kclientset.Interface, labels map[string]string, isKubeNamespace bool) (*corev1.Namespace, error) {
	if !strings.HasPrefix(baseName, "e2e-") {
		baseName = "e2e-" + baseName
	}

	if labels == nil {
		labels = map[string]string{}
	}
	// turn off the OpenShift label syncer so that it does not attempt to sync
	// the PodSecurity admission labels
	labels["security.openshift.io/scc.podSecurityLabelSync"] = "false"

	if isKubeNamespace {
		labels["security.openshift.io/disable-securitycontextconstraints"] = "true"
	}

	ns, err := framework.CreateTestingNS(ctx, baseName, c, labels)
	if err != nil {
		return ns, err
	}

	if !isKubeNamespace {
		return ns, err
	}

	// Add anyuid and privileged permissions for upstream tests
	clientConfig, err := framework.LoadConfig()
	if err != nil {
		return ns, err
	}

	rbacClient, err := rbacv1client.NewForConfig(clientConfig)
	if err != nil {
		return ns, err
	}
	framework.Logf("About to run a Kube e2e test, ensuring namespace/%s is privileged", ns.Name)
	// add the "privileged" scc to ensure pods that explicitly
	// request extra capabilities are not rejected
	addRoleToE2EServiceAccounts(ctx, rbacClient, []corev1.Namespace{*ns}, "system:openshift:scc:privileged")
	// add the "anyuid" scc to ensure pods that don't specify a
	// uid don't get forced into a range (mimics upstream
	// behavior)
	addRoleToE2EServiceAccounts(ctx, rbacClient, []corev1.Namespace{*ns}, "system:openshift:scc:anyuid")
	// add the "hostmount-anyuid" scc to ensure pods using hostPath
	// can execute tests
	addRoleToE2EServiceAccounts(ctx, rbacClient, []corev1.Namespace{*ns}, "system:openshift:scc:hostmount-anyuid")

	// The intra-pod test requires that the service account have
	// permission to retrieve service endpoints.
	addRoleToE2EServiceAccounts(ctx, rbacClient, []corev1.Namespace{*ns}, "view")

	// in practice too many kube tests ignore scheduling constraints
	allowAllNodeScheduling(ctx, c, ns.Name)

	return ns, err
}

var longRetry = wait.Backoff{Steps: 100}

func fatalErr(msg interface{}) {
	// the path that leads to this being called isn't always clear...
	fmt.Fprintln(ginkgo.GinkgoWriter, string(debug.Stack()))
	framework.Failf("%v", msg)
}

func addRoleToE2EServiceAccounts(ctx context.Context, rbacClient rbacv1client.RbacV1Interface, namespaces []corev1.Namespace, roleName string) {
	err := retry.RetryOnConflict(longRetry, func() error {
		for _, ns := range namespaces {
			if ns.Status.Phase != corev1.NamespaceTerminating {
				_, err := rbacClient.RoleBindings(ns.Name).Create(ctx, &rbacv1.RoleBinding{
					ObjectMeta: metav1.ObjectMeta{GenerateName: "default-" + roleName, Namespace: ns.Name},
					RoleRef: rbacv1.RoleRef{
						Kind: "ClusterRole",
						Name: roleName,
					},
					Subjects: []rbacv1.Subject{
						{Name: "default", Namespace: ns.Name, Kind: rbacv1.ServiceAccountKind},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					framework.Logf("Warning: Failed to add role to e2e service account: %v", err)
				}
			}
		}
		return nil
	})
	if err != nil {
		fatalErr(err)
	}
}

// allowAllNodeScheduling sets the annotation on namespace that allows all nodes to be scheduled onto.
func allowAllNodeScheduling(ctx context.Context, c kclientset.Interface, namespace string) {
	err := retry.RetryOnConflict(longRetry, func() error {
		ns, err := c.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if ns.Annotations == nil {
			ns.Annotations = make(map[string]string)
		}
		ns.Annotations[projectv1.ProjectNodeSelector] = ""
		_, err = c.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		fatalErr(err)
	}
}
