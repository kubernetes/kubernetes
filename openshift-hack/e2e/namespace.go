package e2e

import (
	"context"
	"fmt"
	"runtime/debug"
	"strings"

	"github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kclientset "k8s.io/client-go/kubernetes"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"

	projectv1 "github.com/openshift/api/project/v1"
	securityv1client "github.com/openshift/client-go/security/clientset/versioned"
)

// CreateTestingNS ensures that kubernetes e2e tests have their service accounts in the privileged and anyuid SCCs
func CreateTestingNS(baseName string, c kclientset.Interface, labels map[string]string, isKubeNamespace bool) (*corev1.Namespace, error) {
	if !strings.HasPrefix(baseName, "e2e-") {
		baseName = "e2e-" + baseName
	}

	if isKubeNamespace {
		if labels == nil {
			labels = map[string]string{}
		}
		labels["security.openshift.io/disable-securitycontextconstraints"] = "true"
	}

	ns, err := framework.CreateTestingNS(context.Background(), baseName, c, labels)
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

	securityClient, err := securityv1client.NewForConfig(clientConfig)
	if err != nil {
		return ns, err
	}
	framework.Logf("About to run a Kube e2e test, ensuring namespace/%s is privileged", ns.Name)
	// add the "privileged" scc to ensure pods that explicitly
	// request extra capabilities are not rejected
	addE2EServiceAccountsToSCC(securityClient, []corev1.Namespace{*ns}, "privileged")
	// add the "anyuid" scc to ensure pods that don't specify a
	// uid don't get forced into a range (mimics upstream
	// behavior)
	addE2EServiceAccountsToSCC(securityClient, []corev1.Namespace{*ns}, "anyuid")
	// add the "hostmount-anyuid" scc to ensure pods using hostPath
	// can execute tests
	addE2EServiceAccountsToSCC(securityClient, []corev1.Namespace{*ns}, "hostmount-anyuid")

	// The intra-pod test requires that the service account have
	// permission to retrieve service endpoints.
	rbacClient, err := rbacv1client.NewForConfig(clientConfig)
	if err != nil {
		return ns, err
	}
	addRoleToE2EServiceAccounts(rbacClient, []corev1.Namespace{*ns}, "view")

	// in practice too many kube tests ignore scheduling constraints
	allowAllNodeScheduling(c, ns.Name)

	return ns, err
}

var longRetry = wait.Backoff{Steps: 100}

// TODO: ideally this should be rewritten to use dynamic client, not to rely on openshift types
func addE2EServiceAccountsToSCC(securityClient securityv1client.Interface, namespaces []corev1.Namespace, sccName string) {
	// Because updates can race, we need to set the backoff retries to be > than the number of possible
	// parallel jobs starting at once. Set very high to allow future high parallelism.
	err := retry.RetryOnConflict(longRetry, func() error {
		scc, err := securityClient.SecurityV1().SecurityContextConstraints().Get(context.Background(), sccName, metav1.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				return nil
			}
			return err
		}

		for _, ns := range namespaces {
			scc.Groups = append(scc.Groups, fmt.Sprintf("system:serviceaccounts:%s", ns.Name))
		}
		if _, err := securityClient.SecurityV1().SecurityContextConstraints().Update(context.Background(), scc, metav1.UpdateOptions{}); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		fatalErr(err)
	}
}

func fatalErr(msg interface{}) {
	// the path that leads to this being called isn't always clear...
	fmt.Fprintln(ginkgo.GinkgoWriter, string(debug.Stack()))
	framework.Failf("%v", msg)
}

func addRoleToE2EServiceAccounts(rbacClient rbacv1client.RbacV1Interface, namespaces []corev1.Namespace, roleName string) {
	err := retry.RetryOnConflict(longRetry, func() error {
		for _, ns := range namespaces {
			if ns.Status.Phase != corev1.NamespaceTerminating {
				_, err := rbacClient.RoleBindings(ns.Name).Create(context.Background(), &rbacv1.RoleBinding{
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
func allowAllNodeScheduling(c kclientset.Interface, namespace string) {
	err := retry.RetryOnConflict(longRetry, func() error {
		ns, err := c.CoreV1().Namespaces().Get(context.Background(), namespace, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if ns.Annotations == nil {
			ns.Annotations = make(map[string]string)
		}
		ns.Annotations[projectv1.ProjectNodeSelector] = ""
		_, err = c.CoreV1().Namespaces().Update(context.Background(), ns, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		fatalErr(err)
	}
}
