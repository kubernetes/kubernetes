/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"context"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"time"

	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	"github.com/onsi/ginkgo"

	certificatesv1 "k8s.io/api/certificates/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils"
)

var _ = SIGDescribe("Certificates API [Privileged:ClusterAdmin]", func() {
	f := framework.NewDefaultFramework("certificates")

	/*
		Release: v1.19
		Testname: CertificateSigningRequest API Client Certificate
		Description:
		The certificatesigningrequests resource must accept a request for a certificate signed by kubernetes.io/kube-apiserver-client.
		The issued certificate must be valid as a client certificate used to authenticate to the kube-apiserver.
	*/
	ginkgo.It("should support building a client with a CSR", func() {
		const commonName = "tester-csr"

		csrClient := f.ClientSet.CertificatesV1().CertificateSigningRequests()

		pk, err := utils.NewPrivateKey()
		e2eutils.ExpectNoError(err)

		pkder := x509.MarshalPKCS1PrivateKey(pk)
		pkpem := pem.EncodeToMemory(&pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: pkder,
		})

		csrb, err := cert.MakeCSR(pk, &pkix.Name{CommonName: commonName}, nil, nil)
		e2eutils.ExpectNoError(err)

		csrTemplate := &certificatesv1.CertificateSigningRequest{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: commonName + "-",
			},
			Spec: certificatesv1.CertificateSigningRequestSpec{
				Request: csrb,
				Usages: []certificatesv1.KeyUsage{
					certificatesv1.UsageDigitalSignature,
					certificatesv1.UsageKeyEncipherment,
					certificatesv1.UsageClientAuth,
				},
				SignerName:        certificatesv1.KubeAPIServerClientSignerName,
				ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
			},
		}

		// Grant permissions to the new user
		clusterRole, err := f.ClientSet.RbacV1().ClusterRoles().Create(context.TODO(), &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			Rules:      []rbacv1.PolicyRule{{Verbs: []string{"create"}, APIGroups: []string{"certificates.k8s.io"}, Resources: []string{"certificatesigningrequests"}}},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			e2eutils.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				e2eutils.ExpectNoError(f.ClientSet.RbacV1().ClusterRoles().Delete(context.TODO(), clusterRole.Name, metav1.DeleteOptions{}))
			}()
		}

		clusterRoleBinding, err := f.ClientSet.RbacV1().ClusterRoleBindings().Create(context.TODO(), &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			RoleRef:    rbacv1.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole", Name: clusterRole.Name},
			Subjects:   []rbacv1.Subject{{APIGroup: "rbac.authorization.k8s.io", Kind: "User", Name: commonName}},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			e2eutils.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				e2eutils.ExpectNoError(f.ClientSet.RbacV1().ClusterRoleBindings().Delete(context.TODO(), clusterRoleBinding.Name, metav1.DeleteOptions{}))
			}()
		}

		e2eutils.Logf("creating CSR")
		csr, err := csrClient.Create(context.TODO(), csrTemplate, metav1.CreateOptions{})
		e2eutils.ExpectNoError(err)
		defer func() {
			e2eutils.ExpectNoError(csrClient.Delete(context.TODO(), csr.Name, metav1.DeleteOptions{}))
		}()

		e2eutils.Logf("approving CSR")
		e2eutils.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr.Status.Conditions = []certificatesv1.CertificateSigningRequestCondition{
				{
					Type:    certificatesv1.CertificateApproved,
					Status:  v1.ConditionTrue,
					Reason:  "E2E",
					Message: "Set from an e2e test",
				},
			}
			csr, err = csrClient.UpdateApproval(context.TODO(), csr.Name, csr, metav1.UpdateOptions{})
			if err != nil {
				csr, _ = csrClient.Get(context.TODO(), csr.Name, metav1.GetOptions{})
				e2eutils.Logf("err updating approval: %v", err)
				return false, nil
			}
			return true, nil
		}))

		e2eutils.Logf("waiting for CSR to be signed")
		e2eutils.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr, err = csrClient.Get(context.TODO(), csr.Name, metav1.GetOptions{})
			if err != nil {
				e2eutils.Logf("error getting csr: %v", err)
				return false, nil
			}
			if len(csr.Status.Certificate) == 0 {
				e2eutils.Logf("csr not signed yet")
				return false, nil
			}
			return true, nil
		}))

		e2eutils.Logf("testing the client")
		rcfg, err := e2eutils.LoadConfig()
		e2eutils.ExpectNoError(err)
		rcfg = rest.AnonymousClientConfig(rcfg)
		rcfg.TLSClientConfig.CertData = csr.Status.Certificate
		rcfg.TLSClientConfig.KeyData = pkpem

		certs, err := cert.ParseCertsPEM(csr.Status.Certificate)
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(certs), 1, "expected a single cert, got %#v", certs)
		cert := certs[0]
		// make sure the cert is not valid for longer than our requested time (plus allowance for backdating)
		if e, a := time.Hour+5*time.Minute, cert.NotAfter.Sub(cert.NotBefore); a > e {
			e2eutils.Failf("expected cert valid for %s or less, got %s: %s", e, a, dynamiccertificates.GetHumanCertDetail(cert))
		}

		newClient, err := certificatesclient.NewForConfig(rcfg)
		e2eutils.ExpectNoError(err)

		e2eutils.Logf("creating CSR as new client")
		newCSR, err := newClient.CertificateSigningRequests().Create(context.TODO(), csrTemplate, metav1.CreateOptions{})
		e2eutils.ExpectNoError(err)
		defer func() {
			e2eutils.ExpectNoError(csrClient.Delete(context.TODO(), newCSR.Name, metav1.DeleteOptions{}))
		}()
		e2eutils.ExpectEqual(newCSR.Spec.Username, commonName)
	})

	/*
		Release: v1.19
		Testname: CertificateSigningRequest API
		Description:
		The certificates.k8s.io API group MUST exists in the /apis discovery document.
		The certificates.k8s.io/v1 API group/version MUST exist in the /apis/certificates.k8s.io discovery document.
		The certificatesigningrequests, certificatesigningrequests/approval, and certificatesigningrequests/status
		  resources MUST exist in the /apis/certificates.k8s.io/v1 discovery document.
		The certificatesigningrequests resource must support create, get, list, watch, update, patch, delete, and deletecollection.
		The certificatesigningrequests/approval resource must support get, update, patch.
		The certificatesigningrequests/status resource must support get, update, patch.
	*/
	framework.ConformanceIt("should support CSR API operations", func() {

		// Setup
		csrVersion := "v1"
		csrClient := f.ClientSet.CertificatesV1().CertificateSigningRequests()
		csrResource := certificatesv1.SchemeGroupVersion.WithResource("certificatesigningrequests")

		pk, err := utils.NewPrivateKey()
		e2eutils.ExpectNoError(err)

		csrData, err := cert.MakeCSR(pk, &pkix.Name{CommonName: "e2e.example.com"}, []string{"e2e.example.com"}, nil)
		e2eutils.ExpectNoError(err)

		certificateData, _, err := cert.GenerateSelfSignedCertKey("e2e.example.com", nil, []string{"e2e.example.com"})
		e2eutils.ExpectNoError(err)
		certificateDataJSON, err := json.Marshal(certificateData)
		e2eutils.ExpectNoError(err)

		signerName := "example.com/e2e-" + f.UniqueName
		csrTemplate := &certificatesv1.CertificateSigningRequest{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-example-csr-"},
			Spec: certificatesv1.CertificateSigningRequestSpec{
				Request:           csrData,
				SignerName:        signerName,
				ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				Usages:            []certificatesv1.KeyUsage{certificatesv1.UsageDigitalSignature, certificatesv1.UsageKeyEncipherment, certificatesv1.UsageServerAuth},
			},
		}

		// Discovery

		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			e2eutils.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == certificatesv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == csrVersion {
							found = true
							break
						}
					}
				}
			}
			e2eutils.ExpectEqual(found, true, fmt.Sprintf("expected certificates API group/version, got %#v", discoveryGroups.Groups))
		}

		ginkgo.By("getting /apis/certificates.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/certificates.k8s.io").Do(context.TODO()).Into(group)
			e2eutils.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == csrVersion {
					found = true
					break
				}
			}
			e2eutils.ExpectEqual(found, true, fmt.Sprintf("expected certificates API version, got %#v", group.Versions))
		}

		ginkgo.By("getting /apis/certificates.k8s.io/" + csrVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(certificatesv1.SchemeGroupVersion.String())
			e2eutils.ExpectNoError(err)
			foundCSR, foundApproval, foundStatus := false, false, false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "certificatesigningrequests":
					foundCSR = true
				case "certificatesigningrequests/approval":
					foundApproval = true
				case "certificatesigningrequests/status":
					foundStatus = true
				}
			}
			e2eutils.ExpectEqual(foundCSR, true, fmt.Sprintf("expected certificatesigningrequests, got %#v", resources.APIResources))
			e2eutils.ExpectEqual(foundApproval, true, fmt.Sprintf("expected certificatesigningrequests/approval, got %#v", resources.APIResources))
			e2eutils.ExpectEqual(foundStatus, true, fmt.Sprintf("expected certificatesigningrequests/status, got %#v", resources.APIResources))
		}

		// Main resource create/read/update/watch operations

		ginkgo.By("creating")
		_, err = csrClient.Create(context.TODO(), csrTemplate, metav1.CreateOptions{})
		e2eutils.ExpectNoError(err)
		_, err = csrClient.Create(context.TODO(), csrTemplate, metav1.CreateOptions{})
		e2eutils.ExpectNoError(err)
		createdCSR, err := csrClient.Create(context.TODO(), csrTemplate, metav1.CreateOptions{})
		e2eutils.ExpectNoError(err)

		ginkgo.By("getting")
		gottenCSR, err := csrClient.Get(context.TODO(), createdCSR.Name, metav1.GetOptions{})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(gottenCSR.UID, createdCSR.UID)
		e2eutils.ExpectEqual(gottenCSR.Spec.ExpirationSeconds, csr.DurationToExpirationSeconds(time.Hour))

		ginkgo.By("listing")
		csrs, err := csrClient.List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(csrs.Items), 3, "filtered list should have 3 items")

		ginkgo.By("watching")
		e2eutils.Logf("starting watch")
		csrWatch, err := csrClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: csrs.ResourceVersion, FieldSelector: "metadata.name=" + createdCSR.Name})
		e2eutils.ExpectNoError(err)

		ginkgo.By("patching")
		patchedCSR, err := csrClient.Patch(context.TODO(), createdCSR.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(patchedCSR.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		csrToUpdate := patchedCSR.DeepCopy()
		csrToUpdate.Annotations["updated"] = "true"
		updatedCSR, err := csrClient.Update(context.TODO(), csrToUpdate, metav1.UpdateOptions{})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(updatedCSR.Annotations["updated"], "true", "updated object should have the applied annotation")

		e2eutils.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-csrWatch.ResultChan():
				e2eutils.ExpectEqual(ok, true, "watch channel should not close")
				e2eutils.ExpectEqual(evt.Type, watch.Modified)
				watchedCSR, isCSR := evt.Object.(*certificatesv1.CertificateSigningRequest)
				e2eutils.ExpectEqual(isCSR, true, fmt.Sprintf("expected CSR, got %T", evt.Object))
				if watchedCSR.Annotations["patched"] == "true" {
					e2eutils.Logf("saw patched and updated annotations")
					sawAnnotations = true
					csrWatch.Stop()
				} else {
					e2eutils.Logf("missing expected annotations, waiting: %#v", watchedCSR.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				e2eutils.Fail("timed out waiting for watch event")
			}
		}

		// /approval subresource operations

		ginkgo.By("getting /approval")
		gottenApproval, err := f.DynamicClient.Resource(csrResource).Get(context.TODO(), createdCSR.Name, metav1.GetOptions{}, "approval")
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(gottenApproval.GetObjectKind().GroupVersionKind(), certificatesv1.SchemeGroupVersion.WithKind("CertificateSigningRequest"))
		e2eutils.ExpectEqual(gottenApproval.GetUID(), createdCSR.UID)

		ginkgo.By("patching /approval")
		patchedApproval, err := csrClient.Patch(context.TODO(), createdCSR.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedapproval":"true"}},"status":{"conditions":[{"type":"ApprovalPatch","status":"True","reason":"e2e"}]}}`),
			metav1.PatchOptions{}, "approval")
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(patchedApproval.Status.Conditions), 1, fmt.Sprintf("patched object should have the applied condition, got %#v", patchedApproval.Status.Conditions))
		e2eutils.ExpectEqual(string(patchedApproval.Status.Conditions[0].Type), "ApprovalPatch", fmt.Sprintf("patched object should have the applied condition, got %#v", patchedApproval.Status.Conditions))
		e2eutils.ExpectEqual(patchedApproval.Annotations["patchedapproval"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating /approval")
		approvalToUpdate := patchedApproval.DeepCopy()
		approvalToUpdate.Status.Conditions = append(approvalToUpdate.Status.Conditions, certificatesv1.CertificateSigningRequestCondition{
			Type:    certificatesv1.CertificateApproved,
			Status:  v1.ConditionTrue,
			Reason:  "E2E",
			Message: "Set from an e2e test",
		})
		updatedApproval, err := csrClient.UpdateApproval(context.TODO(), approvalToUpdate.Name, approvalToUpdate, metav1.UpdateOptions{})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(updatedApproval.Status.Conditions), 2, fmt.Sprintf("updated object should have the applied condition, got %#v", updatedApproval.Status.Conditions))
		e2eutils.ExpectEqual(updatedApproval.Status.Conditions[1].Type, certificatesv1.CertificateApproved, fmt.Sprintf("updated object should have the approved condition, got %#v", updatedApproval.Status.Conditions))

		// /status subresource operations

		ginkgo.By("getting /status")
		gottenStatus, err := f.DynamicClient.Resource(csrResource).Get(context.TODO(), createdCSR.Name, metav1.GetOptions{}, "status")
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(gottenStatus.GetObjectKind().GroupVersionKind(), certificatesv1.SchemeGroupVersion.WithKind("CertificateSigningRequest"))
		e2eutils.ExpectEqual(gottenStatus.GetUID(), createdCSR.UID)

		ginkgo.By("patching /status")
		patchedStatus, err := csrClient.Patch(context.TODO(), createdCSR.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"certificate":`+string(certificateDataJSON)+`}}`),
			metav1.PatchOptions{}, "status")
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(patchedStatus.Status.Certificate, certificateData, "patched object should have the applied certificate")
		e2eutils.ExpectEqual(patchedStatus.Annotations["patchedstatus"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		statusToUpdate := patchedStatus.DeepCopy()
		statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, certificatesv1.CertificateSigningRequestCondition{
			Type:    "StatusUpdate",
			Status:  v1.ConditionTrue,
			Reason:  "E2E",
			Message: "Set from an e2e test",
		})
		updatedStatus, err := csrClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(updatedStatus.Status.Conditions), len(statusToUpdate.Status.Conditions), fmt.Sprintf("updated object should have the applied condition, got %#v", updatedStatus.Status.Conditions))
		e2eutils.ExpectEqual(string(updatedStatus.Status.Conditions[len(updatedStatus.Status.Conditions)-1].Type), "StatusUpdate", fmt.Sprintf("updated object should have the approved condition, got %#v", updatedStatus.Status.Conditions))

		// main resource delete operations

		ginkgo.By("deleting")
		err = csrClient.Delete(context.TODO(), createdCSR.Name, metav1.DeleteOptions{})
		e2eutils.ExpectNoError(err)
		_, err = csrClient.Get(context.TODO(), createdCSR.Name, metav1.GetOptions{})
		e2eutils.ExpectEqual(apierrors.IsNotFound(err), true, fmt.Sprintf("expected 404, got %#v", err))
		csrs, err = csrClient.List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(csrs.Items), 2, "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = csrClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		e2eutils.ExpectNoError(err)
		csrs, err = csrClient.List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(csrs.Items), 0, "filtered list should have 0 items")
	})
})
