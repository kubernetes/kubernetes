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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

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
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Certificates API [Privileged:ClusterAdmin]", func() {
	f := framework.NewDefaultFramework("certificates")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.19
		Testname: CertificateSigningRequest API Client Certificate
		Description:
		The certificatesigningrequests resource must accept a request for a certificate signed by kubernetes.io/kube-apiserver-client.
		The issued certificate must be valid as a client certificate used to authenticate to the kube-apiserver.
	*/
	ginkgo.It("should support building a client with a CSR", func(ctx context.Context) {
		const commonName = "tester-csr"

		csrClient := f.ClientSet.CertificatesV1().CertificateSigningRequests()

		pk, err := utils.NewPrivateKey()
		framework.ExpectNoError(err)

		pkder := x509.MarshalPKCS1PrivateKey(pk)
		pkpem := pem.EncodeToMemory(&pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: pkder,
		})

		csrb, err := cert.MakeCSR(pk, &pkix.Name{CommonName: commonName}, nil, nil)
		framework.ExpectNoError(err)

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
		clusterRole, err := f.ClientSet.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			Rules:      []rbacv1.PolicyRule{{Verbs: []string{"create"}, APIGroups: []string{"certificates.k8s.io"}, Resources: []string{"certificatesigningrequests"}}},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			framework.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				framework.ExpectNoError(f.ClientSet.RbacV1().ClusterRoles().Delete(ctx, clusterRole.Name, metav1.DeleteOptions{}))
			}()
		}

		clusterRoleBinding, err := f.ClientSet.RbacV1().ClusterRoleBindings().Create(ctx, &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			RoleRef:    rbacv1.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole", Name: clusterRole.Name},
			Subjects:   []rbacv1.Subject{{APIGroup: "rbac.authorization.k8s.io", Kind: "User", Name: commonName}},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			framework.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				framework.ExpectNoError(f.ClientSet.RbacV1().ClusterRoleBindings().Delete(ctx, clusterRoleBinding.Name, metav1.DeleteOptions{}))
			}()
		}

		framework.Logf("creating CSR")
		csr, err := csrClient.Create(ctx, csrTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(csrClient.Delete(ctx, csr.Name, metav1.DeleteOptions{}))
		}()

		framework.Logf("approving CSR")
		framework.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr.Status.Conditions = []certificatesv1.CertificateSigningRequestCondition{
				{
					Type:    certificatesv1.CertificateApproved,
					Status:  v1.ConditionTrue,
					Reason:  "E2E",
					Message: "Set from an e2e test",
				},
			}
			csr, err = csrClient.UpdateApproval(ctx, csr.Name, csr, metav1.UpdateOptions{})
			if err != nil {
				csr, _ = csrClient.Get(ctx, csr.Name, metav1.GetOptions{})
				framework.Logf("err updating approval: %v", err)
				return false, nil
			}
			return true, nil
		}))

		framework.Logf("waiting for CSR to be signed")
		framework.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr, err = csrClient.Get(ctx, csr.Name, metav1.GetOptions{})
			if err != nil {
				framework.Logf("error getting csr: %v", err)
				return false, nil
			}
			if len(csr.Status.Certificate) == 0 {
				framework.Logf("csr not signed yet")
				return false, nil
			}
			return true, nil
		}))

		framework.Logf("testing the client")
		rcfg, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		rcfg = rest.AnonymousClientConfig(rcfg)
		rcfg.TLSClientConfig.CertData = csr.Status.Certificate
		rcfg.TLSClientConfig.KeyData = pkpem

		certs, err := cert.ParseCertsPEM(csr.Status.Certificate)
		framework.ExpectNoError(err)
		gomega.Expect(certs).To(gomega.HaveLen(1), "expected a single cert, got %#v", certs)
		cert := certs[0]
		// make sure the cert is not valid for longer than our requested time (plus allowance for backdating)
		if e, a := time.Hour+5*time.Minute, cert.NotAfter.Sub(cert.NotBefore); a > e {
			framework.Failf("expected cert valid for %s or less, got %s: %s", e, a, dynamiccertificates.GetHumanCertDetail(cert))
		}

		newClient, err := certificatesclient.NewForConfig(rcfg)
		framework.ExpectNoError(err)

		framework.Logf("creating CSR as new client")
		newCSR, err := newClient.CertificateSigningRequests().Create(ctx, csrTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(csrClient.Delete(ctx, newCSR.Name, metav1.DeleteOptions{}))
		}()
		gomega.Expect(newCSR.Spec.Username).To(gomega.Equal(commonName))
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
	framework.ConformanceIt("should support CSR API operations", func(ctx context.Context) {

		// Setup
		csrVersion := "v1"
		csrClient := f.ClientSet.CertificatesV1().CertificateSigningRequests()
		csrResource := certificatesv1.SchemeGroupVersion.WithResource("certificatesigningrequests")

		pk, err := utils.NewPrivateKey()
		framework.ExpectNoError(err)

		csrData, err := cert.MakeCSR(pk, &pkix.Name{CommonName: "e2e.example.com"}, []string{"e2e.example.com"}, nil)
		framework.ExpectNoError(err)

		certificateData, _, err := cert.GenerateSelfSignedCertKey("e2e.example.com", nil, []string{"e2e.example.com"})
		framework.ExpectNoError(err)
		certificateDataJSON, err := json.Marshal(certificateData)
		framework.ExpectNoError(err)

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
			framework.ExpectNoError(err)
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
			if !found {
				framework.Failf("expected certificates API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/certificates.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/certificates.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == csrVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected certificates API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/certificates.k8s.io/" + csrVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(certificatesv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
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
			if !foundCSR {
				framework.Failf("expected certificatesigningrequests, got %#v", resources.APIResources)
			}
			if !foundApproval {
				framework.Failf("expected certificatesigningrequests/approval, got %#v", resources.APIResources)
			}
			if !foundStatus {
				framework.Failf("expected certificatesigningrequests/status, got %#v", resources.APIResources)
			}
		}

		// Main resource create/read/update/watch operations

		ginkgo.By("creating")
		_, err = csrClient.Create(ctx, csrTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = csrClient.Create(ctx, csrTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdCSR, err := csrClient.Create(ctx, csrTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenCSR, err := csrClient.Get(ctx, createdCSR.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenCSR.UID).To(gomega.Equal(createdCSR.UID))
		gomega.Expect(gottenCSR.Spec.ExpirationSeconds).To(gomega.Equal(csr.DurationToExpirationSeconds(time.Hour)))

		ginkgo.By("listing")
		csrs, err := csrClient.List(ctx, metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		framework.ExpectNoError(err)
		gomega.Expect(csrs.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		csrWatch, err := csrClient.Watch(ctx, metav1.ListOptions{ResourceVersion: csrs.ResourceVersion, FieldSelector: "metadata.name=" + createdCSR.Name})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedCSR, err := csrClient.Patch(ctx, createdCSR.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedCSR.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		csrToUpdate := patchedCSR.DeepCopy()
		csrToUpdate.Annotations["updated"] = "true"
		updatedCSR, err := csrClient.Update(ctx, csrToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedCSR.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-csrWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedCSR, isCSR := evt.Object.(*certificatesv1.CertificateSigningRequest)
				if !isCSR {
					framework.Failf("expected CSR, got %T", evt.Object)
				}
				if watchedCSR.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					csrWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedCSR.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		// /approval subresource operations

		ginkgo.By("getting /approval")
		gottenApproval, err := f.DynamicClient.Resource(csrResource).Get(ctx, createdCSR.Name, metav1.GetOptions{}, "approval")
		framework.ExpectNoError(err)
		gomega.Expect(gottenApproval.GetObjectKind().GroupVersionKind()).To(gomega.Equal(certificatesv1.SchemeGroupVersion.WithKind("CertificateSigningRequest")))
		gomega.Expect(gottenApproval.GetUID()).To(gomega.Equal(createdCSR.UID))

		ginkgo.By("patching /approval")
		patchedApproval, err := csrClient.Patch(ctx, createdCSR.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedapproval":"true"}},"status":{"conditions":[{"type":"ApprovalPatch","status":"True","reason":"e2e"}]}}`),
			metav1.PatchOptions{}, "approval")
		framework.ExpectNoError(err)
		gomega.Expect(patchedApproval.Status.Conditions).To(gomega.HaveLen(1), "patched object should have the applied condition")
		gomega.Expect(string(patchedApproval.Status.Conditions[0].Type)).To(gomega.Equal("ApprovalPatch"), "patched object should have the applied condition, got %#v", patchedApproval.Status.Conditions)
		gomega.Expect(patchedApproval.Annotations).To(gomega.HaveKeyWithValue("patchedapproval", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating /approval")
		approvalToUpdate := patchedApproval.DeepCopy()
		approvalToUpdate.Status.Conditions = append(approvalToUpdate.Status.Conditions, certificatesv1.CertificateSigningRequestCondition{
			Type:    certificatesv1.CertificateApproved,
			Status:  v1.ConditionTrue,
			Reason:  "E2E",
			Message: "Set from an e2e test",
		})
		updatedApproval, err := csrClient.UpdateApproval(ctx, approvalToUpdate.Name, approvalToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedApproval.Status.Conditions).To(gomega.HaveLen(2), "updated object should have the applied condition, got %#v", updatedApproval.Status.Conditions)
		gomega.Expect(updatedApproval.Status.Conditions[1].Type).To(gomega.Equal(certificatesv1.CertificateApproved), "updated object should have the approved condition, got %#v", updatedApproval.Status.Conditions)

		// /status subresource operations

		ginkgo.By("getting /status")
		gottenStatus, err := f.DynamicClient.Resource(csrResource).Get(ctx, createdCSR.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(gottenStatus.GetObjectKind().GroupVersionKind()).To(gomega.Equal(certificatesv1.SchemeGroupVersion.WithKind("CertificateSigningRequest")))
		gomega.Expect(gottenStatus.GetUID()).To(gomega.Equal(createdCSR.UID))

		ginkgo.By("patching /status")
		patchedStatus, err := csrClient.Patch(ctx, createdCSR.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"certificate":`+string(certificateDataJSON)+`}}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(patchedStatus.Status.Certificate).To(gomega.Equal(certificateData), "patched object should have the applied certificate")
		gomega.Expect(patchedStatus.Annotations).To(gomega.HaveKeyWithValue("patchedstatus", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		statusToUpdate := patchedStatus.DeepCopy()
		statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, certificatesv1.CertificateSigningRequestCondition{
			Type:    "StatusUpdate",
			Status:  v1.ConditionTrue,
			Reason:  "E2E",
			Message: "Set from an e2e test",
		})
		updatedStatus, err := csrClient.UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedStatus.Status.Conditions).To(gomega.HaveLen(len(statusToUpdate.Status.Conditions)), "updated object should have the applied condition, got %#v", updatedStatus.Status.Conditions)
		gomega.Expect(string(updatedStatus.Status.Conditions[len(updatedStatus.Status.Conditions)-1].Type)).To(gomega.Equal("StatusUpdate"), "updated object should have the approved condition, got %#v", updatedStatus.Status.Conditions)

		// main resource delete operations

		ginkgo.By("deleting")
		err = csrClient.Delete(ctx, createdCSR.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = csrClient.Get(ctx, createdCSR.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}
		csrs, err = csrClient.List(ctx, metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		framework.ExpectNoError(err)
		gomega.Expect(csrs.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = csrClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		framework.ExpectNoError(err)
		csrs, err = csrClient.List(ctx, metav1.ListOptions{FieldSelector: "spec.signerName=" + signerName})
		framework.ExpectNoError(err)
		gomega.Expect(csrs.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})
})
