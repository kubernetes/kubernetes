package kubemaster

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/square/go-jose"

	"k8s.io/kubernetes/pkg/api"
	unversionedapi "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	//"k8s.io/kubernetes/pkg/client/unversioned"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
)

func NewDiscoveryEndpoint(params *kubeadmapi.BootstrapParams, caCert string) {

	clusterInfo, err := json.MarshalIndent(kubeadmapi.ClusterInfo{
		Endpoints:              []string{fmt.Sprintf("https://%s:443", params.Discovery.ListenIP)},
		CertificateAuthorities: []string{caCert},
	}, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	signer, err := jose.NewSigner(jose.SignatureAlgorithm("HS256"), []byte(params.Discovery.BearerToken))
	if err != nil {
		log.Fatal(err)
	}

	http.HandleFunc("/api/v1alpha1/testclusterinfo", func(w http.ResponseWriter, r *http.Request) {

		obj, err := signer.Sign(clusterInfo)
		if err != nil {
			log.Println("Error:", err)
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintln(w, "Error:", err)
			return
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, obj.FullSerialize())
		return
	})

	listener, err := net.ListenTCP("tcp", &net.TCPAddr{Port: 8081})
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Listening on %v", listener.Addr())
	log.Printf("Shared secret is %q", params.Discovery.BearerToken)
	log.Fatal(http.Serve(listener, nil))

}

func CreateDiscoveryDeploymentAndSecret(params *kubeadmapi.BootstrapParams, client *clientset.Clientset, caCert string) error {
	//TODO create the secret
	//zero := int64(0)
	l := map[string]string{"name": "kube-discovery"}
	kubediscoveryDeployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{Name: "kube-discovery"},
		Spec: extensions.DeploymentSpec{
			Replicas: 1,
			Selector: &unversionedapi.LabelSelector{MatchLabels: l},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{Labels: l},
				Spec: api.PodSpec{
					//TerminationGracePeriodSeconds: &zero,
					SecurityContext: &api.PodSecurityContext{HostNetwork: true},
					Containers: []api.Container{
						{
							Name:    "kube-discovery",
							Image:   HYPERKUBE_IMAGE,
							Command: []string{"/usr/bin/kube-discovery"},
							VolumeMounts: []api.VolumeMount{
								{
									Name:      "clusterinfosecret",
									MountPath: "/tmp/secret", // TODO use a shared constant
									ReadOnly:  true,
								},
							},
						},
					},
					Volumes: []api.Volume{
						{
							Name: "clusterinfosecret",
							VolumeSource: api.VolumeSource{
								Secret: &api.SecretVolumeSource{
									SecretName: "clusterinfo",
								},
							},
						},
					},
				},
			},
		},
	}

	// TODO ListenIP is probably not the right now, although it's best we have right now
	// if user provides a DNS name, or anything else, we should use that, may be it's really
	// the list of all SANs (minus internal DNS names and service IP)?
	endpointList := []string{fmt.Sprintf("https://%s:443", params.Discovery.ListenIP)}

	tokenMap := map[string]string{"todo": params.Discovery.BearerToken} // TODO should use dot-separated tokens

	secretData := map[string][]byte{
		"ca.pem": []byte(caCert),
	}
	var err error
	secretData["endpoint-list.json"], err = json.MarshalIndent(endpointList, "", "  ")
	if err != nil {
		return err
	}
	secretData["token-map.json"], err = json.MarshalIndent(tokenMap, "", "  ")
	if err != nil {
		return err
	}

	kubediscoverySecret := &api.Secret{
		ObjectMeta: api.ObjectMeta{Name: "clusterinfo"},
		Data:       secretData,
		Type:       api.SecretTypeOpaque,
	}

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kubediscoveryDeployment); err != nil {
		return err
	}
	if _, err := client.Secrets(api.NamespaceSystem).Create(kubediscoverySecret); err != nil {
		return err
	}

	return nil
}
