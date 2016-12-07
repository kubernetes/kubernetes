package master

import (
	"fmt"

	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	ext "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
)

// Sources from bootkube templates.go
func getAPIServerDS(cfg *kubeadmapi.MasterConfiguration) ext.DaemonSet {

	volumes := []v1.Volume{k8sVolume(cfg)}
	volumeMounts := []v1.VolumeMount{k8sVolumeMount()}

	if isCertsVolumeMountNeeded() {
		volumes = append(volumes, certsVolume(cfg))
		volumeMounts = append(volumeMounts, certsVolumeMount())
	}

	if isPkiVolumeMountNeeded() {
		volumes = append(volumes, pkiVolume(cfg))
		volumeMounts = append(volumeMounts, pkiVolumeMount())
	}

	ds := ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      kubeAPIServer,
			Namespace: "kube-system",
			// TODO: label from bootkube, not sure if necessary
			Labels: map[string]string{"k8s-app": "kube-apiserver"},
		},
		Spec: ext.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app":   "kube-apiserver", // # TODO: from bootkube, not sure if necessary
						"tier":      "control-plane",
						"component": kubeAPIServer,
					},
				},
				Spec: v1.PodSpec{
					// TODO: Make sure masters get this label
					NodeSelector: map[string]string{metav1.NodeLabelKubeadmAlphaRole: metav1.NodeLabelRoleMaster},
					HostNetwork:  true,
					Volumes:      volumes,

					Containers: []v1.Container{
						v1.Container{
							Name:    kubeAPIServer,
							Image:   images.GetCoreImage(images.KubeAPIServerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command: getAPIServerCommand(cfg),
							Env: []v1.EnvVar{
								v1.EnvVar{
									Name: "MY_POD_IP",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "status.podIP",
										},
									},
								},
							},
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(8080, "/healthz"),
							Resources:     componentResources("250m"),
						},
					},
				},
			},
		},
	}
	return ds
}

func CreateSelfHostedControlPlane(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	ds := getAPIServerDS(cfg)
	fmt.Printf("%+v\n", ds)
	if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(&ds); err != nil {
		return fmt.Errorf("failed to create self-hosted %q DaemonSet [%v]", kubeAPIServer, err)
	}
	return nil
}
