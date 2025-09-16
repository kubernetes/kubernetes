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

package projected

import (
	"bytes"
	"context"
	"fmt"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/configmap"
	"k8s.io/kubernetes/pkg/volume/downwardapi"
	"k8s.io/kubernetes/pkg/volume/secret"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	utilstrings "k8s.io/utils/strings"
)

// ProbeVolumePlugins is the entry point for plugin detection in a package.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&projectedPlugin{}}
}

const (
	projectedPluginName = "kubernetes.io/projected"
)

type projectedPlugin struct {
	host                      volume.VolumeHost
	kvHost                    volume.KubeletVolumeHost
	getSecret                 func(namespace, name string) (*v1.Secret, error)
	getConfigMap              func(namespace, name string) (*v1.ConfigMap, error)
	getServiceAccountToken    func(namespace, name string, tr *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error)
	deleteServiceAccountToken func(podUID types.UID)
}

var _ volume.VolumePlugin = &projectedPlugin{}

func wrappedVolumeSpec() volume.Spec {
	return volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
			},
		},
	}
}

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(projectedPluginName), volName)
}

func (plugin *projectedPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.kvHost = host.(volume.KubeletVolumeHost)
	plugin.getSecret = host.GetSecretFunc()
	plugin.getConfigMap = host.GetConfigMapFunc()
	plugin.getServiceAccountToken = host.GetServiceAccountTokenFunc()
	plugin.deleteServiceAccountToken = host.DeleteServiceAccountTokenFunc()
	return nil
}

func (plugin *projectedPlugin) GetPluginName() string {
	return projectedPluginName
}

func (plugin *projectedPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	_, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return spec.Name(), nil
}

func (plugin *projectedPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.Projected != nil
}

func (plugin *projectedPlugin) RequiresRemount(spec *volume.Spec) bool {
	return true
}

func (plugin *projectedPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *projectedPlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return false, nil
}

func (plugin *projectedPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	return &projectedVolumeMounter{
		projectedVolume: &projectedVolume{
			volName:         spec.Name(),
			sources:         spec.Volume.Projected.Sources,
			podUID:          pod.UID,
			plugin:          plugin,
			MetricsProvider: volume.NewCachedMetrics(volume.NewMetricsDu(getPath(pod.UID, spec.Name(), plugin.host))),
		},
		source: *spec.Volume.Projected,
		pod:    pod,
	}, nil
}

func (plugin *projectedPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &projectedVolumeUnmounter{
		&projectedVolume{
			volName:         volName,
			podUID:          podUID,
			plugin:          plugin,
			MetricsProvider: volume.NewCachedMetrics(volume.NewMetricsDu(getPath(podUID, volName, plugin.host))),
		},
	}, nil
}

func (plugin *projectedPlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	projectedVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Projected: &v1.ProjectedVolumeSource{},
		},
	}

	return volume.ReconstructedVolume{
		Spec: volume.NewSpecFromVolume(projectedVolume),
	}, nil
}

type projectedVolume struct {
	volName string
	sources []v1.VolumeProjection
	podUID  types.UID
	plugin  *projectedPlugin
	volume.MetricsProvider
}

var _ volume.Volume = &projectedVolume{}

func (sv *projectedVolume) GetPath() string {
	return getPath(sv.podUID, sv.volName, sv.plugin.host)
}

type projectedVolumeMounter struct {
	*projectedVolume

	source v1.ProjectedVolumeSource
	pod    *v1.Pod
}

var _ volume.Mounter = &projectedVolumeMounter{}

func (sv *projectedVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       true,
		Managed:        true,
		SELinuxRelabel: true,
	}

}

func (s *projectedVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return s.SetUpAt(s.GetPath(), mounterArgs)
}

func (s *projectedVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	klog.V(3).Infof("Setting up volume %v for pod %v at %v", s.volName, s.pod.UID, dir)

	wrapped, err := s.plugin.host.NewWrapperMounter(s.volName, wrappedVolumeSpec(), s.pod)
	if err != nil {
		return err
	}

	data, err := s.collectData(mounterArgs)
	if err != nil {
		klog.Errorf("Error preparing data for projected volume %v for pod %v/%v: %s", s.volName, s.pod.Namespace, s.pod.Name, err.Error())
		return err
	}

	setupSuccess := false
	if err := wrapped.SetUpAt(dir, mounterArgs); err != nil {
		return err
	}

	if err := volumeutil.MakeNestedMountpoints(s.volName, dir, *s.pod); err != nil {
		return err
	}

	defer func() {
		// Clean up directories if setup fails
		if !setupSuccess {
			unmounter, unmountCreateErr := s.plugin.NewUnmounter(s.volName, s.podUID)
			if unmountCreateErr != nil {
				klog.Errorf("error cleaning up mount %s after failure. Create unmounter failed with %v", s.volName, unmountCreateErr)
				return
			}
			tearDownErr := unmounter.TearDown()
			if tearDownErr != nil {
				klog.Errorf("error tearing down volume %s with : %v", s.volName, tearDownErr)
			}
		}
	}()

	writerContext := fmt.Sprintf("pod %v/%v volume %v", s.pod.Namespace, s.pod.Name, s.volName)
	writer, err := volumeutil.NewAtomicWriter(dir, writerContext)
	if err != nil {
		klog.Errorf("Error creating atomic writer: %v", err)
		return err
	}

	setPerms := func(_ string) error {
		// This may be the first time writing and new files get created outside the timestamp subdirectory:
		// change the permissions on the whole volume and not only in the timestamp directory.
		ownershipChanger := volume.NewVolumeOwnership(s, dir, mounterArgs.FsGroup, nil /*fsGroupChangePolicy*/, volumeutil.FSGroupCompleteHook(s.plugin, nil))
		return ownershipChanger.ChangePermissions()
	}
	err = writer.Write(data, setPerms)
	if err != nil {
		klog.Errorf("Error writing payload to dir: %v", err)
		return err
	}

	setupSuccess = true
	return nil
}

func (s *projectedVolumeMounter) collectData(mounterArgs volume.MounterArgs) (map[string]volumeutil.FileProjection, error) {
	if s.source.DefaultMode == nil {
		return nil, fmt.Errorf("no defaultMode used, not even the default value for it")
	}

	kubeClient := s.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("cannot setup projected volume %v because kube client is not configured", s.volName)
	}

	errlist := []error{}
	payload := make(map[string]volumeutil.FileProjection)
	for sourceIndex, source := range s.source.Sources {
		switch {
		case source.Secret != nil:
			optional := source.Secret.Optional != nil && *source.Secret.Optional
			secretapi, err := s.plugin.getSecret(s.pod.Namespace, source.Secret.Name)
			if err != nil {
				if !(errors.IsNotFound(err) && optional) {
					klog.Errorf("Couldn't get secret %v/%v: %v", s.pod.Namespace, source.Secret.Name, err)
					errlist = append(errlist, err)
					continue
				}
				secretapi = &v1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: s.pod.Namespace,
						Name:      source.Secret.Name,
					},
				}
			}
			secretPayload, err := secret.MakePayload(source.Secret.Items, secretapi, s.source.DefaultMode, optional)
			if err != nil {
				klog.Errorf("Couldn't get secret payload %v/%v: %v", s.pod.Namespace, source.Secret.Name, err)
				errlist = append(errlist, err)
				continue
			}
			for k, v := range secretPayload {
				payload[k] = v
			}
		case source.ConfigMap != nil:
			optional := source.ConfigMap.Optional != nil && *source.ConfigMap.Optional
			configMap, err := s.plugin.getConfigMap(s.pod.Namespace, source.ConfigMap.Name)
			if err != nil {
				if !(errors.IsNotFound(err) && optional) {
					klog.Errorf("Couldn't get configMap %v/%v: %v", s.pod.Namespace, source.ConfigMap.Name, err)
					errlist = append(errlist, err)
					continue
				}
				configMap = &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: s.pod.Namespace,
						Name:      source.ConfigMap.Name,
					},
				}
			}
			configMapPayload, err := configmap.MakePayload(source.ConfigMap.Items, configMap, s.source.DefaultMode, optional)
			if err != nil {
				klog.Errorf("Couldn't get configMap payload %v/%v: %v", s.pod.Namespace, source.ConfigMap.Name, err)
				errlist = append(errlist, err)
				continue
			}
			for k, v := range configMapPayload {
				payload[k] = v
			}
		case source.DownwardAPI != nil:
			downwardAPIPayload, err := downwardapi.CollectData(source.DownwardAPI.Items, s.pod, s.plugin.host, s.source.DefaultMode)
			if err != nil {
				errlist = append(errlist, err)
				continue
			}
			for k, v := range downwardAPIPayload {
				payload[k] = v
			}
		case source.ServiceAccountToken != nil:
			tp := source.ServiceAccountToken

			// When FsGroup is set, we depend on SetVolumeOwnership to
			// change from 0600 to 0640.
			mode := *s.source.DefaultMode
			if mounterArgs.FsUser != nil || mounterArgs.FsGroup != nil {
				mode = 0600
			}

			var auds []string
			if len(tp.Audience) != 0 {
				auds = []string{tp.Audience}
			}
			tr, err := s.plugin.getServiceAccountToken(s.pod.Namespace, s.pod.Spec.ServiceAccountName, &authenticationv1.TokenRequest{
				Spec: authenticationv1.TokenRequestSpec{
					Audiences:         auds,
					ExpirationSeconds: tp.ExpirationSeconds,
					BoundObjectRef: &authenticationv1.BoundObjectReference{
						APIVersion: "v1",
						Kind:       "Pod",
						Name:       s.pod.Name,
						UID:        s.pod.UID,
					},
				},
			})
			if err != nil {
				errlist = append(errlist, err)
				continue
			}
			payload[tp.Path] = volumeutil.FileProjection{
				Data:   []byte(tr.Status.Token),
				Mode:   mode,
				FsUser: mounterArgs.FsUser,
			}
		case source.ClusterTrustBundle != nil:
			allowEmpty := false
			if source.ClusterTrustBundle.Optional != nil && *source.ClusterTrustBundle.Optional {
				allowEmpty = true
			}

			var trustAnchors []byte
			if source.ClusterTrustBundle.Name != nil {
				var err error
				trustAnchors, err = s.plugin.kvHost.GetTrustAnchorsByName(*source.ClusterTrustBundle.Name, allowEmpty)
				if err != nil {
					errlist = append(errlist, err)
					continue
				}
			} else if source.ClusterTrustBundle.SignerName != nil {
				var err error
				trustAnchors, err = s.plugin.kvHost.GetTrustAnchorsBySigner(*source.ClusterTrustBundle.SignerName, source.ClusterTrustBundle.LabelSelector, allowEmpty)
				if err != nil {
					errlist = append(errlist, err)
					continue
				}
			} else {
				errlist = append(errlist, fmt.Errorf("ClusterTrustBundle projection requires either name or signerName to be set"))
				continue
			}

			mode := *s.source.DefaultMode
			if mounterArgs.FsUser != nil || mounterArgs.FsGroup != nil {
				mode = 0600
			}

			payload[source.ClusterTrustBundle.Path] = volumeutil.FileProjection{
				Data:   trustAnchors,
				Mode:   mode,
				FsUser: mounterArgs.FsUser,
			}
		case source.PodCertificate != nil:
			key, certificates, err := s.plugin.kvHost.GetPodCertificateCredentialBundle(context.TODO(), s.pod.ObjectMeta.Namespace, s.pod.ObjectMeta.Name, string(s.pod.ObjectMeta.UID), s.volName, sourceIndex)
			if err != nil {
				errlist = append(errlist, err)
				continue
			}

			mode := *s.source.DefaultMode
			if mounterArgs.FsUser != nil || mounterArgs.FsGroup != nil {
				mode = 0600
			}

			if source.PodCertificate.CredentialBundlePath != "" {
				credentialBundle := bytes.Buffer{}
				credentialBundle.Write(key)
				credentialBundle.Write(certificates)
				payload[source.PodCertificate.CredentialBundlePath] = volumeutil.FileProjection{
					Data:   credentialBundle.Bytes(),
					Mode:   mode,
					FsUser: mounterArgs.FsUser,
				}
			}
			if source.PodCertificate.KeyPath != "" {
				payload[source.PodCertificate.KeyPath] = volumeutil.FileProjection{
					Data:   key,
					Mode:   mode,
					FsUser: mounterArgs.FsUser,
				}
			}
			if source.PodCertificate.CertificateChainPath != "" {
				payload[source.PodCertificate.CertificateChainPath] = volumeutil.FileProjection{
					Data:   certificates,
					Mode:   mode,
					FsUser: mounterArgs.FsUser,
				}
			}

		}
	}
	return payload, utilerrors.NewAggregate(errlist)
}

type projectedVolumeUnmounter struct {
	*projectedVolume
}

var _ volume.Unmounter = &projectedVolumeUnmounter{}

func (c *projectedVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *projectedVolumeUnmounter) TearDownAt(dir string) error {
	klog.V(3).Infof("Tearing down volume %v for pod %v at %v", c.volName, c.podUID, dir)

	wrapped, err := c.plugin.host.NewWrapperUnmounter(c.volName, wrappedVolumeSpec(), c.podUID)
	if err != nil {
		return err
	}
	if err = wrapped.TearDownAt(dir); err != nil {
		return err
	}

	c.plugin.deleteServiceAccountToken(c.podUID)

	return nil
}

func getVolumeSource(spec *volume.Spec) (*v1.ProjectedVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Projected != nil {
		return spec.Volume.Projected, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("spec does not reference a projected volume type")
}
