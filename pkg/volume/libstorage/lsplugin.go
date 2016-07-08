/*
Copyright 2016 The Kubernetes Authors.

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

package libstorage

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/akutz/gofig"
	"github.com/golang/glog"

	lsctx "github.com/emccode/libstorage/api/context"
	lstypes "github.com/emccode/libstorage/api/types"
	lsclient "github.com/emccode/libstorage/client"
)

type lsPlugin struct {
	host    volume.VolumeHost
	opts    string
	service string
	client  lstypes.Client
	ctx     lstypes.Context
	cfg     gofig.Config
}

var _ volume.VolumePlugin = &lsPlugin{}
var _ volume.PersistentVolumePlugin = &lsPlugin{}
var _ volume.ProvisionableVolumePlugin = &lsPlugin{}
var _ volume.DeletableVolumePlugin = &lsPlugin{}

var lsValidOpts = []string{
	"service",
	"host",
}

const ymlTemplate = `libstorage:
  host: %s
`

const (
	lsPluginName  = "kubernetes.io/libstorage"
	lsServiceName = "kubernetes"
	lsDefaultOpts = "service=kubernetes;host=tcp://127.0.0.1:7981"
)

//ProbeVolumePlugins is the entry point for libStorage plugin implementation.
//It expects a semicolon-delimited list of key/pair values used to configure the
//libstorage client:
// service=kubernetes;host=127.0.0.1;...;etc
func ProbeVolumePlugins(opts string) []volume.VolumePlugin {
	p := &lsPlugin{
		host:    nil,
		opts:    opts,
		service: lsServiceName,
	}
	if opts == "" {
		p.opts = lsDefaultOpts
	}
	return []volume.VolumePlugin{p}
}

func (p *lsPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	return p.initLibStorage() // setup internal client
}

func (p *lsPlugin) GetPluginName() string {
	return lsPluginName
}

func (p *lsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LibStorage != nil) ||
		(spec.Volume != nil && spec.Volume.LibStorage != nil)
}

func (p *lsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	source, err := p.getLibStorageSource(spec)
	if err != nil {
		return "", err
	}
	return source.VolumeName, nil
}

func (p *lsPlugin) RequiresRemount() bool {
	return false
}

func (p *lsPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {
	lsVol, err := p.getLibStorageSource(spec)
	if err != nil {
		return nil, err
	}
	volName := lsVol.VolumeName
	volID := lsVol.VolumeID
	volRO := spec.ReadOnly

	return &lsVolume{
		podUID: pod.UID,
		volume: &lstypes.Volume{
			Name: volName,
			ID:   volID,
		},
		mounter:  mount.New(),
		plugin:   p,
		readOnly: volRO,
		k8mtx:    keymutex.NewKeyMutex(),
	}, nil
}

func (p *lsPlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return &lsVolume{
		podUID: podUID,
		volume: &lstypes.Volume{
			Name: name,
		},
		plugin:  p,
		mounter: mount.New(),
	}, nil
}

func (p *lsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (p *lsPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	glog.V(4).Infof("libStorage: creating Provisioner with options %+v\n", options)
	if len(options.AccessModes) == 0 {
		options.AccessModes = p.GetAccessModes()
	}

	return &lsVolume{
		mounter: mount.New(),
		plugin:  p,
		options: options,
	}, nil
}

func (p *lsPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	vol := new(lstypes.Volume)
	vol.Name = spec.PersistentVolume.Spec.PersistentVolumeSource.LibStorage.VolumeName
	vol.ID = spec.PersistentVolume.Spec.PersistentVolumeSource.LibStorage.VolumeID

	return &lsVolume{
		volume:   vol,
		plugin:   p,
		mounter:  mount.New(),
		readOnly: spec.ReadOnly,
	}, nil
}

func (p *lsPlugin) initLibStorage() error {
	glog.V(4).Infof("LibStorage init with opts %v\n", p.opts)

	// parse raw opts
	opts := strings.Split(strings.TrimSpace(p.opts), ";")
	if len(opts) == 0 {
		glog.Error("LibStorage opts values are invalid")
		return fmt.Errorf("invalid libstorage options")
	}

	cfg := gofig.New()
	// parse each option
	for _, opt := range opts {
		parts := strings.Split(strings.TrimSpace(opt), "=")
		if len(parts) != 2 {
			continue
		}
		if !p.isValidOpt(parts[0]) {
			continue
		}

		if parts[0] == "service" {
			p.service = parts[1]
		}
		if parts[0] == "host" {
			cfg.Set("libstorage.host", parts[1])
		}
	}

	// init client
	ctx := lsctx.Background()
	ctx = ctx.WithValue(lsctx.ServiceKey, p.service)
	p.ctx = ctx
	p.cfg = cfg
	return nil
}

func (p *lsPlugin) getClient() (lstypes.Client, error) {
	if p.client == nil {
		client, err := lsclient.New(p.ctx, p.cfg)
		if err != nil {
			glog.Errorf("LibStorage client initialization failed: %s\n", err)
			return nil, err
		}
		p.client = client
		return p.client, nil
	}
	return p.client, nil
}

func (p *lsPlugin) isValidOpt(opt string) bool {
	for _, val := range lsValidOpts {
		if val == opt {
			return true
		}
	}
	return false
}

func (p *lsPlugin) getLibStorageSource(spec *volume.Spec) (*api.LibStorageVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.LibStorage != nil {
		return spec.Volume.LibStorage, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.LibStorage != nil {
		return spec.PersistentVolume.Spec.LibStorage, nil
	}

	return nil, fmt.Errorf("LibStorage is not found in spec")
}
