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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/akutz/gofig"
	"github.com/golang/glog"

	lsctx "github.com/emccode/libstorage/api/context"
	lstypes "github.com/emccode/libstorage/api/types"
	lsclient "github.com/emccode/libstorage/client"
)

const (
	lsPluginName = "kubernetes.io/libstorage"
)

type lsPlugin struct {
	host volume.VolumeHost

	lsHost  string
	service string
	client  lstypes.Client
	ctx     lstypes.Context
	cfg     gofig.Config
}

// Helper methods

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

func (p *lsPlugin) initLibStorage(lsHost, service string) error {
	glog.V(4).Infoln("LibStorage init")
	cfg := gofig.New()
	ctx := lsctx.Background()
	ctx = ctx.WithValue(lsctx.ServiceKey, service)
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
