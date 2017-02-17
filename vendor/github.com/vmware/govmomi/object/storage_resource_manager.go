/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type StorageResourceManager struct {
	Common
}

func NewStorageResourceManager(c *vim25.Client) *StorageResourceManager {
	sr := StorageResourceManager{
		Common: NewCommon(c, *c.ServiceContent.StorageResourceManager),
	}

	return &sr
}

func (sr StorageResourceManager) ApplyStorageDrsRecommendation(ctx context.Context, key []string) (*Task, error) {
	req := types.ApplyStorageDrsRecommendation_Task{
		This: sr.Reference(),
		Key:  key,
	}

	res, err := methods.ApplyStorageDrsRecommendation_Task(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(sr.c, res.Returnval), nil
}

func (sr StorageResourceManager) ApplyStorageDrsRecommendationToPod(ctx context.Context, pod *StoragePod, key string) (*Task, error) {
	req := types.ApplyStorageDrsRecommendationToPod_Task{
		This: sr.Reference(),
		Key:  key,
	}

	if pod != nil {
		req.Pod = pod.Reference()
	}

	res, err := methods.ApplyStorageDrsRecommendationToPod_Task(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(sr.c, res.Returnval), nil
}

func (sr StorageResourceManager) CancelStorageDrsRecommendation(ctx context.Context, key []string) error {
	req := types.CancelStorageDrsRecommendation{
		This: sr.Reference(),
		Key:  key,
	}

	_, err := methods.CancelStorageDrsRecommendation(ctx, sr.c, &req)

	return err
}

func (sr StorageResourceManager) ConfigureDatastoreIORM(ctx context.Context, datastore *Datastore, spec types.StorageIORMConfigSpec, key string) (*Task, error) {
	req := types.ConfigureDatastoreIORM_Task{
		This: sr.Reference(),
		Spec: spec,
	}

	if datastore != nil {
		req.Datastore = datastore.Reference()
	}

	res, err := methods.ConfigureDatastoreIORM_Task(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(sr.c, res.Returnval), nil
}

func (sr StorageResourceManager) ConfigureStorageDrsForPod(ctx context.Context, pod *StoragePod, spec types.StorageDrsConfigSpec, modify bool) (*Task, error) {
	req := types.ConfigureStorageDrsForPod_Task{
		This:   sr.Reference(),
		Spec:   spec,
		Modify: modify,
	}

	if pod != nil {
		req.Pod = pod.Reference()
	}

	res, err := methods.ConfigureStorageDrsForPod_Task(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(sr.c, res.Returnval), nil
}

func (sr StorageResourceManager) QueryDatastorePerformanceSummary(ctx context.Context, datastore *Datastore) ([]types.StoragePerformanceSummary, error) {
	req := types.QueryDatastorePerformanceSummary{
		This: sr.Reference(),
	}

	if datastore != nil {
		req.Datastore = datastore.Reference()
	}

	res, err := methods.QueryDatastorePerformanceSummary(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (sr StorageResourceManager) QueryIORMConfigOption(ctx context.Context, host *HostSystem) (*types.StorageIORMConfigOption, error) {
	req := types.QueryIORMConfigOption{
		This: sr.Reference(),
	}

	if host != nil {
		req.Host = host.Reference()
	}

	res, err := methods.QueryIORMConfigOption(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (sr StorageResourceManager) RecommendDatastores(ctx context.Context, storageSpec types.StoragePlacementSpec) (*types.StoragePlacementResult, error) {
	req := types.RecommendDatastores{
		This:        sr.Reference(),
		StorageSpec: storageSpec,
	}

	res, err := methods.RecommendDatastores(ctx, sr.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (sr StorageResourceManager) RefreshStorageDrsRecommendation(ctx context.Context, pod *StoragePod) error {
	req := types.RefreshStorageDrsRecommendation{
		This: sr.Reference(),
	}

	if pod != nil {
		req.Pod = pod.Reference()
	}

	_, err := methods.RefreshStorageDrsRecommendation(ctx, sr.c, &req)

	return err
}
