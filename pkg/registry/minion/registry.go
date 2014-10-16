/*
Copyright 2014 Google Inc. All rights reserved.

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

package minion

import "github.com/GoogleCloudPlatform/kubernetes/pkg/api"

// MinionRegistry is an interface for things that know how to store minions.
type Registry interface {
	ListMinions(ctx api.Context) (*api.MinionList, error)
	CreateMinion(ctx api.Context, minion *api.Minion) error
	GetMinion(ctx api.Context, minionID string) (*api.Minion, error)
	DeleteMinion(ctx api.Context, minionID string) error
}
