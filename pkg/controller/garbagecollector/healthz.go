/*
Copyright 2018 The Kubernetes Authors.

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

package garbagecollector

import (
	"errors"
	"net/http"

	"k8s.io/apiserver/pkg/server/healthz"
)

func NewHealthzHandlers(controller *GarbageCollector) []healthz.HealthzChecker {
	return []healthz.HealthzChecker{&cacheSyncedCheck{controller}}
}

type cacheSyncedCheck struct {
	controller *GarbageCollector
}

func (c cacheSyncedCheck) Name() string {
	return "cachesSynced"
}

func (c *cacheSyncedCheck) Check(r *http.Request) error {
	if !c.controller.IsSynced() {
		return errors.New("not all caches have been synced for garbage collector controller")
	}
	return nil
}
