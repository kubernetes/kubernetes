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

package apps

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type GroupKindVisitor interface {
	VisitDaemonSet(kind GroupKindElement)
	VisitDeployment(kind GroupKindElement)
	VisitJob(kind GroupKindElement)
	VisitPod(kind GroupKindElement)
	VisitReplicaSet(kind GroupKindElement)
	VisitReplicationController(kind GroupKindElement)
	VisitStatefulSet(kind GroupKindElement)
}

var _ GroupKindVisitor = &DefaultGroupKindVisitor{}

type DefaultGroupKindVisitor struct{}

func (*DefaultGroupKindVisitor) VisitDaemonSet(kind GroupKindElement)             {}
func (*DefaultGroupKindVisitor) VisitDeployment(kind GroupKindElement)            {}
func (*DefaultGroupKindVisitor) VisitJob(kind GroupKindElement)                   {}
func (*DefaultGroupKindVisitor) VisitPod(kind GroupKindElement)                   {}
func (*DefaultGroupKindVisitor) VisitReplicaSet(kind GroupKindElement)            {}
func (*DefaultGroupKindVisitor) VisitReplicationController(kind GroupKindElement) {}
func (*DefaultGroupKindVisitor) VisitStatefulSet(kind GroupKindElement)           {}

type GroupKindElement schema.GroupKind

func (kind GroupKindElement) GroupMatch(groups ...string) bool {
	for _, g := range groups {
		if kind.Group == g {
			return true
		}
	}
	return false
}

func (kind GroupKindElement) Accept(visitor GroupKindVisitor) error {
	if kind.GroupMatch("apps", "extensions") && kind.Kind == "DaemonSet" {
		visitor.VisitDaemonSet(kind)
		return nil
	}
	if kind.GroupMatch("apps", "extensions") && kind.Kind == "Deployment" {
		visitor.VisitDeployment(kind)
		return nil
	}
	if kind.GroupMatch("apps", "extensions") && kind.Kind == "Job" {
		visitor.VisitDeployment(kind)
		return nil
	}
	if kind.GroupMatch("", "core") && kind.Kind == "Pod" {
		visitor.VisitPod(kind)
		return nil
	}
	if kind.GroupMatch("apps", "extensions") && kind.Kind == "ReplicaSet" {
		visitor.VisitReplicationController(kind)
		return nil
	}
	if kind.GroupMatch("", "core") && kind.Kind == "ReplicationController" {
		visitor.VisitReplicationController(kind)
		return nil
	}
	if kind.GroupMatch("apps") && kind.Kind == "StatefulSet" {
		visitor.VisitStatefulSet(kind)
		return nil
	}
	return fmt.Errorf("no visitor method exists for %v", kind)
}
