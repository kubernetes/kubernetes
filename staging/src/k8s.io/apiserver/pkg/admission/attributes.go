/*
Copyright 2014 The Kubernetes Authors.

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

package admission

import (
	"fmt"
	"strings"
	"sync"

	"github.com/kcp-dev/logicalcluster/v3"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/user"
)

type attributesRecord struct {
	kind        schema.GroupVersionKind
	namespace   string
	name        string
	resource    schema.GroupVersionResource
	subresource string
	operation   Operation
	options     runtime.Object
	dryRun      bool
	object      runtime.Object
	oldObject   runtime.Object
	userInfo    user.Info

	cluster logicalcluster.Name

	// other elements are always accessed in single goroutine.
	// But ValidatingAdmissionWebhook add annotations concurrently.
	annotations     map[string]annotation
	annotationsLock sync.RWMutex

	reinvocationContext ReinvocationContext
}

type annotation struct {
	level auditinternal.Level
	value string
}

func NewAttributesRecord(object runtime.Object, oldObject runtime.Object, kind schema.GroupVersionKind, namespace, name string, resource schema.GroupVersionResource, subresource string, operation Operation, operationOptions runtime.Object, dryRun bool, userInfo user.Info) Attributes {
	return &attributesRecord{
		kind:                kind,
		namespace:           namespace,
		name:                name,
		resource:            resource,
		subresource:         subresource,
		operation:           operation,
		options:             operationOptions,
		dryRun:              dryRun,
		object:              object,
		oldObject:           oldObject,
		userInfo:            userInfo,
		reinvocationContext: &reinvocationContext{},
	}
}

func (record *attributesRecord) GetKind() schema.GroupVersionKind {
	return record.kind
}

func (record *attributesRecord) GetNamespace() string {
	return record.namespace
}

func (record *attributesRecord) GetName() string {
	return record.name
}

func (record *attributesRecord) GetResource() schema.GroupVersionResource {
	return record.resource
}

func (record *attributesRecord) GetSubresource() string {
	return record.subresource
}

func (record *attributesRecord) GetOperation() Operation {
	return record.operation
}

func (record *attributesRecord) GetOperationOptions() runtime.Object {
	return record.options
}

func (record *attributesRecord) IsDryRun() bool {
	return record.dryRun
}

func (record *attributesRecord) GetObject() runtime.Object {
	return record.object
}

func (record *attributesRecord) GetOldObject() runtime.Object {
	return record.oldObject
}

func (record *attributesRecord) GetUserInfo() user.Info {
	return record.userInfo
}

func (record *attributesRecord) SetCluster(cluster logicalcluster.Name) {
	record.cluster = cluster
}

func (record *attributesRecord) GetCluster() logicalcluster.Name {
	return record.cluster
}

// getAnnotations implements privateAnnotationsGetter.It's a private method used
// by WithAudit decorator.
func (record *attributesRecord) getAnnotations(maxLevel auditinternal.Level) map[string]string {
	record.annotationsLock.RLock()
	defer record.annotationsLock.RUnlock()

	if record.annotations == nil {
		return nil
	}
	cp := make(map[string]string, len(record.annotations))
	for key, value := range record.annotations {
		if value.level.Less(maxLevel) || value.level == maxLevel {
			cp[key] = value.value
		}
	}
	return cp
}

// AddAnnotation adds an annotation to attributesRecord with Metadata audit level
func (record *attributesRecord) AddAnnotation(key, value string) error {
	return record.AddAnnotationWithLevel(key, value, auditinternal.LevelMetadata)
}

func (record *attributesRecord) AddAnnotationWithLevel(key, value string, level auditinternal.Level) error {
	if err := checkKeyFormat(key); err != nil {
		return err
	}
	if level.Less(auditinternal.LevelMetadata) {
		return fmt.Errorf("admission annotations are not allowed to be set at audit level lower than Metadata, key: %q, level: %s", key, level)
	}
	record.annotationsLock.Lock()
	defer record.annotationsLock.Unlock()

	if record.annotations == nil {
		record.annotations = make(map[string]annotation)
	}
	annotation := annotation{level: level, value: value}
	if v, ok := record.annotations[key]; ok && v != annotation {
		return fmt.Errorf("admission annotations are not allowd to be overwritten, key:%q, old value: %v, new value: %v", key, record.annotations[key], annotation)
	}
	record.annotations[key] = annotation
	return nil
}

func (record *attributesRecord) GetReinvocationContext() ReinvocationContext {
	return record.reinvocationContext
}

type reinvocationContext struct {
	// isReinvoke is true when admission plugins are being reinvoked
	isReinvoke bool
	// reinvokeRequested is true when an admission plugin requested a re-invocation of the chain
	reinvokeRequested bool
	// values stores reinvoke context values per plugin.
	values map[string]interface{}
}

func (rc *reinvocationContext) IsReinvoke() bool {
	return rc.isReinvoke
}

func (rc *reinvocationContext) SetIsReinvoke() {
	rc.isReinvoke = true
}

func (rc *reinvocationContext) ShouldReinvoke() bool {
	return rc.reinvokeRequested
}

func (rc *reinvocationContext) SetShouldReinvoke() {
	rc.reinvokeRequested = true
}

func (rc *reinvocationContext) SetValue(plugin string, v interface{}) {
	if rc.values == nil {
		rc.values = map[string]interface{}{}
	}
	rc.values[plugin] = v
}

func (rc *reinvocationContext) Value(plugin string) interface{} {
	return rc.values[plugin]
}

func checkKeyFormat(key string) error {
	parts := strings.Split(key, "/")
	if len(parts) != 2 {
		return fmt.Errorf("annotation key has invalid format, the right format is a DNS subdomain prefix and '/' and key name. (e.g. 'podsecuritypolicy.admission.k8s.io/admit-policy')")
	}
	if msgs := validation.IsQualifiedName(key); len(msgs) != 0 {
		return fmt.Errorf("annotation key has invalid format %s. A qualified name like 'podsecuritypolicy.admission.k8s.io/admit-policy' is required.", strings.Join(msgs, ","))
	}
	return nil
}
