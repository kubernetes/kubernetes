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

package log

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/pborman/uuid"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/apiserver/pkg/audit"
)

// fakeEvent creates one fake event
func fakeEvent(namespace, username string) *auditinternal.Event {
	return &auditinternal.Event{
		AuditID: types.UID(uuid.NewRandom().String()),
		Level:   auditinternal.LevelMetadata,
		ObjectRef: &auditinternal.ObjectReference{
			Resource:    "foo",
			APIVersion:  "v1",
			Subresource: "bar",
			Namespace:   namespace,
		},
		User: auditinternal.UserInfo{
			Username: username,
			Groups: []string{
				"system:masters",
				"system:authenticated",
			},
		},
	}
}

func TestLogEvents(t *testing.T) {
	// create 1000 audit events
	events := map[types.UID]*auditinternal.Event{}
	for i := 0; i < 10; i++ {
		for j := 0; j < 100; j++ {
			namespace := "" // cluster scope
			if i > 0 {
				namespace = "namespace-" + strconv.Itoa(i)
			}
			username := "user-" + strconv.Itoa(j)
			ev := fakeEvent(namespace, username)
			events[ev.AuditID] = ev
		}
	}

	// log events
	dir, err := ioutil.TempDir("", "k8s-test-audit-event-logger")
	if err != nil {
		t.Fatalf("Unable to create the test directory %q: %v", dir, err)
	}
	path := filepath.Join(dir, "{namespace}", "{username}", "audit.log")
	logger, err := NewEventLogger(path, 0, 0, 0)
	backend := NewBackend(logger, FormatJson, auditv1beta1.SchemeGroupVersion)

	var wg sync.WaitGroup
	ch := make(chan struct{}, 20)
	for _, ev := range events {
		ch <- struct{}{}
		wg.Add(1)
		go func(ev *auditinternal.Event) {
			backend.ProcessEvents(ev)
			<-ch
			wg.Done()
		}(ev)
	}
	wg.Wait()
	backend.Shutdown()

	// check events
	for i := 0; i < 10; i++ {
		for j := 0; j < 100; j++ {
			namespace := "" // cluster scope
			nsInPath := "none"
			if i > 0 {
				namespace = "namespace-" + strconv.Itoa(i)
				nsInPath = namespace
			}
			username := "user-" + strconv.Itoa(j)
			filepath := strings.Replace(path, "{namespace}", nsInPath, -1)
			filepath = strings.Replace(filepath, "{username}", username, -1)
			bytes, err := ioutil.ReadFile(filepath)
			if err != nil {
				t.Errorf("failed to read audit events from file %q: %+v", filepath, err)
				continue
			}
			result := &auditinternal.Event{}
			decoder := audit.Codecs.UniversalDecoder(auditv1beta1.SchemeGroupVersion)
			if err := runtime.DecodeInto(decoder, bytes, result); err != nil {
				t.Errorf("failed decoding buf: %s", bytes)
				continue
			}
			if result.User.Username != username {
				t.Errorf("Unexpected username: %q in file: %s, expected username is %q", result.User.Username, filepath, username)
			}
			if result.ObjectRef.Namespace != namespace {
				t.Errorf("Unexpected namespace: %q in file: %s, expected namespace is %q", result.ObjectRef.Namespace, filepath, namespace)
			}
			if !reflect.DeepEqual(events[result.AuditID], result) {
				t.Errorf("The result event should be the same with the original one, \noriginal: \n%#v\n result: \n%#v", events[result.AuditID], result)
			}

		}
	}

	// clean up
	if err := os.RemoveAll(dir); err != nil {
		t.Errorf("Unable to clean up test directory %q: %v", dir, err)
	}
}
