/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package examples_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/yaml"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	schedulerapilatest "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
)

func validateObject(obj runtime.Object) (errors []error) {
	switch t := obj.(type) {
	case *api.ReplicationController:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateReplicationController(t)
	case *api.ReplicationControllerList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Service:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateService(t)
	case *api.ServiceList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Pod:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePod(t)
	case *api.PodList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.PersistentVolume:
		errors = validation.ValidatePersistentVolume(t)
	case *api.PersistentVolumeClaim:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePersistentVolumeClaim(t)
	case *api.PodTemplate:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePodTemplate(t)
	case *api.Endpoints:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateEndpoints(t)
	case *api.Namespace:
		errors = validation.ValidateNamespace(t)
	case *api.Secret:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateSecret(t)
	case *api.LimitRange:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateLimitRange(t)
	case *api.ResourceQuota:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateResourceQuota(t)
	default:
		return []error{fmt.Errorf("no validation defined for %#v", obj)}
	}
	return errors
}

func walkJSONFiles(inDir string, fn func(name, path string, data []byte)) error {
	return filepath.Walk(inDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() && path != inDir {
			return filepath.SkipDir
		}

		file := filepath.Base(path)
		if ext := filepath.Ext(file); ext == ".json" || ext == ".yaml" {
			glog.Infof("Testing %s", path)
			data, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			name := strings.TrimSuffix(file, ext)

			if ext == ".yaml" {
				out, err := yaml.ToJSON(data)
				if err != nil {
					return fmt.Errorf("%s: %v", path, err)
				}
				data = out
			}

			fn(name, path, data)
		}
		return nil
	})
}

func TestExampleObjectSchemas(t *testing.T) {
	cases := map[string]map[string]runtime.Object{
		"../cmd/integration": {
			"v1-controller": &api.ReplicationController{},
		},
		"../examples/guestbook": {
			"frontend-controller":     &api.ReplicationController{},
			"redis-slave-controller":  &api.ReplicationController{},
			"redis-master-controller": &api.ReplicationController{},
			"frontend-service":        &api.Service{},
			"redis-master-service":    &api.Service{},
			"redis-slave-service":     &api.Service{},
		},
		"../examples/guestbook-go": {
			"guestbook-controller":    &api.ReplicationController{},
			"redis-slave-controller":  &api.ReplicationController{},
			"redis-master-controller": &api.ReplicationController{},
			"guestbook-service":       &api.Service{},
			"redis-master-service":    &api.Service{},
			"redis-slave-service":     &api.Service{},
		},
		"../docs/user-guide/walkthrough": {
			"pod-nginx":                 &api.Pod{},
			"pod-nginx-with-label":      &api.Pod{},
			"pod-redis":                 &api.Pod{},
			"pod-with-http-healthcheck": &api.Pod{},
			"service":                   &api.Service{},
			"replication-controller":    &api.ReplicationController{},
			"podtemplate":               &api.PodTemplate{},
		},
		"../docs/user-guide/update-demo": {
			"kitten-rc":   &api.ReplicationController{},
			"nautilus-rc": &api.ReplicationController{},
		},
		"../docs/user-guide/persistent-volumes/volumes": {
			"local-01": &api.PersistentVolume{},
			"local-02": &api.PersistentVolume{},
			"gce":      &api.PersistentVolume{},
			"nfs":      &api.PersistentVolume{},
		},
		"../docs/user-guide/persistent-volumes/claims": {
			"claim-01": &api.PersistentVolumeClaim{},
			"claim-02": &api.PersistentVolumeClaim{},
			"claim-03": &api.PersistentVolumeClaim{},
		},
		"../docs/user-guide/persistent-volumes/simpletest": {
			"namespace": &api.Namespace{},
			"pod":       &api.Pod{},
			"service":   &api.Service{},
		},
		"../examples/iscsi": {
			"iscsi": &api.Pod{},
		},
		"../examples/glusterfs": {
			"glusterfs-pod":       &api.Pod{},
			"glusterfs-endpoints": &api.Endpoints{},
			"glusterfs-service":   &api.Service{},
		},
		"../docs/user-guide/liveness": {
			"exec-liveness": &api.Pod{},
			"http-liveness": &api.Pod{},
		},
		"../docs/user-guide": {
			"multi-pod":   nil,
			"pod":         &api.Pod{},
			"replication": &api.ReplicationController{},
		},
		"../examples": {
			"scheduler-policy-config": &schedulerapi.Policy{},
		},
		"../examples/rbd/secret": {
			"ceph-secret": &api.Secret{},
		},
		"../examples/rbd": {
			"rbd":             &api.Pod{},
			"rbd-with-secret": &api.Pod{},
		},
		"../examples/cassandra": {
			"cassandra-controller": &api.ReplicationController{},
			"cassandra-service":    &api.Service{},
			"cassandra":            &api.Pod{},
		},
		"../examples/celery-rabbitmq": {
			"celery-controller":   &api.ReplicationController{},
			"flower-controller":   &api.ReplicationController{},
			"flower-service":      &api.Service{},
			"rabbitmq-controller": &api.ReplicationController{},
			"rabbitmq-service":    &api.Service{},
		},
		"../examples/cluster-dns": {
			"dns-backend-rc":      &api.ReplicationController{},
			"dns-backend-service": &api.Service{},
			"dns-frontend-pod":    &api.Pod{},
			"namespace-dev":       &api.Namespace{},
			"namespace-prod":      &api.Namespace{},
		},
		"../docs/user-guide/downward-api": {
			"dapi-pod": &api.Pod{},
		},
		"../docs/user-guide/downward-api/volume/": {
			"dapi-volume": &api.Pod{},
		},
		"../examples/elasticsearch": {
			"es-rc":           &api.ReplicationController{},
			"es-svc":          &api.Service{},
			"service-account": nil,
		},
		"../examples/explorer": {
			"pod": &api.Pod{},
		},
		"../examples/hazelcast": {
			"hazelcast-controller": &api.ReplicationController{},
			"hazelcast-service":    &api.Service{},
		},
		"../docs/admin/namespaces": {
			"namespace-dev":  &api.Namespace{},
			"namespace-prod": &api.Namespace{},
		},
		"../docs/admin/limitrange": {
			"invalid-pod": &api.Pod{},
			"limits":      &api.LimitRange{},
			"namespace":   &api.Namespace{},
			"valid-pod":   &api.Pod{},
		},
		"../docs/user-guide/logging-demo": {
			"synthetic_0_25lps": &api.Pod{},
			"synthetic_10lps":   &api.Pod{},
		},
		"../examples/meteor": {
			"meteor-controller": &api.ReplicationController{},
			"meteor-service":    &api.Service{},
			"mongo-pod":         &api.Pod{},
			"mongo-service":     &api.Service{},
		},
		"../examples/mysql-wordpress-pd": {
			"mysql-service":     &api.Service{},
			"mysql":             &api.Pod{},
			"wordpress-service": &api.Service{},
			"wordpress":         &api.Pod{},
		},
		"../examples/nfs": {
			"nfs-server-pod":     &api.Pod{},
			"nfs-server-service": &api.Service{},
			"nfs-web-pod":        &api.Pod{},
		},
		"../docs/user-guide/node-selection": {
			"pod": &api.Pod{},
		},
		"../examples/openshift-origin": {
			"openshift-origin-namespace": &api.Namespace{},
			"openshift-controller":       &api.ReplicationController{},
			"openshift-service":          &api.Service{},
			"etcd-controller":            &api.ReplicationController{},
			"etcd-service":               &api.Service{},
			"etcd-discovery-controller":  &api.ReplicationController{},
			"etcd-discovery-service":     &api.Service{},
			"secret":                     nil,
		},
		"../examples/phabricator": {
			"authenticator-controller": &api.ReplicationController{},
			"phabricator-controller":   &api.ReplicationController{},
			"phabricator-service":      &api.Service{},
		},
		"../examples/redis": {
			"redis-controller":          &api.ReplicationController{},
			"redis-master":              &api.Pod{},
			"redis-proxy":               &api.Pod{},
			"redis-sentinel-controller": &api.ReplicationController{},
			"redis-sentinel-service":    &api.Service{},
		},
		"../docs/user-guide/resourcequota": {
			"namespace": &api.Namespace{},
			"limits":    &api.LimitRange{},
			"quota":     &api.ResourceQuota{},
		},
		"../examples/rethinkdb": {
			"admin-pod":      &api.Pod{},
			"admin-service":  &api.Service{},
			"driver-service": &api.Service{},
			"rc":             &api.ReplicationController{},
		},
		"../docs/user-guide/secrets": {
			"secret-pod": &api.Pod{},
			"secret":     &api.Secret{},
		},
		"../examples/spark": {
			"spark-master-service":    &api.Service{},
			"spark-master":            &api.Pod{},
			"spark-worker-controller": &api.ReplicationController{},
			"spark-driver":            &api.Pod{},
		},
		"../examples/storm": {
			"storm-nimbus-service":    &api.Service{},
			"storm-nimbus":            &api.Pod{},
			"storm-worker-controller": &api.ReplicationController{},
			"zookeeper-service":       &api.Service{},
			"zookeeper":               &api.Pod{},
		},
		"../examples/cephfs/": {
			"cephfs":             &api.Pod{},
			"cephfs-with-secret": &api.Pod{},
		},
	}

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: true,
	})

	for path, expected := range cases {
		tested := 0
		err := walkJSONFiles(path, func(name, path string, data []byte) {
			expectedType, found := expected[name]
			if !found {
				t.Errorf("%s: %s does not have a test case defined", path, name)
				return
			}
			tested++
			if expectedType == nil {
				t.Logf("skipping : %s/%s\n", path, name)
				return
			}
			if name == "scheduler-policy-config" {
				if err := schedulerapilatest.Codec.DecodeInto(data, expectedType); err != nil {
					t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(data))
					return
				}
				//TODO: Add validate method for &schedulerapi.Policy
			} else {
				if err := latest.Codec.DecodeInto(data, expectedType); err != nil {
					t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(data))
					return
				}
				if errors := validateObject(expectedType); len(errors) > 0 {
					t.Errorf("%s did not validate correctly: %v", path, errors)
				}
			}
		})
		if err != nil {
			t.Errorf("Expected no error, Got %v", err)
		}
		if tested != len(expected) {
			t.Errorf("Directory %v: Expected %d examples, Got %d", path, len(expected), tested)
		}
	}
}

// This regex is tricky, but it works.  For future me, here is the decode:
//
// Flags: (?ms) = multiline match, allow . to match \n
// 1) Look for a line that starts with ``` (a markdown code block)
// 2) (?: ... ) = non-capturing group
// 3) (P<name>) = capture group as "name"
// 4) Look for #1 followed by either:
// 4a)    "yaml" followed by any word-characters followed by a newline (e.g. ```yamlfoo\n)
// 4b)    "any word-characters followed by a newline (e.g. ```json\n)
// 5) Look for either:
// 5a)    #4a followed by one or more characters (non-greedy)
// 5b)    #4b followed by { followed by one or more characters (non-greedy) followed by }
// 6) Look for #5 followed by a newline followed by ``` (end of the code block)
//
// This could probably be simplified, but is already too delicate.  Before any
// real changes, we should have a testscase that just tests this regex.
var sampleRegexp = regexp.MustCompile("(?ms)^```(?:(?P<type>yaml)\\w*\\n(?P<content>.+?)|\\w*\\n(?P<content>\\{.+?\\}))\\n^```")
var subsetRegexp = regexp.MustCompile("(?ms)\\.{3}")

func TestReadme(t *testing.T) {
	paths := []struct {
		file         string
		expectedType []runtime.Object
	}{
		{"../README.md", []runtime.Object{&api.Pod{}}},
		{"../docs/user-guide/walkthrough/README.md", []runtime.Object{&api.Pod{}}},
		{"../examples/iscsi/README.md", []runtime.Object{&api.Pod{}}},
		{"../docs/user-guide/simple-yaml.md", []runtime.Object{&api.Pod{}, &api.ReplicationController{}}},
	}

	for _, path := range paths {
		data, err := ioutil.ReadFile(path.file)
		if err != nil {
			t.Errorf("Unable to read file %s: %v", path, err)
			continue
		}

		matches := sampleRegexp.FindAllStringSubmatch(string(data), -1)
		if matches == nil {
			continue
		}
		ix := 0
		for _, match := range matches {
			var content, subtype string
			for i, name := range sampleRegexp.SubexpNames() {
				if name == "type" {
					subtype = match[i]
				}
				if name == "content" && match[i] != "" {
					content = match[i]
				}
			}
			if subtype == "yaml" && subsetRegexp.FindString(content) != "" {
				t.Logf("skipping (%s): \n%s", subtype, content)
				continue
			}

			var expectedType runtime.Object
			if len(path.expectedType) == 1 {
				expectedType = path.expectedType[0]
			} else {
				expectedType = path.expectedType[ix]
				ix++
			}
			json, err := yaml.ToJSON([]byte(content))
			if err != nil {
				t.Errorf("%s could not be converted to JSON: %v\n%s", path, err, string(content))
			}
			if err := latest.Codec.DecodeInto(json, expectedType); err != nil {
				t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(content))
				continue
			}
			if errors := validateObject(expectedType); len(errors) > 0 {
				t.Errorf("%s did not validate correctly: %v", path, errors)
			}
			_, err = latest.Codec.Encode(expectedType)
			if err != nil {
				t.Errorf("Could not encode object: %v", err)
				continue
			}
		}
	}
}
