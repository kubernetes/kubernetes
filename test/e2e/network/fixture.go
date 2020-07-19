/*
Copyright 2019 The Kubernetes Authors.

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

package network

import (
	"context"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// TestFixture is a simple helper class to avoid too much boilerplate in tests
type TestFixture struct {
	ServiceName string
	Namespace   string
	Client      clientset.Interface

	TestID string
	Labels map[string]string

	rcs      map[string]bool
	services map[string]bool
	Name     string
	Image    string
}

// NewServerTest creates a new TestFixture for the tests.
func NewServerTest(client clientset.Interface, namespace string, serviceName string) *TestFixture {
	t := &TestFixture{}
	t.Client = client
	t.Namespace = namespace
	t.ServiceName = serviceName
	t.TestID = t.ServiceName + "-" + string(uuid.NewUUID())
	t.Labels = map[string]string{
		"testid": t.TestID,
	}

	t.rcs = make(map[string]bool)
	t.services = make(map[string]bool)

	t.Name = "webserver"
	t.Image = imageutils.GetE2EImage(imageutils.Agnhost)

	return t
}

// BuildServiceSpec builds default config for a service (which can then be changed)
func (t *TestFixture) BuildServiceSpec() *v1.Service {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      t.ServiceName,
			Namespace: t.Namespace,
		},
		Spec: v1.ServiceSpec{
			Selector: t.Labels,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	return service
}

// CreateRC creates a replication controller and records it for cleanup.
func (t *TestFixture) CreateRC(rc *v1.ReplicationController) (*v1.ReplicationController, error) {
	rc, err := t.Client.CoreV1().ReplicationControllers(t.Namespace).Create(context.TODO(), rc, metav1.CreateOptions{})
	if err == nil {
		t.rcs[rc.Name] = true
	}
	return rc, err
}

// CreateService creates a service, and record it for cleanup
func (t *TestFixture) CreateService(service *v1.Service) (*v1.Service, error) {
	result, err := t.Client.CoreV1().Services(t.Namespace).Create(context.TODO(), service, metav1.CreateOptions{})
	if err == nil {
		t.services[service.Name] = true
	}
	return result, err
}

// DeleteService deletes a service, and remove it from the cleanup list
func (t *TestFixture) DeleteService(serviceName string) error {
	err := t.Client.CoreV1().Services(t.Namespace).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
	if err == nil {
		delete(t.services, serviceName)
	}
	return err
}

// Cleanup cleans all ReplicationControllers and Services which this object holds.
func (t *TestFixture) Cleanup() []error {
	var errs []error
	for rcName := range t.rcs {
		ginkgo.By("stopping RC " + rcName + " in namespace " + t.Namespace)
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// First, resize the RC to 0.
			old, err := t.Client.CoreV1().ReplicationControllers(t.Namespace).Get(context.TODO(), rcName, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					return nil
				}
				return err
			}
			x := int32(0)
			old.Spec.Replicas = &x
			if _, err := t.Client.CoreV1().ReplicationControllers(t.Namespace).Update(context.TODO(), old, metav1.UpdateOptions{}); err != nil {
				if apierrors.IsNotFound(err) {
					return nil
				}
				return err
			}
			return nil
		})
		if err != nil {
			errs = append(errs, err)
		}
		// TODO(mikedanese): Wait.
		// Then, delete the RC altogether.
		if err := t.Client.CoreV1().ReplicationControllers(t.Namespace).Delete(context.TODO(), rcName, metav1.DeleteOptions{}); err != nil {
			if !apierrors.IsNotFound(err) {
				errs = append(errs, err)
			}
		}
	}

	for serviceName := range t.services {
		ginkgo.By("deleting service " + serviceName + " in namespace " + t.Namespace)
		err := t.Client.CoreV1().Services(t.Namespace).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
		if err != nil {
			if !apierrors.IsNotFound(err) {
				errs = append(errs, err)
			}
		}
	}

	return errs
}
