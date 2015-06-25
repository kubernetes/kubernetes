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

package kubectl

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

const (
	Interval = time.Second * 1
	Timeout  = time.Minute * 5
)

// A Reaper handles terminating an object as gracefully as possible.
// timeout is how long we'll wait for the termination to be successful
// gracePeriod is time given to an API object for it to delete itself cleanly (e.g. pod shutdown)
type Reaper interface {
	Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) (string, error)
}

type NoSuchReaperError struct {
	kind string
}

func (n *NoSuchReaperError) Error() string {
	return fmt.Sprintf("no reaper has been implemented for %q", n.kind)
}

func IsNoSuchReaperError(err error) bool {
	_, ok := err.(*NoSuchReaperError)
	return ok
}

func ReaperFor(kind string, c client.Interface) (Reaper, error) {
	switch kind {
	case "ReplicationController":
		return &ReplicationControllerReaper{c, Interval, Timeout}, nil
	case "Pod":
		return &PodReaper{c}, nil
	case "Service":
		return &ServiceReaper{c}, nil
	}
	return nil, &NoSuchReaperError{kind}
}

func ReaperForReplicationController(c client.Interface, timeout time.Duration) (Reaper, error) {
	return &ReplicationControllerReaper{c, Interval, timeout}, nil
}

type ReplicationControllerReaper struct {
	client.Interface
	pollInterval, timeout time.Duration
}
type PodReaper struct {
	client.Interface
}
type ServiceReaper struct {
	client.Interface
}

type objInterface interface {
	Delete(name string) error
	Get(name string) (meta.Interface, error)
}

func (reaper *ReplicationControllerReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) (string, error) {
	rc := reaper.ReplicationControllers(namespace)
	scaler, err := ScalerFor("ReplicationController", NewScalerClient(*reaper))
	if err != nil {
		return "", err
	}
	if timeout == 0 {
		rc, err := rc.Get(name)
		if err != nil {
			return "", err
		}
		timeout = Timeout + time.Duration(10*rc.Spec.Replicas)*time.Second
	}
	retry := NewRetryParams(reaper.pollInterval, reaper.timeout)
	waitForReplicas := NewRetryParams(reaper.pollInterval, timeout)
	if err = scaler.Scale(namespace, name, 0, nil, retry, waitForReplicas); err != nil {
		return "", err
	}
	if err := rc.Delete(name); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s stopped", name), nil
}

func (reaper *PodReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) (string, error) {
	pods := reaper.Pods(namespace)
	_, err := pods.Get(name)
	if err != nil {
		return "", err
	}
	if err := pods.Delete(name, gracePeriod); err != nil {
		return "", err
	}

	return fmt.Sprintf("%s stopped", name), nil
}

func (reaper *ServiceReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) (string, error) {
	services := reaper.Services(namespace)
	_, err := services.Get(name)
	if err != nil {
		return "", err
	}
	if err := services.Delete(name); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s stopped", name), nil
}
