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

package config

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	api "k8s.io/kubernetes/pkg/apis/core"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	utilio "k8s.io/utils/io"
)

type sourceURL struct {
	url         string
	header      http.Header
	nodeName    types.NodeName
	updates     chan<- sourceUpdate
	data        []byte
	failureLogs int
	client      *http.Client
}

// NewSourceURL specifies the URL where to read the Pod configuration from, then
// watches it for changes. The watch stops when ctx is canceled.
func NewSourceURL(ctx context.Context, url string, header http.Header, nodeName types.NodeName, period time.Duration, updates chan<- sourceUpdate) {
	config := &sourceURL{
		url:      url,
		header:   header,
		nodeName: nodeName,
		updates:  updates,
		data:     nil,
		// Timing out requests leads to retries. This client is only used to
		// read the manifest URL passed to kubelet.
		client: &http.Client{Timeout: 10 * time.Second},
	}
	klog.FromContext(ctx).V(1).Info("Watching URL", "URL", url)
	go wait.UntilWithContext(ctx, config.run, period)
}

func (s *sourceURL) run(ctx context.Context) {
	logger := klog.FromContext(ctx)
	if err := s.extractFromURL(ctx); err != nil {
		// Don't log this multiple times per minute. The first few entries should be
		// enough to get the point across.
		if s.failureLogs < 3 {
			logger.Info("Failed to read pods from URL", "err", err)
		} else if s.failureLogs == 3 {
			logger.Info("Failed to read pods from URL. Dropping verbosity of this message to V(4)", "err", err)
		} else {
			logger.V(4).Info("Failed to read pods from URL", "err", err)
		}
		s.failureLogs++
	} else {
		if s.failureLogs > 0 {
			logger.Info("Successfully read pods from URL")
			s.failureLogs = 0
		}
	}
}

func (s *sourceURL) applyDefaults(logger klog.Logger, pod *api.Pod) error {
	return applyDefaults(logger, pod, s.url, false, s.nodeName)
}

// sendUpdate delivers update unless ctx is canceled first, so that a stopped
// source never blocks forever on a channel nobody reads anymore.
func (s *sourceURL) sendUpdate(ctx context.Context, update sourceUpdate) {
	select {
	case s.updates <- update:
	case <-ctx.Done():
	}
}

func (s *sourceURL) extractFromURL(ctx context.Context) error {
	logger := klog.FromContext(ctx)
	req, err := http.NewRequestWithContext(ctx, "GET", s.url, nil)
	if err != nil {
		return err
	}
	req.Header = s.header
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := utilio.ReadAtMost(resp.Body, maxConfigLength)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%v: %v", s.url, resp.Status)
	}
	if len(data) == 0 {
		// Emit an update with an empty PodList to allow HTTPSource to be marked as seen
		s.sendUpdate(ctx, sourceUpdate{Pods: []*v1.Pod{}})
		return fmt.Errorf("zero-length data received from %v", s.url)
	}
	// Short circuit if the data has not changed since the last time it was read.
	if bytes.Equal(data, s.data) {
		return nil
	}
	s.data = data

	// First try as it is a single pod.
	parsed, pod, singlePodErr := tryDecodeSinglePod(logger, data, s.applyDefaults)
	if parsed {
		if singlePodErr != nil {
			// It parsed but could not be used.
			return singlePodErr
		}
		s.sendUpdate(ctx, sourceUpdate{Pods: []*v1.Pod{pod}})
		return nil
	}

	// That didn't work, so try a list of pods.
	parsed, podList, multiPodErr := tryDecodePodList(logger, data, s.applyDefaults)
	if parsed {
		if multiPodErr != nil {
			// It parsed but could not be used.
			return multiPodErr
		}
		pods := make([]*v1.Pod, 0, len(podList.Items))
		for i := range podList.Items {
			pods = append(pods, &podList.Items[i])
		}
		s.sendUpdate(ctx, sourceUpdate{Pods: pods})
		return nil
	}

	return fmt.Errorf("%v: received '%v', but couldn't parse as "+
		"single (%v) or multiple pods (%v)",
		s.url, string(data), singlePodErr, multiPodErr)
}
