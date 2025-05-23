/*
Copyright 2025 The Kubernetes Authors.

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

package imagepullprogress

import (
	"context"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/imagepullprogress"
)

type Options struct {
}

type ImagePullProgresser interface {
	ImagePullProgress(ctx context.Context, name string, uid types.UID, progresses chan<- imagepullprogress.Progress) error
}

func ServeImagePullProgressed(w http.ResponseWriter, req *http.Request, imagePullProgresser ImagePullProgresser, podName string, uid types.UID, idleTimeout time.Duration, supportedProtocols []string) {
	err := handleHTTPStreams(req, w, imagePullProgresser, podName, uid, supportedProtocols, idleTimeout)
	if err != nil {
		runtime.HandleError(err)
		return
	}
}
