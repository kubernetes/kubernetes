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

package scheme

import (
	runtime "k8s.io/apimachinery/pkg/runtime"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	federationinternalversion "k8s.io/kubernetes/federation/apis/federation"
	coreinternalversion "k8s.io/kubernetes/pkg/api"
	autoscalinginternalversion "k8s.io/kubernetes/pkg/apis/autoscaling"
	batchinternalversion "k8s.io/kubernetes/pkg/apis/batch"
	extensionsinternalversion "k8s.io/kubernetes/pkg/apis/extensions"
)

var Scheme = runtime.NewScheme()
var Codecs = serializer.NewCodecFactory(Scheme)
var ParameterCodec = runtime.NewParameterCodec(Scheme)

func init() {
	coreinternalversion.AddToScheme(Scheme)
	autoscalinginternalversion.AddToScheme(Scheme)
	batchinternalversion.AddToScheme(Scheme)
	extensionsinternalversion.AddToScheme(Scheme)
	federationinternalversion.AddToScheme(Scheme)

}
