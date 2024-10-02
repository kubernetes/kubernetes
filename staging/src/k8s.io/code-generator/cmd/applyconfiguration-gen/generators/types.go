/*
Copyright 2021 The Kubernetes Authors.

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

package generators

import "k8s.io/gengo/v2/types"

var (
	applyConfiguration   = types.Ref("k8s.io/apimachinery/pkg/runtime", "ApplyConfiguration")
	groupVersionKind     = types.Ref("k8s.io/apimachinery/pkg/runtime/schema", "GroupVersionKind")
	typeMeta             = types.Ref("k8s.io/apimachinery/pkg/apis/meta/v1", "TypeMeta")
	objectMeta           = types.Ref("k8s.io/apimachinery/pkg/apis/meta/v1", "ObjectMeta")
	rawExtension         = types.Ref("k8s.io/apimachinery/pkg/runtime", "RawExtension")
	unknown              = types.Ref("k8s.io/apimachinery/pkg/runtime", "Unknown")
	extractInto          = types.Ref("k8s.io/apimachinery/pkg/util/managedfields", "ExtractInto")
	runtimeScheme        = types.Ref("k8s.io/apimachinery/pkg/runtime", "Scheme")
	smdNewParser         = types.Ref("sigs.k8s.io/structured-merge-diff/v4/typed", "NewParser")
	smdParser            = types.Ref("sigs.k8s.io/structured-merge-diff/v4/typed", "Parser")
	testingTypeConverter = types.Ref("k8s.io/client-go/testing", "TypeConverter")
	yamlObject           = types.Ref("sigs.k8s.io/structured-merge-diff/v4/typed", "YAMLObject")
	yamlUnmarshal        = types.Ref("gopkg.in/yaml.v2", "Unmarshal")
)
