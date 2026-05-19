/*
Copyright The Kubernetes Authors.

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

package flagz

import (
	"fmt"
	"net/http"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	v1alpha1 "k8s.io/apiserver/pkg/server/flagz/api/v1alpha1"
	"k8s.io/apiserver/pkg/server/flagz/negotiate"
)

const (
	DefaultFlagzPath = "/flagz"
	Kind             = "Flagz"
	GroupName        = "config.k8s.io"
	Version          = "v1alpha1"
)

type mux interface {
	Handle(path string, handler http.Handler)
}

// Install installs the flagz endpoint to the given mux.
func Install(m mux, componentName string, flagReader Reader, opts ...Option) {
	reg := &registry{
		reader:                flagReader,
		deprecatedVersionsMap: map[string]bool{},
	}
	for _, opt := range opts {
		opt(reg)
	}

	scheme := runtime.NewScheme()
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	codecFactory := serializer.NewCodecFactory(
		scheme,
		serializer.WithSerializer(func(_ runtime.ObjectCreater, _ runtime.ObjectTyper) runtime.SerializerInfo {
			textSerializer := flagzTextSerializer{componentName, reg.reader}
			return runtime.SerializerInfo{
				MediaType:        "text/plain",
				MediaTypeType:    "text",
				MediaTypeSubType: "plain",
				EncodesAsText:    true,
				Serializer:       textSerializer,
				PrettySerializer: textSerializer,
			}
		}),
	)
	m.Handle(DefaultFlagzPath, handleFlagz(componentName, reg, codecFactory, negotiate.FlagzEndpointRestrictions{}))
}

func handleFlagz(componentName string, reg *registry, serializer runtime.NegotiatedSerializer, restrictions negotiate.FlagzEndpointRestrictions) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		obj := flagz(componentName, reg.reader)
		acceptHeader := r.Header.Get("Accept")
		if strings.TrimSpace(acceptHeader) == "" {
			writePlainTextResponse(obj, serializer, w, reg)
			return
		}

		mediaType, serializerInfo, err := negotiation.NegotiateOutputMediaType(r, serializer, restrictions)
		if err != nil {
			utilruntime.HandleError(err)
			responsewriters.ErrorNegotiated(
				err,
				serializer,
				schema.GroupVersion{},
				w,
				r,
			)
			return
		}

		var targetGV schema.GroupVersion
		switch serializerInfo.MediaType {
		case "application/json":
			if mediaType.Convert == nil {
				err := fmt.Errorf("content negotiation failed: mediaType.Convert is nil for application/json")
				utilruntime.HandleError(err)
				responsewriters.ErrorNegotiated(
					err,
					serializer,
					schema.GroupVersion{},
					w,
					r,
				)
				return
			}
			targetGV = mediaType.Convert.GroupVersion()
			if reg.deprecatedVersions()[targetGV.Version] {
				w.Header().Set("Warning", `299 - "This version of the flagz endpoint is deprecated. Please use a newer version."`)
			}
		case "text/plain":
			writePlainTextResponse(obj, serializer, w, reg)
			return
		default:
			err = fmt.Errorf("content negotiation failed: unsupported media type '%s'", serializerInfo.MediaType)
			utilruntime.HandleError(err)
			responsewriters.ErrorNegotiated(
				err,
				serializer,
				schema.GroupVersion{},
				w,
				r,
			)
			return
		}

		writeResponse(obj, serializer, targetGV, restrictions, w, r)
	}
}

func writePlainTextResponse(obj runtime.Object, serializer runtime.NegotiatedSerializer, w http.ResponseWriter, reg *registry) {
	reg.cachedPlainTextResponseLock.Lock()
	defer reg.cachedPlainTextResponseLock.Unlock()
	if reg.cachedPlainTextResponse != nil {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if _, err := w.Write(reg.cachedPlainTextResponse); err != nil {
			utilruntime.HandleError(fmt.Errorf("error writing cached flagz as text/plain: %w", err))
		}
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	var textSerializer runtime.Serializer
	for _, info := range serializer.SupportedMediaTypes() {
		if info.MediaType == "text/plain" {
			textSerializer = info.Serializer
			break
		}
	}
	if textSerializer == nil {
		utilruntime.HandleError(fmt.Errorf("text/plain serializer not available"))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	var buf strings.Builder
	if err := textSerializer.Encode(obj, &buf); err != nil {
		utilruntime.HandleError(fmt.Errorf("error encoding flagz as text/plain: %w", err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	reg.cachedPlainTextResponse = []byte(buf.String())
	if _, err := w.Write(reg.cachedPlainTextResponse); err != nil {
		utilruntime.HandleError(fmt.Errorf("error writing flagz as text/plain: %w", err))
	}
}

func writeResponse(obj runtime.Object, serializer runtime.NegotiatedSerializer, targetGV schema.GroupVersion, restrictions negotiate.FlagzEndpointRestrictions, w http.ResponseWriter, r *http.Request) {
	responsewriters.WriteObjectNegotiated(
		serializer,
		restrictions,
		targetGV,
		w,
		r,
		http.StatusOK,
		obj,
		true,
	)
}

func flagz(componentName string, flagReader Reader) *v1alpha1.Flagz {
	flags := flagReader.GetFlagz()
	return &v1alpha1.Flagz{
		TypeMeta: metav1.TypeMeta{
			Kind:       Kind,
			APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: componentName,
		},
		Flags: flags,
	}
}
