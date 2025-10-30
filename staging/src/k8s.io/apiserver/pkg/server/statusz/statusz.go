/*
Copyright 2024 The Kubernetes Authors.

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

package statusz

import (
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/component-base/compatibility"

	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/server/statusz/negotiate"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	v1alpha1 "k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"
)

var (
	delimiters            = []string{":", ": ", "=", " "}
	nonDebuggingEndpoints = map[string]bool{
		"/apis":        true,
		"/api":         true,
		"/openid":      true,
		"/openapi":     true,
		"/.well-known": true,
	}
)

const (
	DefaultStatuszPath = "/statusz"
	Kind               = "Statusz"
	GroupName          = "config.k8s.io"
	Version            = "v1alpha1"
)

const headerFmt = `
%s statusz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.
`

var schemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: Version}

type mux interface {
	Handle(path string, handler http.Handler)
}

type ListedPathsOption []string

func NewRegistry(effectiveVersion compatibility.EffectiveVersion, opts ...Option) statuszRegistry {
	r := &registry{
		effectiveVersion: effectiveVersion,
	}
	for _, opt := range opts {
		opt(r)
	}

	return r
}

func Install(m mux, componentName string, reg statuszRegistry) {
	scheme := runtime.NewScheme()
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	codecFactory := serializer.NewCodecFactory(
		scheme,
		serializer.WithSerializer(func(_ runtime.ObjectCreater, _ runtime.ObjectTyper) runtime.SerializerInfo {
			textSerializer := statuszTextSerializer{componentName, reg}
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
	m.Handle(DefaultStatuszPath, handleStatusz(componentName, reg, codecFactory, negotiate.StatuszEndpointRestrictions{}))
}

func handleStatusz(componentName string, reg statuszRegistry, serializer runtime.NegotiatedSerializer, restrictions negotiate.StatuszEndpointRestrictions) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		obj := statusz(componentName, reg)
		acceptHeader := r.Header.Get("Accept")
		if strings.TrimSpace(acceptHeader) == "" {
			writePlainTextResponse(obj, serializer, w)
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
			deprecated := reg.deprecatedVersions()[targetGV.Version]
			if deprecated {
				w.Header().Set("Warning", `299 - "This version of the statusz endpoint is deprecated. Please use a newer version."`)
			}
		case "text/plain":
			// Even though text/plain serialization does not use the group/version,
			// the serialization machinery expects a non-zero schema.GroupVersion to be passed.
			// Passing the zero value can cause errors or unexpected behavior in the negotiation logic.
			targetGV = schemeGroupVersion
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

// writePlainTextResponse writes the statusz response as text/plain using the registered serializer.
func writePlainTextResponse(obj runtime.Object, serializer runtime.NegotiatedSerializer, w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	// Find the text/plain serializer
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
	if err := textSerializer.Encode(obj, w); err != nil {
		utilruntime.HandleError(fmt.Errorf("error encoding statusz as text/plain: %w", err))
		w.WriteHeader(http.StatusInternalServerError)
	}
}

func writeResponse(obj runtime.Object, serializer runtime.NegotiatedSerializer, targetGV schema.GroupVersion, restrictions negotiate.StatuszEndpointRestrictions, w http.ResponseWriter, r *http.Request) {
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

func statusz(componentName string, reg statuszRegistry) *v1alpha1.Statusz {
	startTime := reg.processStartTime()
	upTimeSeconds := max(0, int64(time.Since(startTime).Seconds()))
	goVersion := reg.goVersion()
	binaryVersion := reg.binaryVersion().String()
	var emulationVersion string
	if reg.emulationVersion() != nil {
		emulationVersion = reg.emulationVersion().String()
	}

	paths := aggregatePaths(reg.paths())
	data := &v1alpha1.Statusz{
		TypeMeta: metav1.TypeMeta{
			Kind:       Kind,
			APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: componentName,
		},
		StartTime:        metav1.Time{Time: startTime},
		UptimeSeconds:    upTimeSeconds,
		GoVersion:        goVersion,
		BinaryVersion:    binaryVersion,
		EmulationVersion: emulationVersion,
		Paths:            paths,
	}

	return data
}

func uptime(t time.Time) string {
	upSince := int64(time.Since(t).Seconds())
	return fmt.Sprintf("%d hr %02d min %02d sec",
		upSince/3600, (upSince/60)%60, upSince%60)
}

func aggregatePaths(listedPaths []string) []string {
	paths := make(map[string]bool)
	for _, listedPath := range listedPaths {
		parts := strings.Split(listedPath, "/")
		if len(parts) < 2 || parts[1] == "" {
			continue
		}
		folder := "/" + parts[1]
		if !paths[folder] && !nonDebuggingEndpoints[folder] {
			paths[folder] = true
		}
	}

	var sortedPaths []string
	for p := range paths {
		sortedPaths = append(sortedPaths, p)
	}
	sort.Strings(sortedPaths)

	return sortedPaths
}
