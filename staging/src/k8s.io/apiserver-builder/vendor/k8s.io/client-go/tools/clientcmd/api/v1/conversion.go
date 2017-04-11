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

package v1

import (
	"sort"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/clientcmd/api"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		func(in *Cluster, out *api.Cluster, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *api.Cluster, out *Cluster, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *Preferences, out *api.Preferences, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *api.Preferences, out *Preferences, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *AuthInfo, out *api.AuthInfo, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *api.AuthInfo, out *AuthInfo, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *Context, out *api.Context, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},
		func(in *api.Context, out *Context, s conversion.Scope) error {
			return s.DefaultConvert(in, out, conversion.IgnoreMissingFields)
		},

		func(in *Config, out *api.Config, s conversion.Scope) error {
			out.CurrentContext = in.CurrentContext
			if err := s.Convert(&in.Preferences, &out.Preferences, 0); err != nil {
				return err
			}

			out.Clusters = make(map[string]*api.Cluster)
			if err := s.Convert(&in.Clusters, &out.Clusters, 0); err != nil {
				return err
			}
			out.AuthInfos = make(map[string]*api.AuthInfo)
			if err := s.Convert(&in.AuthInfos, &out.AuthInfos, 0); err != nil {
				return err
			}
			out.Contexts = make(map[string]*api.Context)
			if err := s.Convert(&in.Contexts, &out.Contexts, 0); err != nil {
				return err
			}
			out.Extensions = make(map[string]runtime.Object)
			if err := s.Convert(&in.Extensions, &out.Extensions, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *api.Config, out *Config, s conversion.Scope) error {
			out.CurrentContext = in.CurrentContext
			if err := s.Convert(&in.Preferences, &out.Preferences, 0); err != nil {
				return err
			}

			out.Clusters = make([]NamedCluster, 0, 0)
			if err := s.Convert(&in.Clusters, &out.Clusters, 0); err != nil {
				return err
			}
			out.AuthInfos = make([]NamedAuthInfo, 0, 0)
			if err := s.Convert(&in.AuthInfos, &out.AuthInfos, 0); err != nil {
				return err
			}
			out.Contexts = make([]NamedContext, 0, 0)
			if err := s.Convert(&in.Contexts, &out.Contexts, 0); err != nil {
				return err
			}
			out.Extensions = make([]NamedExtension, 0, 0)
			if err := s.Convert(&in.Extensions, &out.Extensions, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *[]NamedCluster, out *map[string]*api.Cluster, s conversion.Scope) error {
			for _, curr := range *in {
				newCluster := api.NewCluster()
				if err := s.Convert(&curr.Cluster, newCluster, 0); err != nil {
					return err
				}
				(*out)[curr.Name] = newCluster
			}

			return nil
		},
		func(in *map[string]*api.Cluster, out *[]NamedCluster, s conversion.Scope) error {
			allKeys := make([]string, 0, len(*in))
			for key := range *in {
				allKeys = append(allKeys, key)
			}
			sort.Strings(allKeys)

			for _, key := range allKeys {
				newCluster := (*in)[key]
				oldCluster := &Cluster{}
				if err := s.Convert(newCluster, oldCluster, 0); err != nil {
					return err
				}

				namedCluster := NamedCluster{key, *oldCluster}
				*out = append(*out, namedCluster)
			}

			return nil
		},
		func(in *[]NamedAuthInfo, out *map[string]*api.AuthInfo, s conversion.Scope) error {
			for _, curr := range *in {
				newAuthInfo := api.NewAuthInfo()
				if err := s.Convert(&curr.AuthInfo, newAuthInfo, 0); err != nil {
					return err
				}
				(*out)[curr.Name] = newAuthInfo
			}

			return nil
		},
		func(in *map[string]*api.AuthInfo, out *[]NamedAuthInfo, s conversion.Scope) error {
			allKeys := make([]string, 0, len(*in))
			for key := range *in {
				allKeys = append(allKeys, key)
			}
			sort.Strings(allKeys)

			for _, key := range allKeys {
				newAuthInfo := (*in)[key]
				oldAuthInfo := &AuthInfo{}
				if err := s.Convert(newAuthInfo, oldAuthInfo, 0); err != nil {
					return err
				}

				namedAuthInfo := NamedAuthInfo{key, *oldAuthInfo}
				*out = append(*out, namedAuthInfo)
			}

			return nil
		},
		func(in *[]NamedContext, out *map[string]*api.Context, s conversion.Scope) error {
			for _, curr := range *in {
				newContext := api.NewContext()
				if err := s.Convert(&curr.Context, newContext, 0); err != nil {
					return err
				}
				(*out)[curr.Name] = newContext
			}

			return nil
		},
		func(in *map[string]*api.Context, out *[]NamedContext, s conversion.Scope) error {
			allKeys := make([]string, 0, len(*in))
			for key := range *in {
				allKeys = append(allKeys, key)
			}
			sort.Strings(allKeys)

			for _, key := range allKeys {
				newContext := (*in)[key]
				oldContext := &Context{}
				if err := s.Convert(newContext, oldContext, 0); err != nil {
					return err
				}

				namedContext := NamedContext{key, *oldContext}
				*out = append(*out, namedContext)
			}

			return nil
		},
		func(in *[]NamedExtension, out *map[string]runtime.Object, s conversion.Scope) error {
			for _, curr := range *in {
				var newExtension runtime.Object
				if err := s.Convert(&curr.Extension, &newExtension, 0); err != nil {
					return err
				}
				(*out)[curr.Name] = newExtension
			}

			return nil
		},
		func(in *map[string]runtime.Object, out *[]NamedExtension, s conversion.Scope) error {
			allKeys := make([]string, 0, len(*in))
			for key := range *in {
				allKeys = append(allKeys, key)
			}
			sort.Strings(allKeys)

			for _, key := range allKeys {
				newExtension := (*in)[key]
				oldExtension := &runtime.RawExtension{}
				if err := s.Convert(newExtension, oldExtension, 0); err != nil {
					return err
				}

				namedExtension := NamedExtension{key, *oldExtension}
				*out = append(*out, namedExtension)
			}

			return nil
		},
	)
}
