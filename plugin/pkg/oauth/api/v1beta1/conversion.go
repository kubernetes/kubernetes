/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta1

import (
	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	onewer "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

func init() {
	newer.Scheme.AddConversionFuncs(
		// Run default conversions, then copy Labels to ObjectMeta.Labels
		func(in *onewer.OAuthAccessToken, out *OAuthAccessToken, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *OAuthAccessToken, out *onewer.OAuthAccessToken, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *onewer.OAuthAuthorizeToken, out *OAuthAuthorizeToken, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *OAuthAuthorizeToken, out *onewer.OAuthAuthorizeToken, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *onewer.OAuthClient, out *OAuthClient, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *OAuthClient, out *onewer.OAuthClient, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *onewer.OAuthClientAuthorization, out *OAuthClientAuthorization, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *OAuthClientAuthorization, out *onewer.OAuthClientAuthorization, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
	)
}
