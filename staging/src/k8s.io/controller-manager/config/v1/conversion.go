/*
Copyright 2022 The Kubernetes Authors.

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
	"unsafe"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/controller-manager/config"
)

func init() {
	localSchemeBuilder.Register(RegisterConversions)
}

const ResourceLockLeases = "leases"

// RegisterConversions adds conversion functions to the given scheme.
// Public to allow building arbitrary schemes.
func RegisterConversions(s *runtime.Scheme) error {
	if err := s.AddGeneratedConversionFunc((*ControllerLeaderConfiguration)(nil), (*config.ControllerLeaderConfiguration)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_v1_ControllerLeaderConfiguration_To_config_ControllerLeaderConfiguration(a.(*ControllerLeaderConfiguration), b.(*config.ControllerLeaderConfiguration), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*config.ControllerLeaderConfiguration)(nil), (*ControllerLeaderConfiguration)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_config_ControllerLeaderConfiguration_To_v1_ControllerLeaderConfiguration(a.(*config.ControllerLeaderConfiguration), b.(*ControllerLeaderConfiguration), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*LeaderMigrationConfiguration)(nil), (*config.LeaderMigrationConfiguration)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_v1_LeaderMigrationConfiguration_To_config_LeaderMigrationConfiguration(a.(*LeaderMigrationConfiguration), b.(*config.LeaderMigrationConfiguration), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*config.LeaderMigrationConfiguration)(nil), (*LeaderMigrationConfiguration)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_config_LeaderMigrationConfiguration_To_v1_LeaderMigrationConfiguration(a.(*config.LeaderMigrationConfiguration), b.(*LeaderMigrationConfiguration), scope)
	}); err != nil {
		return err
	}
	return nil
}

func Convert_config_LeaderMigrationConfiguration_To_v1_LeaderMigrationConfiguration(in *config.LeaderMigrationConfiguration, out *LeaderMigrationConfiguration, s conversion.Scope) error {
	out.LeaderName = in.LeaderName
	out.ControllerLeaders = *(*[]ControllerLeaderConfiguration)(unsafe.Pointer(&in.ControllerLeaders))
	return nil
}

func Convert_v1_LeaderMigrationConfiguration_To_config_LeaderMigrationConfiguration(in *LeaderMigrationConfiguration, out *config.LeaderMigrationConfiguration, s conversion.Scope) error {
	out.LeaderName = in.LeaderName
	out.ControllerLeaders = *(*[]config.ControllerLeaderConfiguration)(unsafe.Pointer(&in.ControllerLeaders))
	out.ResourceLock = ResourceLockLeases
	return nil
}

func Convert_v1_ControllerLeaderConfiguration_To_config_ControllerLeaderConfiguration(in *ControllerLeaderConfiguration, out *config.ControllerLeaderConfiguration, s conversion.Scope) error {
	out.Name = in.Name
	out.Component = in.Component
	return nil
}

func Convert_config_ControllerLeaderConfiguration_To_v1_ControllerLeaderConfiguration(in *config.ControllerLeaderConfiguration, out *ControllerLeaderConfiguration, s conversion.Scope) error {
	out.Name = in.Name
	out.Component = in.Component
	return nil
}
