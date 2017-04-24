// Copyright 2016 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package version

import "fmt"

type ErrorIncompatible struct {
	Config    string
	Supported []string
}

func (e *ErrorIncompatible) Details() string {
	return fmt.Sprintf("config is %q, plugin supports %q", e.Config, e.Supported)
}

func (e *ErrorIncompatible) Error() string {
	return fmt.Sprintf("incompatible CNI versions: %s", e.Details())
}

type Reconciler struct{}

func (r *Reconciler) Check(configVersion string, pluginInfo PluginInfo) *ErrorIncompatible {
	return r.CheckRaw(configVersion, pluginInfo.SupportedVersions())
}

func (*Reconciler) CheckRaw(configVersion string, supportedVersions []string) *ErrorIncompatible {
	for _, supportedVersion := range supportedVersions {
		if configVersion == supportedVersion {
			return nil
		}
	}

	return &ErrorIncompatible{
		Config:    configVersion,
		Supported: supportedVersions,
	}
}
