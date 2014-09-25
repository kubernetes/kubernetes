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

package resources

import (
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const (
	CPU    api.ResourceName = "cpu"
	Memory api.ResourceName = "memory"
)

// TODO: None of these currently handle SI units

func GetFloatResource(resources api.ResourceList, name api.ResourceName, def float64) float64 {
	value, found := resources[name]
	if !found {
		return def
	}
	if value.Kind == util.IntstrInt {
		return float64(value.IntVal)
	}
	result, err := strconv.ParseFloat(value.StrVal, 64)
	if err != nil {
		glog.Errorf("parsing failed for %s: %s", name, value.StrVal)
		return def
	}
	return result
}

func GetIntegerResource(resources api.ResourceList, name api.ResourceName, def int) int {
	value, found := resources[name]
	if !found {
		return def
	}
	if value.Kind == util.IntstrInt {
		return value.IntVal
	}
	result, err := strconv.Atoi(value.StrVal)
	if err != nil {
		glog.Errorf("parsing failed for %s: %s", name, value.StrVal)
		return def
	}
	return result
}

func GetStringResource(resources api.ResourceList, name api.ResourceName, def string) string {
	value, found := resources[name]
	if !found {
		return def
	}
	if value.Kind == util.IntstrInt {
		return strconv.Itoa(value.IntVal)
	}
	return value.StrVal
}
