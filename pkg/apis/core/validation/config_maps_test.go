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

package validation

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateConfigMap(t *testing.T) {
	newConfigMap := func(name, namespace string, data map[string]string, binaryData map[string][]byte) core.ConfigMap {
		return core.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: namespace,
			},
			Data:       data,
			BinaryData: binaryData,
		}
	}

	var (
		validConfigMap = newConfigMap("validname", "validns", map[string]string{"key": "value"}, map[string][]byte{"bin": []byte("value")})
		maxKeyLength   = newConfigMap("validname", "validns", map[string]string{strings.Repeat("a", 253): "value"}, nil)

		emptyName               = newConfigMap("", "validns", nil, nil)
		invalidName             = newConfigMap("NoUppercaseOrSpecialCharsLike=Equals", "validns", nil, nil)
		emptyNs                 = newConfigMap("validname", "", nil, nil)
		invalidNs               = newConfigMap("validname", "NoUppercaseOrSpecialCharsLike=Equals", nil, nil)
		invalidKey              = newConfigMap("validname", "validns", map[string]string{"a*b": "value"}, nil)
		leadingDotKey           = newConfigMap("validname", "validns", map[string]string{".ab": "value"}, nil)
		dotKey                  = newConfigMap("validname", "validns", map[string]string{".": "value"}, nil)
		doubleDotKey            = newConfigMap("validname", "validns", map[string]string{"..": "value"}, nil)
		overMaxKeyLength        = newConfigMap("validname", "validns", map[string]string{strings.Repeat("a", 254): "value"}, nil)
		overMaxSize             = newConfigMap("validname", "validns", map[string]string{"key": strings.Repeat("a", v1.MaxSecretSize+1)}, nil)
		duplicatedKey           = newConfigMap("validname", "validns", map[string]string{"key": "value1"}, map[string][]byte{"key": []byte("value2")})
		binDataInvalidKey       = newConfigMap("validname", "validns", nil, map[string][]byte{"a*b": []byte("value")})
		binDataLeadingDotKey    = newConfigMap("validname", "validns", nil, map[string][]byte{".ab": []byte("value")})
		binDataDotKey           = newConfigMap("validname", "validns", nil, map[string][]byte{".": []byte("value")})
		binDataDoubleDotKey     = newConfigMap("validname", "validns", nil, map[string][]byte{"..": []byte("value")})
		binDataOverMaxKeyLength = newConfigMap("validname", "validns", nil, map[string][]byte{strings.Repeat("a", 254): []byte("value")})
		binDataOverMaxSize      = newConfigMap("validname", "validns", nil, map[string][]byte{"bin": bytes.Repeat([]byte("a"), v1.MaxSecretSize+1)})
		binNonUtf8Value         = newConfigMap("validname", "validns", nil, map[string][]byte{"key": {0, 0xFE, 0, 0xFF}})
	)

	tests := map[string]struct {
		cfg     core.ConfigMap
		isValid bool
	}{
		"valid":                           {validConfigMap, true},
		"max key length":                  {maxKeyLength, true},
		"leading dot key":                 {leadingDotKey, true},
		"empty name":                      {emptyName, false},
		"invalid name":                    {invalidName, false},
		"invalid key":                     {invalidKey, false},
		"empty namespace":                 {emptyNs, false},
		"invalid namespace":               {invalidNs, false},
		"dot key":                         {dotKey, false},
		"double dot key":                  {doubleDotKey, false},
		"over max key length":             {overMaxKeyLength, false},
		"over max size":                   {overMaxSize, false},
		"duplicated key":                  {duplicatedKey, false},
		"binary data invalid key":         {binDataInvalidKey, false},
		"binary data leading dot key":     {binDataLeadingDotKey, true},
		"binary data dot key":             {binDataDotKey, false},
		"binary data double dot key":      {binDataDoubleDotKey, false},
		"binary data over max key length": {binDataOverMaxKeyLength, false},
		"binary data max size":            {binDataOverMaxSize, false},
		"binary data non utf-8 bytes":     {binNonUtf8Value, true},
	}

	for name, tc := range tests {
		errs := ValidateConfigMap(&tc.cfg)
		if tc.isValid && len(errs) > 0 {
			t.Errorf("%v: unexpected error: %v", name, errs)
		}
		if !tc.isValid && len(errs) == 0 {
			t.Errorf("%v: unexpected non-error", name)
		}
	}
}

func TestValidateConfigMapUpdate(t *testing.T) {
	newConfigMap := func(version, name, namespace string, data map[string]string) core.ConfigMap {
		return core.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       namespace,
				ResourceVersion: version,
			},
			Data: data,
		}
	}

	var (
		validConfigMap = newConfigMap("1", "validname", "validns", map[string]string{"key": "value"})
		noVersion      = newConfigMap("", "validname", "validns", map[string]string{"key": "value"})
	)

	cases := []struct {
		name    string
		newCfg  core.ConfigMap
		oldCfg  core.ConfigMap
		isValid bool
	}{
		{
			name:    "valid",
			newCfg:  validConfigMap,
			oldCfg:  validConfigMap,
			isValid: true,
		},
		{
			name:    "invalid",
			newCfg:  noVersion,
			oldCfg:  validConfigMap,
			isValid: false,
		},
	}

	for _, tc := range cases {
		errs := ValidateConfigMapUpdate(&tc.newCfg, &tc.oldCfg)
		if tc.isValid && len(errs) > 0 {
			t.Errorf("%v: unexpected error: %v", tc.name, errs)
		}
		if !tc.isValid && len(errs) == 0 {
			t.Errorf("%v: unexpected non-error", tc.name)
		}
	}
}
