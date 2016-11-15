/*
Copyright 2016 The Kubernetes Authors.

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

package i18n

import (
	"os"
	"strings"
	"sync"

	"k8s.io/kubernetes/pkg/generated"
	"k8s.io/kubernetes/pkg/util/yaml"

	"github.com/golang/glog"
	"github.com/nicksnyder/go-i18n/i18n/bundle"
	"golang.org/x/text/language"
)

// see https://blog.golang.org/matchlang
var supported = []language.Tag{
	language.AmericanEnglish, // en-US: first language is fallback
}

var (
	lang language.Tag
	bdl  *bundle.Bundle
	lock sync.Mutex
)

func init() {
	if err := LoadTranslations(); err != nil {
		glog.Errorf("Failed to load translations: %v", err)
	}
}

func loadSystemLanguage() string {
	langStr := os.Getenv("LANG")
	if langStr == "" {
		glog.V(3).Infof("Couldn't find the LANG environment variable, defaulting to en-US")
		return "en-US"
	}
	pieces := strings.Split(langStr, ".")
	if len(pieces) == 0 {
		glog.V(3).Infof("Unexpected system language (%s), defaulting to en-US", langStr)
		return "en-US"
	} else {
		return pieces[0]
	}
}

// LoadTranslations loads translation files.
func LoadTranslations() error {
	langStr := loadSystemLanguage()
	glog.V(3).Infof("Setting language to %s", langStr)
	// TODO: list the directory and load all files.
	for _, file := range []string{"en-US.all.yaml"} {
		data, err := generated.Asset("translations/" + file)
		if err != nil {
			return err
		}
		jsonData, err := yaml.ToJSON(data)
		if err != nil {
			return err
		}
		if err := loadTranslationsInternal(langStr, file, jsonData); err != nil {
			return err
		}
	}
	return nil
}

func loadTranslationsInternal(localLanguage, filename string, data []byte) error {
	lock.Lock()
	defer lock.Unlock()
	if bdl != nil {
		return nil
	}

	local, err := language.Parse(localLanguage)
	if err != nil {
		return err
	}
	bdl = bundle.New()
	if err = bdl.ParseTranslationFileBytes(filename, data); err != nil {
		return err
	}
	matcher := language.NewMatcher(supported)
	lang, _, _ = matcher.Match(local)

	return nil
}

func T(defaultValue string, args ...interface{}) string {
	fn, err := bdl.Tfunc(lang.String())
	if err != nil {
		glog.Errorf("Failed to translate: %s (%v)", defaultValue, err)
		return defaultValue
	}
	return fn(defaultValue, args...)
}
