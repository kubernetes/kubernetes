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
	"archive/zip"
	"bytes"
	"embed"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/chai2010/gettext-go"

	"k8s.io/klog/v2"
)

//go:embed translations
var translations embed.FS

var knownTranslations = map[string][]string{
	"kubectl": {
		"default",
		"en_US",
		"fr_FR",
		"zh_CN",
		"ja_JP",
		"zh_TW",
		"it_IT",
		"de_DE",
		"ko_KR",
		"pt_BR",
	},
	// only used for unit tests.
	"test": {
		"default",
		"en_US",
	},
}

var (
	lazyLoadTranslationsOnce sync.Once
	LoadTranslationsFunc     = func() error {
		return LoadTranslations("kubectl", nil)
	}
	translationsLoaded bool
)

// SetLoadTranslationsFunc sets the function called to lazy load translations.
// It must be called in an init() func that occurs BEFORE any i18n.T() calls are made by any package. You can
// accomplish this by creating a separate package containing your init() func, and then importing that package BEFORE
// any other packages that call i18n.T().
//
// Example Usage:
//
//	package myi18n
//
//	import "k8s.io/kubectl/pkg/util/i18n"
//
//	func init() {
//		if err := i18n.SetLoadTranslationsFunc(loadCustomTranslations); err != nil {
//			panic(err)
//		}
//	}
//
//	func loadCustomTranslations() error {
//		// Load your custom translations here...
//	}
//
// And then in your main or root command package, import your custom package like this:
//
//	import (
//		// Other imports that don't need i18n...
//		_ "example.com/myapp/myi18n"
//		// Other imports that do need i18n...
//	)
func SetLoadTranslationsFunc(f func() error) error {
	if translationsLoaded {
		return errors.New("translations have already been loaded")
	}
	LoadTranslationsFunc = func() error {
		if err := f(); err != nil {
			return err
		}
		translationsLoaded = true
		return nil
	}
	return nil
}

func loadSystemLanguage() string {
	// Implements the following locale priority order: LC_ALL, LC_MESSAGES, LANG
	// Similarly to: https://www.gnu.org/software/gettext/manual/html_node/Locale-Environment-Variables.html
	langStr := os.Getenv("LC_ALL")
	if langStr == "" {
		langStr = os.Getenv("LC_MESSAGES")
	}
	if langStr == "" {
		langStr = os.Getenv("LANG")
	}

	if langStr == "" {
		klog.V(3).Infof("Couldn't find the LC_ALL, LC_MESSAGES or LANG environment variables, defaulting to en_US")
		return "default"
	}
	pieces := strings.Split(langStr, ".")
	if len(pieces) != 2 {
		klog.V(3).Infof("Unexpected system language (%s), defaulting to en_US", langStr)
		return "default"
	}
	return pieces[0]
}

func findLanguage(root string, getLanguageFn func() string) string {
	langStr := getLanguageFn()

	translations := knownTranslations[root]
	for ix := range translations {
		if translations[ix] == langStr {
			return langStr
		}
	}
	klog.V(3).Infof("Couldn't find translations for %s, using default", langStr)
	return "default"
}

// LoadTranslations loads translation files. getLanguageFn should return a language
// string (e.g. 'en-US'). If getLanguageFn is nil, then the loadSystemLanguage function
// is used, which uses the 'LANG' environment variable.
func LoadTranslations(root string, getLanguageFn func() string) error {
	if getLanguageFn == nil {
		getLanguageFn = loadSystemLanguage
	}

	langStr := findLanguage(root, getLanguageFn)
	translationFiles := []string{
		fmt.Sprintf("%s/%s/LC_MESSAGES/k8s.po", root, langStr),
		fmt.Sprintf("%s/%s/LC_MESSAGES/k8s.mo", root, langStr),
	}

	// TODO: list the directory and load all files.
	buf := new(bytes.Buffer)
	w := zip.NewWriter(buf)

	// Make sure to check the error on Close.
	for _, file := range translationFiles {
		filename := "translations/" + file
		f, err := w.Create(file)
		if err != nil {
			return err
		}
		data, err := translations.ReadFile(filename)
		if err != nil {
			return err
		}
		if _, err := f.Write(data); err != nil {
			return nil
		}
	}
	if err := w.Close(); err != nil {
		return err
	}
	gettext.BindLocale(gettext.New("k8s", root+".zip", buf.Bytes()))
	gettext.SetDomain("k8s")
	gettext.SetLanguage(langStr)
	translationsLoaded = true
	return nil
}

func lazyLoadTranslations() {
	lazyLoadTranslationsOnce.Do(func() {
		if translationsLoaded {
			return
		}
		if err := LoadTranslationsFunc(); err != nil {
			klog.Warning("Failed to load translations")
		}
	})
}

// T translates a string, possibly substituting arguments into it along
// the way. If len(args) is > 0, args1 is assumed to be the plural value
// and plural translation is used.
func T(defaultValue string, args ...int) string {
	lazyLoadTranslations()
	if len(args) == 0 {
		return gettext.PGettext("", defaultValue)
	}
	return fmt.Sprintf(gettext.PNGettext("", defaultValue, defaultValue+".plural", args[0]),
		args[0])
}

// Errorf produces an error with a translated error string.
// Substitution is performed via the `T` function above, following
// the same rules.
func Errorf(defaultValue string, args ...int) error {
	return errors.New(T(defaultValue, args...))
}
