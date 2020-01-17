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
	"errors"
	"fmt"
	"os"
	"strings"

	"k8s.io/kubectl/pkg/generated"

	"github.com/chai2010/gettext-go/gettext"
	"k8s.io/klog"
)

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
	},
	// only used for unit tests.
	"test": {
		"default",
		"en_US",
	},
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

	klog.V(3).Infof("Setting language to %s", langStr)
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
		data, err := generated.Asset(filename)
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
	gettext.BindTextdomain("k8s", root+".zip", buf.Bytes())
	gettext.Textdomain("k8s")
	gettext.SetLocale(langStr)
	return nil
}

// T translates a string, possibly substituting arguments into it along
// the way. If len(args) is > 0, args1 is assumed to be the plural value
// and plural translation is used.
func T(defaultValue string, args ...int) string {
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
