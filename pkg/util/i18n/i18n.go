package i18n

import (
	"strings"
	"sync"

	"k8s.io/kubernetes/pkg/util/i18n/translations"

	"os"

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

// LoadTranslations loads translation files.
func LoadTranslations() error {
	lang := os.Getenv("LANG")
	pieces := strings.Split(lang, ".")
	if len(pieces) == 0 {
		glog.V(3).Infof("Couldn't find system language defaulting to en-US")
		lang = "en-US"
	} else {
		lang = pieces[0]
	}
	return loadTranslationsInternal(lang)
}

func loadTranslationsInternal(localLanguage string) error {
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
	if err = bdl.ParseTranslationFileBytes(translations.EnUSFileName, translations.EnUSData); err != nil {
		return err
	}
	matcher := language.NewMatcher(supported)
	lang, _, _ = matcher.Match(local)

	return nil
}

func T(input string) string {
	fn, err := bdl.Tfunc(lang.String())
	if err != nil {
		glog.Errorf("Failed to translate: %s (%v)", input, err)
		return input
	}
	return fn(input)
}
