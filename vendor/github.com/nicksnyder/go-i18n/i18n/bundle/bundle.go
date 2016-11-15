// Package bundle manages translations for multiple languages.
package bundle

import (
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"reflect"

	"path/filepath"

	"github.com/nicksnyder/go-i18n/i18n/language"
	"github.com/nicksnyder/go-i18n/i18n/translation"
)

// TranslateFunc is a copy of i18n.TranslateFunc to avoid a circular dependency.
type TranslateFunc func(translationID string, args ...interface{}) string

// Bundle stores the translations for multiple languages.
type Bundle struct {
	// The primary translations for a language tag and translation id.
	translations map[string]map[string]translation.Translation

	// Translations that can be used when an exact language match is not possible.
	fallbackTranslations map[string]map[string]translation.Translation
}

// New returns an empty bundle.
func New() *Bundle {
	return &Bundle{
		translations:         make(map[string]map[string]translation.Translation),
		fallbackTranslations: make(map[string]map[string]translation.Translation),
	}
}

// MustLoadTranslationFile is similar to LoadTranslationFile
// except it panics if an error happens.
func (b *Bundle) MustLoadTranslationFile(filename string) {
	if err := b.LoadTranslationFile(filename); err != nil {
		panic(err)
	}
}

// LoadTranslationFile loads the translations from filename into memory.
//
// The language that the translations are associated with is parsed from the filename (e.g. en-US.json).
//
// Generally you should load translation files once during your program's initialization.
func (b *Bundle) LoadTranslationFile(filename string) error {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	return b.ParseTranslationFileBytes(filename, buf)
}

// ParseTranslationFileBytes is similar to LoadTranslationFile except it parses the bytes in buf.
//
// It is useful for parsing translation files embedded with go-bindata.
func (b *Bundle) ParseTranslationFileBytes(filename string, buf []byte) error {
	basename := filepath.Base(filename)
	langs := language.Parse(basename)
	switch l := len(langs); {
	case l == 0:
		return fmt.Errorf("no language found in %q", basename)
	case l > 1:
		return fmt.Errorf("multiple languages found in filename %q: %v; expected one", basename, langs)
	}
	translations, err := parseTranslations(filename, buf)
	if err != nil {
		return err
	}
	b.AddTranslation(langs[0], translations...)
	return nil
}

func parseTranslations(filename string, buf []byte) ([]translation.Translation, error) {
	var unmarshalFunc func([]byte, interface{}) error
	switch format := filepath.Ext(filename); format {
	case ".json":
		unmarshalFunc = json.Unmarshal
	case ".yaml":
		unmarshalFunc = yaml.Unmarshal
	default:
		return nil, fmt.Errorf("unsupported file extension %s", format)
	}

	var translationsData []map[string]interface{}
	if len(buf) > 0 {
		if err := unmarshalFunc(buf, &translationsData); err != nil {
			return nil, err
		}
	}

	translations := make([]translation.Translation, 0, len(translationsData))
	for i, translationData := range translationsData {
		t, err := translation.NewTranslation(translationData)
		if err != nil {
			return nil, fmt.Errorf("unable to parse translation #%d in %s because %s\n%v", i, filename, err, translationData)
		}
		translations = append(translations, t)
	}
	return translations, nil
}

// AddTranslation adds translations for a language.
//
// It is useful if your translations are in a format not supported by LoadTranslationFile.
func (b *Bundle) AddTranslation(lang *language.Language, translations ...translation.Translation) {
	if b.translations[lang.Tag] == nil {
		b.translations[lang.Tag] = make(map[string]translation.Translation, len(translations))
	}
	currentTranslations := b.translations[lang.Tag]
	for _, newTranslation := range translations {
		if currentTranslation := currentTranslations[newTranslation.ID()]; currentTranslation != nil {
			currentTranslations[newTranslation.ID()] = currentTranslation.Merge(newTranslation)
		} else {
			currentTranslations[newTranslation.ID()] = newTranslation
		}
	}

	// lang can provide translations for less specific language tags.
	for _, tag := range lang.MatchingTags() {
		b.fallbackTranslations[tag] = currentTranslations
	}
}

// Translations returns all translations in the bundle.
func (b *Bundle) Translations() map[string]map[string]translation.Translation {
	return b.translations
}

// LanguageTags returns the tags of all languages that that have been added.
func (b *Bundle) LanguageTags() []string {
	var tags []string
	for k := range b.translations {
		tags = append(tags, k)
	}
	return tags
}

// LanguageTranslationIDs returns the ids of all translations that have been added for a given language.
func (b *Bundle) LanguageTranslationIDs(languageTag string) []string {
	var ids []string
	for id := range b.translations[languageTag] {
		ids = append(ids, id)
	}
	return ids
}

// MustTfunc is similar to Tfunc except it panics if an error happens.
func (b *Bundle) MustTfunc(pref string, prefs ...string) TranslateFunc {
	tfunc, err := b.Tfunc(pref, prefs...)
	if err != nil {
		panic(err)
	}
	return tfunc
}

// MustTfuncAndLanguage is similar to TfuncAndLanguage except it panics if an error happens.
func (b *Bundle) MustTfuncAndLanguage(pref string, prefs ...string) (TranslateFunc, *language.Language) {
	tfunc, language, err := b.TfuncAndLanguage(pref, prefs...)
	if err != nil {
		panic(err)
	}
	return tfunc, language
}

// Tfunc is similar to TfuncAndLanguage except is doesn't return the Language.
func (b *Bundle) Tfunc(pref string, prefs ...string) (TranslateFunc, error) {
	tfunc, _, err := b.TfuncAndLanguage(pref, prefs...)
	return tfunc, err
}

// TfuncAndLanguage returns a TranslateFunc for the first Language that
// has a non-zero number of translations in the bundle.
//
// The returned Language matches the the first language preference that could be satisfied,
// but this may not strictly match the language of the translations used to satisfy that preference.
//
// For example, the user may request "zh". If there are no translations for "zh" but there are translations
// for "zh-cn", then the translations for "zh-cn" will be used but the returned Language will be "zh".
//
// It can parse languages from Accept-Language headers (RFC 2616),
// but it assumes weights are monotonically decreasing.
func (b *Bundle) TfuncAndLanguage(pref string, prefs ...string) (TranslateFunc, *language.Language, error) {
	lang := b.supportedLanguage(pref, prefs...)
	var err error
	if lang == nil {
		err = fmt.Errorf("no supported languages found %#v", append(prefs, pref))
	}
	return func(translationID string, args ...interface{}) string {
		return b.translate(lang, translationID, args...)
	}, lang, err
}

// supportedLanguage returns the first language which
// has a non-zero number of translations in the bundle.
func (b *Bundle) supportedLanguage(pref string, prefs ...string) *language.Language {
	lang := b.translatedLanguage(pref)
	if lang == nil {
		for _, pref := range prefs {
			lang = b.translatedLanguage(pref)
			if lang != nil {
				break
			}
		}
	}
	return lang
}

func (b *Bundle) translatedLanguage(src string) *language.Language {
	langs := language.Parse(src)
	for _, lang := range langs {
		if len(b.translations[lang.Tag]) > 0 ||
			len(b.fallbackTranslations[lang.Tag]) > 0 {
			return lang
		}
	}
	return nil
}

func (b *Bundle) translate(lang *language.Language, translationID string, args ...interface{}) string {
	if lang == nil {
		return translationID
	}

	translations := b.translations[lang.Tag]
	if translations == nil {
		translations = b.fallbackTranslations[lang.Tag]
		if translations == nil {
			return translationID
		}
	}

	translation := translations[translationID]
	if translation == nil {
		return translationID
	}

	var data interface{}
	var count interface{}
	if argc := len(args); argc > 0 {
		if isNumber(args[0]) {
			count = args[0]
			if argc > 1 {
				data = args[1]
			}
		} else {
			data = args[0]
		}
	}

	if count != nil {
		if data == nil {
			data = map[string]interface{}{"Count": count}
		} else {
			dataMap := toMap(data)
			dataMap["Count"] = count
			data = dataMap
		}
	}

	p, _ := lang.Plural(count)
	template := translation.Template(p)
	if template == nil {
		return translationID
	}

	s := template.Execute(data)
	if s == "" {
		return translationID
	}
	return s
}

func isNumber(n interface{}) bool {
	switch n.(type) {
	case int, int8, int16, int32, int64, string:
		return true
	}
	return false
}

func toMap(input interface{}) map[string]interface{} {
	if data, ok := input.(map[string]interface{}); ok {
		return data
	}
	v := reflect.ValueOf(input)
	switch v.Kind() {
	case reflect.Ptr:
		return toMap(v.Elem().Interface())
	case reflect.Struct:
		return structToMap(v)
	default:
		return nil
	}
}

// Converts the top level of a struct to a map[string]interface{}.
// Code inspired by github.com/fatih/structs.
func structToMap(v reflect.Value) map[string]interface{} {
	out := make(map[string]interface{})
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" {
			// unexported field. skip.
			continue
		}
		out[field.Name] = v.FieldByName(field.Name).Interface()
	}
	return out
}
