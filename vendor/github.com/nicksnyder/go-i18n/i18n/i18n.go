// Package i18n supports string translations with variable substitution and CLDR pluralization.
// It is intended to be used in conjunction with the goi18n command, although that is not strictly required.
//
// Initialization
//
// Your Go program should load translations during its initialization.
//     i18n.MustLoadTranslationFile("path/to/fr-FR.all.json")
// If your translations are in a file format not supported by (Must)?LoadTranslationFile,
// then you can use the AddTranslation function to manually add translations.
//
// Fetching a translation
//
// Use Tfunc or MustTfunc to fetch a TranslateFunc that will return the translated string for a specific language.
//     func handleRequest(w http.ResponseWriter, r *http.Request) {
//         cookieLang := r.Cookie("lang")
//         acceptLang := r.Header.Get("Accept-Language")
//         defaultLang = "en-US"  // known valid language
//         T, err := i18n.Tfunc(cookieLang, acceptLang, defaultLang)
//         fmt.Println(T("Hello world"))
//     }
//
// Usually it is a good idea to identify strings by a generic id rather than the English translation,
// but the rest of this documentation will continue to use the English translation for readability.
//     T("Hello world")     // ok
//     T("programGreeting") // better!
//
// Variables
//
// TranslateFunc supports strings that have variables using the text/template syntax.
//     T("Hello {{.Person}}", map[string]interface{}{
//         "Person": "Bob",
//     })
//
// Pluralization
//
// TranslateFunc supports the pluralization of strings using the CLDR pluralization rules defined here:
// http://www.unicode.org/cldr/charts/latest/supplemental/language_plural_rules.html
//     T("You have {{.Count}} unread emails.", 2)
//     T("I am {{.Count}} meters tall.", "1.7")
//
// Plural strings may also have variables.
//     T("{{.Person}} has {{.Count}} unread emails", 2, map[string]interface{}{
//         "Person": "Bob",
//     })
//
// Sentences with multiple plural components can be supported with nesting.
//     T("{{.Person}} has {{.Count}} unread emails in the past {{.Timeframe}}.", 3, map[string]interface{}{
//         "Person":    "Bob",
//         "Timeframe": T("{{.Count}} days", 2),
//     })
//
// Templates
//
// You can use the .Funcs() method of a text/template or html/template to register a TranslateFunc
// for usage inside of that template.
package i18n

import (
	"github.com/nicksnyder/go-i18n/i18n/bundle"
	"github.com/nicksnyder/go-i18n/i18n/language"
	"github.com/nicksnyder/go-i18n/i18n/translation"
)

// TranslateFunc returns the translation of the string identified by translationID.
//
// If there is no translation for translationID, then the translationID itself is returned.
// This makes it easy to identify missing translations in your app.
//
// If translationID is a non-plural form, then the first variadic argument may be a map[string]interface{}
// or struct that contains template data.
//
// If translationID is a plural form, then the first variadic argument must be an integer type
// (int, int8, int16, int32, int64) or a float formatted as a string (e.g. "123.45").
// The second variadic argument may be a map[string]interface{} or struct that contains template data.
type TranslateFunc func(translationID string, args ...interface{}) string

// IdentityTfunc returns a TranslateFunc that always returns the translationID passed to it.
//
// It is a useful placeholder when parsing a text/template or html/template
// before the actual Tfunc is available.
func IdentityTfunc() TranslateFunc {
	return func(translationID string, args ...interface{}) string {
		return translationID
	}
}

var defaultBundle = bundle.New()

// MustLoadTranslationFile is similar to LoadTranslationFile
// except it panics if an error happens.
func MustLoadTranslationFile(filename string) {
	defaultBundle.MustLoadTranslationFile(filename)
}

// LoadTranslationFile loads the translations from filename into memory.
//
// The language that the translations are associated with is parsed from the filename (e.g. en-US.json).
//
// Generally you should load translation files once during your program's initialization.
func LoadTranslationFile(filename string) error {
	return defaultBundle.LoadTranslationFile(filename)
}

// ParseTranslationFileBytes is similar to LoadTranslationFile except it parses the bytes in buf.
//
// It is useful for parsing translation files embedded with go-bindata.
func ParseTranslationFileBytes(filename string, buf []byte) error {
	return defaultBundle.ParseTranslationFileBytes(filename, buf)
}

// AddTranslation adds translations for a language.
//
// It is useful if your translations are in a format not supported by LoadTranslationFile.
func AddTranslation(lang *language.Language, translations ...translation.Translation) {
	defaultBundle.AddTranslation(lang, translations...)
}

// LanguageTags returns the tags of all languages that have been added.
func LanguageTags() []string {
	return defaultBundle.LanguageTags()
}

// LanguageTranslationIDs returns the ids of all translations that have been added for a given language.
func LanguageTranslationIDs(languageTag string) []string {
	return defaultBundle.LanguageTranslationIDs(languageTag)
}

// MustTfunc is similar to Tfunc except it panics if an error happens.
func MustTfunc(languageSource string, languageSources ...string) TranslateFunc {
	return TranslateFunc(defaultBundle.MustTfunc(languageSource, languageSources...))
}

// Tfunc returns a TranslateFunc that will be bound to the first language which
// has a non-zero number of translations.
//
// It can parse languages from Accept-Language headers (RFC 2616).
func Tfunc(languageSource string, languageSources ...string) (TranslateFunc, error) {
	tfunc, err := defaultBundle.Tfunc(languageSource, languageSources...)
	return TranslateFunc(tfunc), err
}

// MustTfuncAndLanguage is similar to TfuncAndLanguage except it panics if an error happens.
func MustTfuncAndLanguage(languageSource string, languageSources ...string) (TranslateFunc, *language.Language) {
	tfunc, lang := defaultBundle.MustTfuncAndLanguage(languageSource, languageSources...)
	return TranslateFunc(tfunc), lang
}

// TfuncAndLanguage is similar to Tfunc except it also returns the language which TranslateFunc is bound to.
func TfuncAndLanguage(languageSource string, languageSources ...string) (TranslateFunc, *language.Language, error) {
	tfunc, lang, err := defaultBundle.TfuncAndLanguage(languageSource, languageSources...)
	return TranslateFunc(tfunc), lang, err
}
