package sprig

import (
	"errors"
	"html/template"
	"os"
	"path"
	"reflect"
	"strconv"
	"strings"
	ttemplate "text/template"
	"time"

	util "github.com/Masterminds/goutils"
	"github.com/huandu/xstrings"
)

// FuncMap produces the function map.
//
// Use this to pass the functions into the template engine:
//
// 	tpl := template.New("foo").Funcs(sprig.FuncMap()))
//
func FuncMap() template.FuncMap {
	return HtmlFuncMap()
}

// HermeticTxtFuncMap returns a 'text/template'.FuncMap with only repeatable functions.
func HermeticTxtFuncMap() ttemplate.FuncMap {
	r := TxtFuncMap()
	for _, name := range nonhermeticFunctions {
		delete(r, name)
	}
	return r
}

// HermeticHtmlFuncMap returns an 'html/template'.Funcmap with only repeatable functions.
func HermeticHtmlFuncMap() template.FuncMap {
	r := HtmlFuncMap()
	for _, name := range nonhermeticFunctions {
		delete(r, name)
	}
	return r
}

// TxtFuncMap returns a 'text/template'.FuncMap
func TxtFuncMap() ttemplate.FuncMap {
	return ttemplate.FuncMap(GenericFuncMap())
}

// HtmlFuncMap returns an 'html/template'.Funcmap
func HtmlFuncMap() template.FuncMap {
	return template.FuncMap(GenericFuncMap())
}

// GenericFuncMap returns a copy of the basic function map as a map[string]interface{}.
func GenericFuncMap() map[string]interface{} {
	gfm := make(map[string]interface{}, len(genericMap))
	for k, v := range genericMap {
		gfm[k] = v
	}
	return gfm
}

// These functions are not guaranteed to evaluate to the same result for given input, because they
// refer to the environment or global state.
var nonhermeticFunctions = []string{
	// Date functions
	"date",
	"date_in_zone",
	"date_modify",
	"now",
	"htmlDate",
	"htmlDateInZone",
	"dateInZone",
	"dateModify",

	// Strings
	"randAlphaNum",
	"randAlpha",
	"randAscii",
	"randNumeric",
	"uuidv4",

	// OS
	"env",
	"expandenv",

	// Network
	"getHostByName",
}

var genericMap = map[string]interface{}{
	"hello": func() string { return "Hello!" },

	// Date functions
	"ago":              dateAgo,
	"date":             date,
	"date_in_zone":     dateInZone,
	"date_modify":      dateModify,
	"dateInZone":       dateInZone,
	"dateModify":       dateModify,
	"durationRound":    durationRound,
	"htmlDate":         htmlDate,
	"htmlDateInZone":   htmlDateInZone,
	"must_date_modify": mustDateModify,
	"mustDateModify":   mustDateModify,
	"mustToDate":       mustToDate,
	"now":              func() time.Time { return time.Now() },
	"toDate":           toDate,
	"unixEpoch":        unixEpoch,

	// Strings
	"abbrev":     abbrev,
	"abbrevboth": abbrevboth,
	"trunc":      trunc,
	"trim":       strings.TrimSpace,
	"upper":      strings.ToUpper,
	"lower":      strings.ToLower,
	"title":      strings.Title,
	"untitle":    untitle,
	"substr":     substring,
	// Switch order so that "foo" | repeat 5
	"repeat": func(count int, str string) string { return strings.Repeat(str, count) },
	// Deprecated: Use trimAll.
	"trimall": func(a, b string) string { return strings.Trim(b, a) },
	// Switch order so that "$foo" | trimall "$"
	"trimAll":      func(a, b string) string { return strings.Trim(b, a) },
	"trimSuffix":   func(a, b string) string { return strings.TrimSuffix(b, a) },
	"trimPrefix":   func(a, b string) string { return strings.TrimPrefix(b, a) },
	"nospace":      util.DeleteWhiteSpace,
	"initials":     initials,
	"randAlphaNum": randAlphaNumeric,
	"randAlpha":    randAlpha,
	"randAscii":    randAscii,
	"randNumeric":  randNumeric,
	"swapcase":     util.SwapCase,
	"shuffle":      xstrings.Shuffle,
	"snakecase":    xstrings.ToSnakeCase,
	"camelcase":    xstrings.ToCamelCase,
	"kebabcase":    xstrings.ToKebabCase,
	"wrap":         func(l int, s string) string { return util.Wrap(s, l) },
	"wrapWith":     func(l int, sep, str string) string { return util.WrapCustom(str, l, sep, true) },
	// Switch order so that "foobar" | contains "foo"
	"contains":   func(substr string, str string) bool { return strings.Contains(str, substr) },
	"hasPrefix":  func(substr string, str string) bool { return strings.HasPrefix(str, substr) },
	"hasSuffix":  func(substr string, str string) bool { return strings.HasSuffix(str, substr) },
	"quote":      quote,
	"squote":     squote,
	"cat":        cat,
	"indent":     indent,
	"nindent":    nindent,
	"replace":    replace,
	"plural":     plural,
	"sha1sum":    sha1sum,
	"sha256sum":  sha256sum,
	"adler32sum": adler32sum,
	"toString":   strval,

	// Wrap Atoi to stop errors.
	"atoi":      func(a string) int { i, _ := strconv.Atoi(a); return i },
	"int64":     toInt64,
	"int":       toInt,
	"float64":   toFloat64,
	"toDecimal": toDecimal,

	//"gt": func(a, b int) bool {return a > b},
	//"gte": func(a, b int) bool {return a >= b},
	//"lt": func(a, b int) bool {return a < b},
	//"lte": func(a, b int) bool {return a <= b},

	// split "/" foo/bar returns map[int]string{0: foo, 1: bar}
	"split":     split,
	"splitList": func(sep, orig string) []string { return strings.Split(orig, sep) },
	// splitn "/" foo/bar/fuu returns map[int]string{0: foo, 1: bar/fuu}
	"splitn":    splitn,
	"toStrings": strslice,

	"until":     until,
	"untilStep": untilStep,

	// VERY basic arithmetic.
	"add1": func(i interface{}) int64 { return toInt64(i) + 1 },
	"add": func(i ...interface{}) int64 {
		var a int64 = 0
		for _, b := range i {
			a += toInt64(b)
		}
		return a
	},
	"sub": func(a, b interface{}) int64 { return toInt64(a) - toInt64(b) },
	"div": func(a, b interface{}) int64 { return toInt64(a) / toInt64(b) },
	"mod": func(a, b interface{}) int64 { return toInt64(a) % toInt64(b) },
	"mul": func(a interface{}, v ...interface{}) int64 {
		val := toInt64(a)
		for _, b := range v {
			val = val * toInt64(b)
		}
		return val
	},
	"biggest": max,
	"max":     max,
	"min":     min,
	"ceil":    ceil,
	"floor":   floor,
	"round":   round,

	// string slices. Note that we reverse the order b/c that's better
	// for template processing.
	"join":      join,
	"sortAlpha": sortAlpha,

	// Defaults
	"default":          dfault,
	"empty":            empty,
	"coalesce":         coalesce,
	"compact":          compact,
	"mustCompact":      mustCompact,
	"toJson":           toJson,
	"toPrettyJson":     toPrettyJson,
	"toRawJson":        toRawJson,
	"mustToJson":       mustToJson,
	"mustToPrettyJson": mustToPrettyJson,
	"mustToRawJson":    mustToRawJson,
	"ternary":          ternary,
	"deepCopy":         deepCopy,
	"mustDeepCopy":     mustDeepCopy,

	// Reflection
	"typeOf":     typeOf,
	"typeIs":     typeIs,
	"typeIsLike": typeIsLike,
	"kindOf":     kindOf,
	"kindIs":     kindIs,
	"deepEqual":  reflect.DeepEqual,

	// OS:
	"env":       func(s string) string { return os.Getenv(s) },
	"expandenv": func(s string) string { return os.ExpandEnv(s) },

	// Network:
	"getHostByName": getHostByName,

	// File Paths:
	"base":  path.Base,
	"dir":   path.Dir,
	"clean": path.Clean,
	"ext":   path.Ext,
	"isAbs": path.IsAbs,

	// Encoding:
	"b64enc": base64encode,
	"b64dec": base64decode,
	"b32enc": base32encode,
	"b32dec": base32decode,

	// Data Structures:
	"tuple":              list, // FIXME: with the addition of append/prepend these are no longer immutable.
	"list":               list,
	"dict":               dict,
	"get":                get,
	"set":                set,
	"unset":              unset,
	"hasKey":             hasKey,
	"pluck":              pluck,
	"keys":               keys,
	"pick":               pick,
	"omit":               omit,
	"merge":              merge,
	"mergeOverwrite":     mergeOverwrite,
	"mustMerge":          mustMerge,
	"mustMergeOverwrite": mustMergeOverwrite,
	"values":             values,

	"append": push, "push": push,
	"mustAppend": mustPush, "mustPush": mustPush,
	"prepend":     prepend,
	"mustPrepend": mustPrepend,
	"first":       first,
	"mustFirst":   mustFirst,
	"rest":        rest,
	"mustRest":    mustRest,
	"last":        last,
	"mustLast":    mustLast,
	"initial":     initial,
	"mustInitial": mustInitial,
	"reverse":     reverse,
	"mustReverse": mustReverse,
	"uniq":        uniq,
	"mustUniq":    mustUniq,
	"without":     without,
	"mustWithout": mustWithout,
	"has":         has,
	"mustHas":     mustHas,
	"slice":       slice,
	"mustSlice":   mustSlice,
	"concat":      concat,

	// Crypto:
	"genPrivateKey":     generatePrivateKey,
	"derivePassword":    derivePassword,
	"buildCustomCert":   buildCustomCertificate,
	"genCA":             generateCertificateAuthority,
	"genSelfSignedCert": generateSelfSignedCertificate,
	"genSignedCert":     generateSignedCertificate,
	"encryptAES":        encryptAES,
	"decryptAES":        decryptAES,

	// UUIDs:
	"uuidv4": uuidv4,

	// SemVer:
	"semver":        semver,
	"semverCompare": semverCompare,

	// Flow Control:
	"fail": func(msg string) (string, error) { return "", errors.New(msg) },

	// Regex
	"regexMatch":                 regexMatch,
	"mustRegexMatch":             mustRegexMatch,
	"regexFindAll":               regexFindAll,
	"mustRegexFindAll":           mustRegexFindAll,
	"regexFind":                  regexFind,
	"mustRegexFind":              mustRegexFind,
	"regexReplaceAll":            regexReplaceAll,
	"mustRegexReplaceAll":        mustRegexReplaceAll,
	"regexReplaceAllLiteral":     regexReplaceAllLiteral,
	"mustRegexReplaceAllLiteral": mustRegexReplaceAllLiteral,
	"regexSplit":                 regexSplit,
	"mustRegexSplit":             mustRegexSplit,

	// URLs:
	"urlParse": urlParse,
	"urlJoin":  urlJoin,
}
