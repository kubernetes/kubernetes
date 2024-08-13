package sprig

import (
	"errors"
	"html/template"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	ttemplate "text/template"
	"time"
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
	"randBytes",
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
	"duration":         duration,
	"durationRound":    durationRound,
	"htmlDate":         htmlDate,
	"htmlDateInZone":   htmlDateInZone,
	"must_date_modify": mustDateModify,
	"mustDateModify":   mustDateModify,
	"mustToDate":       mustToDate,
	"now":              time.Now,
	"toDate":           toDate,
	"unixEpoch":        unixEpoch,

	// Strings
	"trunc":  trunc,
	"trim":   strings.TrimSpace,
	"upper":  strings.ToUpper,
	"lower":  strings.ToLower,
	"title":  strings.Title,
	"substr": substring,
	// Switch order so that "foo" | repeat 5
	"repeat": func(count int, str string) string { return strings.Repeat(str, count) },
	// Deprecated: Use trimAll.
	"trimall": func(a, b string) string { return strings.Trim(b, a) },
	// Switch order so that "$foo" | trimall "$"
	"trimAll":    func(a, b string) string { return strings.Trim(b, a) },
	"trimSuffix": func(a, b string) string { return strings.TrimSuffix(b, a) },
	"trimPrefix": func(a, b string) string { return strings.TrimPrefix(b, a) },
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
	"seq":       seq,
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
	"randInt": func(min, max int) int { return rand.Intn(max-min) + min },
	"biggest": max,
	"max":     max,
	"min":     min,
	"maxf":    maxf,
	"minf":    minf,
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
	"all":              all,
	"any":              any,
	"compact":          compact,
	"mustCompact":      mustCompact,
	"fromJson":         fromJson,
	"toJson":           toJson,
	"toPrettyJson":     toPrettyJson,
	"toRawJson":        toRawJson,
	"mustFromJson":     mustFromJson,
	"mustToJson":       mustToJson,
	"mustToPrettyJson": mustToPrettyJson,
	"mustToRawJson":    mustToRawJson,
	"ternary":          ternary,

	// Reflection
	"typeOf":     typeOf,
	"typeIs":     typeIs,
	"typeIsLike": typeIsLike,
	"kindOf":     kindOf,
	"kindIs":     kindIs,
	"deepEqual":  reflect.DeepEqual,

	// OS:
	"env":       os.Getenv,
	"expandenv": os.ExpandEnv,

	// Network:
	"getHostByName": getHostByName,

	// Paths:
	"base":  path.Base,
	"dir":   path.Dir,
	"clean": path.Clean,
	"ext":   path.Ext,
	"isAbs": path.IsAbs,

	// Filepaths:
	"osBase":  filepath.Base,
	"osClean": filepath.Clean,
	"osDir":   filepath.Dir,
	"osExt":   filepath.Ext,
	"osIsAbs": filepath.IsAbs,

	// Encoding:
	"b64enc": base64encode,
	"b64dec": base64decode,
	"b32enc": base32encode,
	"b32dec": base32decode,

	// Data Structures:
	"tuple":  list, // FIXME: with the addition of append/prepend these are no longer immutable.
	"list":   list,
	"dict":   dict,
	"get":    get,
	"set":    set,
	"unset":  unset,
	"hasKey": hasKey,
	"pluck":  pluck,
	"keys":   keys,
	"pick":   pick,
	"omit":   omit,
	"values": values,

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
	"dig":         dig,
	"chunk":       chunk,
	"mustChunk":   mustChunk,

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
	"regexQuoteMeta":             regexQuoteMeta,

	// URLs:
	"urlParse": urlParse,
	"urlJoin":  urlJoin,
}
