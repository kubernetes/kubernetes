package configuration

import (
	"fmt"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/Sirupsen/logrus"
	"gopkg.in/yaml.v2"
)

// Version is a major/minor version pair of the form Major.Minor
// Major version upgrades indicate structure or type changes
// Minor version upgrades should be strictly additive
type Version string

// MajorMinorVersion constructs a Version from its Major and Minor components
func MajorMinorVersion(major, minor uint) Version {
	return Version(fmt.Sprintf("%d.%d", major, minor))
}

func (version Version) major() (uint, error) {
	majorPart := strings.Split(string(version), ".")[0]
	major, err := strconv.ParseUint(majorPart, 10, 0)
	return uint(major), err
}

// Major returns the major version portion of a Version
func (version Version) Major() uint {
	major, _ := version.major()
	return major
}

func (version Version) minor() (uint, error) {
	minorPart := strings.Split(string(version), ".")[1]
	minor, err := strconv.ParseUint(minorPart, 10, 0)
	return uint(minor), err
}

// Minor returns the minor version portion of a Version
func (version Version) Minor() uint {
	minor, _ := version.minor()
	return minor
}

// VersionedParseInfo defines how a specific version of a configuration should
// be parsed into the current version
type VersionedParseInfo struct {
	// Version is the version which this parsing information relates to
	Version Version
	// ParseAs defines the type which a configuration file of this version
	// should be parsed into
	ParseAs reflect.Type
	// ConversionFunc defines a method for converting the parsed configuration
	// (of type ParseAs) into the current configuration version
	// Note: this method signature is very unclear with the absence of generics
	ConversionFunc func(interface{}) (interface{}, error)
}

type envVar struct {
	name  string
	value string
}

type envVars []envVar

func (a envVars) Len() int           { return len(a) }
func (a envVars) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a envVars) Less(i, j int) bool { return a[i].name < a[j].name }

// Parser can be used to parse a configuration file and environment of a defined
// version into a unified output structure
type Parser struct {
	prefix  string
	mapping map[Version]VersionedParseInfo
	env     envVars
}

// NewParser returns a *Parser with the given environment prefix which handles
// versioned configurations which match the given parseInfos
func NewParser(prefix string, parseInfos []VersionedParseInfo) *Parser {
	p := Parser{prefix: prefix, mapping: make(map[Version]VersionedParseInfo)}

	for _, parseInfo := range parseInfos {
		p.mapping[parseInfo.Version] = parseInfo
	}

	for _, env := range os.Environ() {
		envParts := strings.SplitN(env, "=", 2)
		p.env = append(p.env, envVar{envParts[0], envParts[1]})
	}

	// We must sort the environment variables lexically by name so that
	// more specific variables are applied before less specific ones
	// (i.e. REGISTRY_STORAGE before
	// REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY). This sucks, but it's a
	// lot simpler and easier to get right than unmarshalling map entries
	// into temporaries and merging with the existing entry.
	sort.Sort(p.env)

	return &p
}

// Parse reads in the given []byte and environment and writes the resulting
// configuration into the input v
//
// Environment variables may be used to override configuration parameters other
// than version, following the scheme below:
// v.Abc may be replaced by the value of PREFIX_ABC,
// v.Abc.Xyz may be replaced by the value of PREFIX_ABC_XYZ, and so forth
func (p *Parser) Parse(in []byte, v interface{}) error {
	var versionedStruct struct {
		Version Version
	}

	if err := yaml.Unmarshal(in, &versionedStruct); err != nil {
		return err
	}

	parseInfo, ok := p.mapping[versionedStruct.Version]
	if !ok {
		return fmt.Errorf("Unsupported version: %q", versionedStruct.Version)
	}

	parseAs := reflect.New(parseInfo.ParseAs)
	err := yaml.Unmarshal(in, parseAs.Interface())
	if err != nil {
		return err
	}

	for _, envVar := range p.env {
		pathStr := envVar.name
		if strings.HasPrefix(pathStr, strings.ToUpper(p.prefix)+"_") {
			path := strings.Split(pathStr, "_")

			err = p.overwriteFields(parseAs, pathStr, path[1:], envVar.value)
			if err != nil {
				return err
			}
		}
	}

	c, err := parseInfo.ConversionFunc(parseAs.Interface())
	if err != nil {
		return err
	}
	reflect.ValueOf(v).Elem().Set(reflect.Indirect(reflect.ValueOf(c)))
	return nil
}

// overwriteFields replaces configuration values with alternate values specified
// through the environment. Precondition: an empty path slice must never be
// passed in.
func (p *Parser) overwriteFields(v reflect.Value, fullpath string, path []string, payload string) error {
	for v.Kind() == reflect.Ptr {
		if v.IsNil() {
			panic("encountered nil pointer while handling environment variable " + fullpath)
		}
		v = reflect.Indirect(v)
	}
	switch v.Kind() {
	case reflect.Struct:
		return p.overwriteStruct(v, fullpath, path, payload)
	case reflect.Map:
		return p.overwriteMap(v, fullpath, path, payload)
	case reflect.Interface:
		if v.NumMethod() == 0 {
			if !v.IsNil() {
				return p.overwriteFields(v.Elem(), fullpath, path, payload)
			}
			// Interface was empty; create an implicit map
			var template map[string]interface{}
			wrappedV := reflect.MakeMap(reflect.TypeOf(template))
			v.Set(wrappedV)
			return p.overwriteMap(wrappedV, fullpath, path, payload)
		}
	}
	return nil
}

func (p *Parser) overwriteStruct(v reflect.Value, fullpath string, path []string, payload string) error {
	// Generate case-insensitive map of struct fields
	byUpperCase := make(map[string]int)
	for i := 0; i < v.NumField(); i++ {
		sf := v.Type().Field(i)
		upper := strings.ToUpper(sf.Name)
		if _, present := byUpperCase[upper]; present {
			panic(fmt.Sprintf("field name collision in configuration object: %s", sf.Name))
		}
		byUpperCase[upper] = i
	}

	fieldIndex, present := byUpperCase[path[0]]
	if !present {
		logrus.Warnf("Ignoring unrecognized environment variable %s", fullpath)
		return nil
	}
	field := v.Field(fieldIndex)
	sf := v.Type().Field(fieldIndex)

	if len(path) == 1 {
		// Env var specifies this field directly
		fieldVal := reflect.New(sf.Type)
		err := yaml.Unmarshal([]byte(payload), fieldVal.Interface())
		if err != nil {
			return err
		}
		field.Set(reflect.Indirect(fieldVal))
		return nil
	}

	// If the field is nil, must create an object
	switch sf.Type.Kind() {
	case reflect.Map:
		if field.IsNil() {
			field.Set(reflect.MakeMap(sf.Type))
		}
	case reflect.Ptr:
		if field.IsNil() {
			field.Set(reflect.New(sf.Type))
		}
	}

	err := p.overwriteFields(field, fullpath, path[1:], payload)
	if err != nil {
		return err
	}

	return nil
}

func (p *Parser) overwriteMap(m reflect.Value, fullpath string, path []string, payload string) error {
	if m.Type().Key().Kind() != reflect.String {
		// non-string keys unsupported
		logrus.Warnf("Ignoring environment variable %s involving map with non-string keys", fullpath)
		return nil
	}

	if len(path) > 1 {
		// If a matching key exists, get its value and continue the
		// overwriting process.
		for _, k := range m.MapKeys() {
			if strings.ToUpper(k.String()) == path[0] {
				mapValue := m.MapIndex(k)
				// If the existing value is nil, we want to
				// recreate it instead of using this value.
				if (mapValue.Kind() == reflect.Ptr ||
					mapValue.Kind() == reflect.Interface ||
					mapValue.Kind() == reflect.Map) &&
					mapValue.IsNil() {
					break
				}
				return p.overwriteFields(mapValue, fullpath, path[1:], payload)
			}
		}
	}

	// (Re)create this key
	var mapValue reflect.Value
	if m.Type().Elem().Kind() == reflect.Map {
		mapValue = reflect.MakeMap(m.Type().Elem())
	} else {
		mapValue = reflect.New(m.Type().Elem())
	}
	if len(path) > 1 {
		err := p.overwriteFields(mapValue, fullpath, path[1:], payload)
		if err != nil {
			return err
		}
	} else {
		err := yaml.Unmarshal([]byte(payload), mapValue.Interface())
		if err != nil {
			return err
		}
	}

	m.SetMapIndex(reflect.ValueOf(strings.ToLower(path[0])), reflect.Indirect(mapValue))

	return nil
}
