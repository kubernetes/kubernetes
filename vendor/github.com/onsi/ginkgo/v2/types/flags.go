package types

import (
	"flag"
	"fmt"
	"io"
	"reflect"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2/formatter"
)

type GinkgoFlag struct {
	Name       string
	KeyPath    string
	SectionKey string

	Usage             string
	UsageArgument     string
	UsageDefaultValue string

	DeprecatedName    string
	DeprecatedDocLink string
	DeprecatedVersion string

	ExportAs     string
	AlwaysExport bool
}

type GinkgoFlags []GinkgoFlag

func (f GinkgoFlags) CopyAppend(flags ...GinkgoFlag) GinkgoFlags {
	out := GinkgoFlags{}
	out = append(out, f...)
	out = append(out, flags...)
	return out
}

func (f GinkgoFlags) WithPrefix(prefix string) GinkgoFlags {
	if prefix == "" {
		return f
	}
	out := GinkgoFlags{}
	for _, flag := range f {
		if flag.Name != "" {
			flag.Name = prefix + "." + flag.Name
		}
		if flag.DeprecatedName != "" {
			flag.DeprecatedName = prefix + "." + flag.DeprecatedName
		}
		if flag.ExportAs != "" {
			flag.ExportAs = prefix + "." + flag.ExportAs
		}
		out = append(out, flag)
	}
	return out
}

func (f GinkgoFlags) SubsetWithNames(names ...string) GinkgoFlags {
	out := GinkgoFlags{}
	for _, flag := range f {
		for _, name := range names {
			if flag.Name == name {
				out = append(out, flag)
				break
			}
		}
	}
	return out
}

type GinkgoFlagSection struct {
	Key         string
	Style       string
	Succinct    bool
	Heading     string
	Description string
}

type GinkgoFlagSections []GinkgoFlagSection

func (gfs GinkgoFlagSections) Lookup(key string) (GinkgoFlagSection, bool) {
	for _, section := range gfs {
		if section.Key == key {
			return section, true
		}
	}

	return GinkgoFlagSection{}, false
}

type GinkgoFlagSet struct {
	flags    GinkgoFlags
	bindings any

	sections            GinkgoFlagSections
	extraGoFlagsSection GinkgoFlagSection

	flagSet *flag.FlagSet
}

// Call NewGinkgoFlagSet to create GinkgoFlagSet that creates and binds to it's own *flag.FlagSet
func NewGinkgoFlagSet(flags GinkgoFlags, bindings any, sections GinkgoFlagSections) (GinkgoFlagSet, error) {
	return bindFlagSet(GinkgoFlagSet{
		flags:    flags,
		bindings: bindings,
		sections: sections,
	}, nil)
}

// Call NewGinkgoFlagSet to create GinkgoFlagSet that extends an existing *flag.FlagSet
func NewAttachedGinkgoFlagSet(flagSet *flag.FlagSet, flags GinkgoFlags, bindings any, sections GinkgoFlagSections, extraGoFlagsSection GinkgoFlagSection) (GinkgoFlagSet, error) {
	return bindFlagSet(GinkgoFlagSet{
		flags:               flags,
		bindings:            bindings,
		sections:            sections,
		extraGoFlagsSection: extraGoFlagsSection,
	}, flagSet)
}

func bindFlagSet(f GinkgoFlagSet, flagSet *flag.FlagSet) (GinkgoFlagSet, error) {
	if flagSet == nil {
		f.flagSet = flag.NewFlagSet("", flag.ContinueOnError)
		//suppress all output as Ginkgo is responsible for formatting usage
		f.flagSet.SetOutput(io.Discard)
	} else {
		f.flagSet = flagSet
		//we're piggybacking on an existing flagset (typically go test) so we have limited control
		//on user feedback
		f.flagSet.Usage = f.substituteUsage
	}

	for _, flag := range f.flags {
		name := flag.Name

		deprecatedUsage := "[DEPRECATED]"
		deprecatedName := flag.DeprecatedName
		if name != "" {
			deprecatedUsage = fmt.Sprintf("[DEPRECATED] use --%s instead", name)
		} else if flag.Usage != "" {
			deprecatedUsage += " " + flag.Usage
		}

		value, ok := valueAtKeyPath(f.bindings, flag.KeyPath)
		if !ok {
			return GinkgoFlagSet{}, fmt.Errorf("could not load KeyPath: %s", flag.KeyPath)
		}

		iface, addr := value.Interface(), value.Addr().Interface()

		switch value.Type() {
		case reflect.TypeOf(string("")):
			if name != "" {
				f.flagSet.StringVar(addr.(*string), name, iface.(string), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.StringVar(addr.(*string), deprecatedName, iface.(string), deprecatedUsage)
			}
		case reflect.TypeOf(int64(0)):
			if name != "" {
				f.flagSet.Int64Var(addr.(*int64), name, iface.(int64), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.Int64Var(addr.(*int64), deprecatedName, iface.(int64), deprecatedUsage)
			}
		case reflect.TypeOf(float64(0)):
			if name != "" {
				f.flagSet.Float64Var(addr.(*float64), name, iface.(float64), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.Float64Var(addr.(*float64), deprecatedName, iface.(float64), deprecatedUsage)
			}
		case reflect.TypeOf(int(0)):
			if name != "" {
				f.flagSet.IntVar(addr.(*int), name, iface.(int), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.IntVar(addr.(*int), deprecatedName, iface.(int), deprecatedUsage)
			}
		case reflect.TypeOf(bool(true)):
			if name != "" {
				f.flagSet.BoolVar(addr.(*bool), name, iface.(bool), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.BoolVar(addr.(*bool), deprecatedName, iface.(bool), deprecatedUsage)
			}
		case reflect.TypeOf(time.Duration(0)):
			if name != "" {
				f.flagSet.DurationVar(addr.(*time.Duration), name, iface.(time.Duration), flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.DurationVar(addr.(*time.Duration), deprecatedName, iface.(time.Duration), deprecatedUsage)
			}

		case reflect.TypeOf([]string{}):
			if name != "" {
				f.flagSet.Var(stringSliceVar{value}, name, flag.Usage)
			}
			if deprecatedName != "" {
				f.flagSet.Var(stringSliceVar{value}, deprecatedName, deprecatedUsage)
			}
		default:
			return GinkgoFlagSet{}, fmt.Errorf("unsupported type %T", iface)
		}
	}

	return f, nil
}

func (f GinkgoFlagSet) IsZero() bool {
	return f.flagSet == nil
}

func (f GinkgoFlagSet) WasSet(name string) bool {
	found := false
	f.flagSet.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})

	return found
}

func (f GinkgoFlagSet) Lookup(name string) *flag.Flag {
	return f.flagSet.Lookup(name)
}

func (f GinkgoFlagSet) Parse(args []string) ([]string, error) {
	if f.IsZero() {
		return args, nil
	}
	err := f.flagSet.Parse(args)
	if err != nil {
		return []string{}, err
	}
	return f.flagSet.Args(), nil
}

func (f GinkgoFlagSet) ValidateDeprecations(deprecationTracker *DeprecationTracker) {
	if f.IsZero() {
		return
	}
	f.flagSet.Visit(func(flag *flag.Flag) {
		for _, ginkgoFlag := range f.flags {
			if ginkgoFlag.DeprecatedName != "" && strings.HasSuffix(flag.Name, ginkgoFlag.DeprecatedName) {
				message := fmt.Sprintf("--%s is deprecated", ginkgoFlag.DeprecatedName)
				if ginkgoFlag.Name != "" {
					message = fmt.Sprintf("--%s is deprecated, use --%s instead", ginkgoFlag.DeprecatedName, ginkgoFlag.Name)
				} else if ginkgoFlag.Usage != "" {
					message += " " + ginkgoFlag.Usage
				}

				deprecationTracker.TrackDeprecation(Deprecation{
					Message: message,
					DocLink: ginkgoFlag.DeprecatedDocLink,
					Version: ginkgoFlag.DeprecatedVersion,
				})
			}
		}
	})
}

func (f GinkgoFlagSet) Usage() string {
	if f.IsZero() {
		return ""
	}
	groupedFlags := map[GinkgoFlagSection]GinkgoFlags{}
	ungroupedFlags := GinkgoFlags{}
	managedFlags := map[string]bool{}
	extraGoFlags := []*flag.Flag{}

	for _, flag := range f.flags {
		managedFlags[flag.Name] = true
		managedFlags[flag.DeprecatedName] = true

		if flag.Name == "" {
			continue
		}

		section, ok := f.sections.Lookup(flag.SectionKey)
		if ok {
			groupedFlags[section] = append(groupedFlags[section], flag)
		} else {
			ungroupedFlags = append(ungroupedFlags, flag)
		}
	}

	f.flagSet.VisitAll(func(flag *flag.Flag) {
		if !managedFlags[flag.Name] {
			extraGoFlags = append(extraGoFlags, flag)
		}
	})

	out := ""
	for _, section := range f.sections {
		flags := groupedFlags[section]
		if len(flags) == 0 {
			continue
		}
		out += f.usageForSection(section)
		if section.Succinct {
			succinctFlags := []string{}
			for _, flag := range flags {
				if flag.Name != "" {
					succinctFlags = append(succinctFlags, fmt.Sprintf("--%s", flag.Name))
				}
			}
			out += formatter.Fiw(1, formatter.COLS, section.Style+strings.Join(succinctFlags, ", ")+"{{/}}\n")
		} else {
			for _, flag := range flags {
				out += f.usageForFlag(flag, section.Style)
			}
		}
		out += "\n"
	}
	if len(ungroupedFlags) > 0 {
		for _, flag := range ungroupedFlags {
			out += f.usageForFlag(flag, "")
		}
		out += "\n"
	}
	if len(extraGoFlags) > 0 {
		out += f.usageForSection(f.extraGoFlagsSection)
		for _, goFlag := range extraGoFlags {
			out += f.usageForGoFlag(goFlag)
		}
	}

	return out
}

func (f GinkgoFlagSet) substituteUsage() {
	fmt.Fprintln(f.flagSet.Output(), f.Usage())
}

func valueAtKeyPath(root any, keyPath string) (reflect.Value, bool) {
	if len(keyPath) == 0 {
		return reflect.Value{}, false
	}

	val := reflect.ValueOf(root)
	components := strings.Split(keyPath, ".")
	for _, component := range components {
		val = reflect.Indirect(val)
		switch val.Kind() {
		case reflect.Map:
			val = val.MapIndex(reflect.ValueOf(component))
			if val.Kind() == reflect.Interface {
				val = reflect.ValueOf(val.Interface())
			}
		case reflect.Struct:
			val = val.FieldByName(component)
		default:
			return reflect.Value{}, false
		}
		if (val == reflect.Value{}) {
			return reflect.Value{}, false
		}
	}

	return val, true
}

func (f GinkgoFlagSet) usageForSection(section GinkgoFlagSection) string {
	out := formatter.F(section.Style + "{{bold}}{{underline}}" + section.Heading + "{{/}}\n")
	if section.Description != "" {
		out += formatter.Fiw(0, formatter.COLS, section.Description+"\n")
	}
	return out
}

func (f GinkgoFlagSet) usageForFlag(flag GinkgoFlag, style string) string {
	argument := flag.UsageArgument
	defValue := flag.UsageDefaultValue
	if argument == "" {
		value, _ := valueAtKeyPath(f.bindings, flag.KeyPath)
		switch value.Type() {
		case reflect.TypeOf(string("")):
			argument = "string"
		case reflect.TypeOf(int64(0)), reflect.TypeOf(int(0)):
			argument = "int"
		case reflect.TypeOf(time.Duration(0)):
			argument = "duration"
		case reflect.TypeOf(float64(0)):
			argument = "float"
		case reflect.TypeOf([]string{}):
			argument = "string"
		}
	}
	if argument != "" {
		argument = "[" + argument + "] "
	}
	if defValue != "" {
		defValue = fmt.Sprintf("(default: %s)", defValue)
	}
	hyphens := "--"
	if len(flag.Name) == 1 {
		hyphens = "-"
	}

	out := formatter.Fi(1, style+"%s%s{{/}} %s{{gray}}%s{{/}}\n", hyphens, flag.Name, argument, defValue)
	out += formatter.Fiw(2, formatter.COLS, "{{light-gray}}%s{{/}}\n", flag.Usage)
	return out
}

func (f GinkgoFlagSet) usageForGoFlag(goFlag *flag.Flag) string {
	//Taken directly from the flag package
	out := fmt.Sprintf("  -%s", goFlag.Name)
	name, usage := flag.UnquoteUsage(goFlag)
	if len(name) > 0 {
		out += " " + name
	}
	if len(out) <= 4 {
		out += "\t"
	} else {
		out += "\n    \t"
	}
	out += strings.ReplaceAll(usage, "\n", "\n    \t")
	out += "\n"
	return out
}

type stringSliceVar struct {
	slice reflect.Value
}

func (ssv stringSliceVar) String() string { return "" }
func (ssv stringSliceVar) Set(s string) error {
	ssv.slice.Set(reflect.AppendSlice(ssv.slice, reflect.ValueOf([]string{s})))
	return nil
}

// given a set of GinkgoFlags and bindings, generate flag arguments suitable to be passed to an application with that set of flags configured.
func GenerateFlagArgs(flags GinkgoFlags, bindings any) ([]string, error) {
	result := []string{}
	for _, flag := range flags {
		name := flag.ExportAs
		if name == "" {
			name = flag.Name
		}
		if name == "" {
			continue
		}

		value, ok := valueAtKeyPath(bindings, flag.KeyPath)
		if !ok {
			return []string{}, fmt.Errorf("could not load KeyPath: %s", flag.KeyPath)
		}

		iface := value.Interface()
		switch value.Type() {
		case reflect.TypeOf(string("")):
			if iface.(string) != "" || flag.AlwaysExport {
				result = append(result, fmt.Sprintf("--%s=%s", name, iface))
			}
		case reflect.TypeOf(int64(0)):
			if iface.(int64) != 0 || flag.AlwaysExport {
				result = append(result, fmt.Sprintf("--%s=%d", name, iface))
			}
		case reflect.TypeOf(float64(0)):
			if iface.(float64) != 0 || flag.AlwaysExport {
				result = append(result, fmt.Sprintf("--%s=%f", name, iface))
			}
		case reflect.TypeOf(int(0)):
			if iface.(int) != 0 || flag.AlwaysExport {
				result = append(result, fmt.Sprintf("--%s=%d", name, iface))
			}
		case reflect.TypeOf(bool(true)):
			if iface.(bool) {
				result = append(result, fmt.Sprintf("--%s", name))
			}
		case reflect.TypeOf(time.Duration(0)):
			if iface.(time.Duration) != time.Duration(0) || flag.AlwaysExport {
				result = append(result, fmt.Sprintf("--%s=%s", name, iface))
			}

		case reflect.TypeOf([]string{}):
			strings := iface.([]string)
			for _, s := range strings {
				result = append(result, fmt.Sprintf("--%s=%s", name, s))
			}
		default:
			return []string{}, fmt.Errorf("unsupported type %T", iface)
		}
	}

	return result, nil
}
