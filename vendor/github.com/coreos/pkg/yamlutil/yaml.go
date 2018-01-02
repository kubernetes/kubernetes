package yamlutil

import (
	"flag"
	"fmt"
	"strings"

	"gopkg.in/yaml.v1"
)

// SetFlagsFromYaml goes through all registered flags in the given flagset,
// and if they are not already set it attempts to set their values from
// the YAML config. It will use the key REPLACE(UPPERCASE(flagname), '-', '_')
func SetFlagsFromYaml(fs *flag.FlagSet, rawYaml []byte) (err error) {
	conf := make(map[string]string)
	if err = yaml.Unmarshal(rawYaml, conf); err != nil {
		return
	}
	alreadySet := map[string]struct{}{}
	fs.Visit(func(f *flag.Flag) {
		alreadySet[f.Name] = struct{}{}
	})

	errs := make([]error, 0)
	fs.VisitAll(func(f *flag.Flag) {
		if f.Name == "" {
			return
		}
		if _, ok := alreadySet[f.Name]; ok {
			return
		}
		tag := strings.Replace(strings.ToUpper(f.Name), "-", "_", -1)
		val, ok := conf[tag]
		if !ok {
			return
		}
		if serr := fs.Set(f.Name, val); serr != nil {
			errs = append(errs, fmt.Errorf("invalid value %q for %s: %v", val, tag, serr))
		}
	})
	if len(errs) != 0 {
		err = ErrorSlice(errs)
	}
	return
}

type ErrorSlice []error

func (e ErrorSlice) Error() string {
	s := ""
	for _, err := range e {
		s += ", " + err.Error()
	}
	return "Errors: " + s
}
