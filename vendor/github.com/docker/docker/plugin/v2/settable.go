package v2

import (
	"errors"
	"fmt"
	"strings"
)

type settable struct {
	name  string
	field string
	value string
}

var (
	allowedSettableFieldsEnv     = []string{"value"}
	allowedSettableFieldsArgs    = []string{"value"}
	allowedSettableFieldsDevices = []string{"path"}
	allowedSettableFieldsMounts  = []string{"source"}

	errMultipleFields = errors.New("multiple fields are settable, one must be specified")
	errInvalidFormat  = errors.New("invalid format, must be <name>[.<field>][=<value>]")
)

func newSettables(args []string) ([]settable, error) {
	sets := make([]settable, 0, len(args))
	for _, arg := range args {
		set, err := newSettable(arg)
		if err != nil {
			return nil, err
		}
		sets = append(sets, set)
	}
	return sets, nil
}

func newSettable(arg string) (settable, error) {
	var set settable
	if i := strings.Index(arg, "="); i == 0 {
		return set, errInvalidFormat
	} else if i < 0 {
		set.name = arg
	} else {
		set.name = arg[:i]
		set.value = arg[i+1:]
	}

	if i := strings.LastIndex(set.name, "."); i > 0 {
		set.field = set.name[i+1:]
		set.name = arg[:i]
	}

	return set, nil
}

// prettyName return name.field if there is a field, otherwise name.
func (set *settable) prettyName() string {
	if set.field != "" {
		return fmt.Sprintf("%s.%s", set.name, set.field)
	}
	return set.name
}

func (set *settable) isSettable(allowedSettableFields []string, settable []string) (bool, error) {
	if set.field == "" {
		if len(settable) == 1 {
			// if field is not specified and there only one settable, default to it.
			set.field = settable[0]
		} else if len(settable) > 1 {
			return false, errMultipleFields
		}
	}

	isAllowed := false
	for _, allowedSettableField := range allowedSettableFields {
		if set.field == allowedSettableField {
			isAllowed = true
			break
		}
	}

	if isAllowed {
		for _, settableField := range settable {
			if set.field == settableField {
				return true, nil
			}
		}
	}

	return false, nil
}

func updateSettingsEnv(env *[]string, set *settable) {
	for i, e := range *env {
		if parts := strings.SplitN(e, "=", 2); parts[0] == set.name {
			(*env)[i] = fmt.Sprintf("%s=%s", set.name, set.value)
			return
		}
	}

	*env = append(*env, fmt.Sprintf("%s=%s", set.name, set.value))
}
