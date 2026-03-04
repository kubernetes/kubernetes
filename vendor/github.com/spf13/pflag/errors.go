package pflag

import "fmt"

// notExistErrorMessageType specifies which flavor of "flag does not exist"
// is printed by NotExistError. This allows the related errors to be grouped
// under a single NotExistError struct without making a breaking change to
// the error message text.
type notExistErrorMessageType int

const (
	flagNotExistMessage notExistErrorMessageType = iota
	flagNotDefinedMessage
	flagNoSuchFlagMessage
	flagUnknownFlagMessage
	flagUnknownShorthandFlagMessage
)

// NotExistError is the error returned when trying to access a flag that
// does not exist in the FlagSet.
type NotExistError struct {
	name                string
	specifiedShorthands string
	messageType         notExistErrorMessageType
}

// Error implements error.
func (e *NotExistError) Error() string {
	switch e.messageType {
	case flagNotExistMessage:
		return fmt.Sprintf("flag %q does not exist", e.name)

	case flagNotDefinedMessage:
		return fmt.Sprintf("flag accessed but not defined: %s", e.name)

	case flagNoSuchFlagMessage:
		return fmt.Sprintf("no such flag -%v", e.name)

	case flagUnknownFlagMessage:
		return fmt.Sprintf("unknown flag: --%s", e.name)

	case flagUnknownShorthandFlagMessage:
		c := rune(e.name[0])
		return fmt.Sprintf("unknown shorthand flag: %q in -%s", c, e.specifiedShorthands)
	}

	panic(fmt.Errorf("unknown flagNotExistErrorMessageType: %v", e.messageType))
}

// GetSpecifiedName returns the name of the flag (without dashes) as it
// appeared in the parsed arguments.
func (e *NotExistError) GetSpecifiedName() string {
	return e.name
}

// GetSpecifiedShortnames returns the group of shorthand arguments
// (without dashes) that the flag appeared within. If the flag was not in a
// shorthand group, this will return an empty string.
func (e *NotExistError) GetSpecifiedShortnames() string {
	return e.specifiedShorthands
}

// ValueRequiredError is the error returned when a flag needs an argument but
// no argument was provided.
type ValueRequiredError struct {
	flag                *Flag
	specifiedName       string
	specifiedShorthands string
}

// Error implements error.
func (e *ValueRequiredError) Error() string {
	if len(e.specifiedShorthands) > 0 {
		c := rune(e.specifiedName[0])
		return fmt.Sprintf("flag needs an argument: %q in -%s", c, e.specifiedShorthands)
	}

	return fmt.Sprintf("flag needs an argument: --%s", e.specifiedName)
}

// GetFlag returns the flag for which the error occurred.
func (e *ValueRequiredError) GetFlag() *Flag {
	return e.flag
}

// GetSpecifiedName returns the name of the flag (without dashes) as it
// appeared in the parsed arguments.
func (e *ValueRequiredError) GetSpecifiedName() string {
	return e.specifiedName
}

// GetSpecifiedShortnames returns the group of shorthand arguments
// (without dashes) that the flag appeared within. If the flag was not in a
// shorthand group, this will return an empty string.
func (e *ValueRequiredError) GetSpecifiedShortnames() string {
	return e.specifiedShorthands
}

// InvalidValueError is the error returned when an invalid value is used
// for a flag.
type InvalidValueError struct {
	flag  *Flag
	value string
	cause error
}

// Error implements error.
func (e *InvalidValueError) Error() string {
	flag := e.flag
	var flagName string
	if flag.Shorthand != "" && flag.ShorthandDeprecated == "" {
		flagName = fmt.Sprintf("-%s, --%s", flag.Shorthand, flag.Name)
	} else {
		flagName = fmt.Sprintf("--%s", flag.Name)
	}
	return fmt.Sprintf("invalid argument %q for %q flag: %v", e.value, flagName, e.cause)
}

// Unwrap implements errors.Unwrap.
func (e *InvalidValueError) Unwrap() error {
	return e.cause
}

// GetFlag returns the flag for which the error occurred.
func (e *InvalidValueError) GetFlag() *Flag {
	return e.flag
}

// GetValue returns the invalid value that was provided.
func (e *InvalidValueError) GetValue() string {
	return e.value
}

// InvalidSyntaxError is the error returned when a bad flag name is passed on
// the command line.
type InvalidSyntaxError struct {
	specifiedFlag string
}

// Error implements error.
func (e *InvalidSyntaxError) Error() string {
	return fmt.Sprintf("bad flag syntax: %s", e.specifiedFlag)
}

// GetSpecifiedName returns the exact flag (with dashes) as it
// appeared in the parsed arguments.
func (e *InvalidSyntaxError) GetSpecifiedFlag() string {
	return e.specifiedFlag
}
