package gcfg

import warnings "gopkg.in/warnings.v0"

// FatalOnly filters the results of a Read*Into invocation and returns only
// fatal errors. That is, errors (warnings) indicating data for unknown
// sections / variables is ignored. Example invocation:
//
//  err := gcfg.FatalOnly(gcfg.ReadFileInto(&cfg, configFile))
//  if err != nil {
//      ...
//
func FatalOnly(err error) error {
	return warnings.FatalOnly(err)
}

func isFatal(err error) bool {
	_, ok := err.(extraData)
	return !ok
}

type loc struct {
	section    string
	subsection *string
	variable   *string
}

type extraData struct {
	loc
}

type locErr struct {
	msg string
	loc
}

func (l loc) String() string {
	s := "section \"" + l.section + "\""
	if l.subsection != nil {
		s += ", subsection \"" + *l.subsection + "\""
	}
	if l.variable != nil {
		s += ", variable \"" + *l.variable + "\""
	}
	return s
}

func (e extraData) Error() string {
	return "can't store data at " + e.loc.String()
}

func (e locErr) Error() string {
	return e.msg + " at " + e.loc.String()
}

var _ error = extraData{}
var _ error = locErr{}
