// Package garif defines all the GO structures required to model a SARIF log file.
// These structures were created using the JSON-schema sarif-schema-2.1.0.json of SARIF logfiles
// available at https://github.com/oasis-tcs/sarif-spec/tree/master/Schemata.
//
// The package provides constructors for all structures (see constructors.go) These constructors
// ensure that the returned structure instantiation is valid with respect to the JSON schema and
// should be used in place of plain structure instantiation.
// The root structure is LogFile.
//
// The package provides utility decorators for the most commonly used structures (see decorators.go)
package garif
