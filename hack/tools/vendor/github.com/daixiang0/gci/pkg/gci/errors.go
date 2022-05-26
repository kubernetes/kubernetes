package gci

import (
	"errors"
	"fmt"

	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	sectionsPkg "github.com/daixiang0/gci/pkg/gci/sections"
)

type EqualSpecificityMatchError struct {
	importDef          importPkg.ImportDef
	sectionA, sectionB sectionsPkg.Section
}

func (e EqualSpecificityMatchError) Error() string {
	return fmt.Sprintf("Import %s matched section %s and %s equally", e.importDef, e.sectionA, e.sectionB)
}

func (e EqualSpecificityMatchError) Is(err error) bool {
	_, ok := err.(EqualSpecificityMatchError)
	return ok
}

type NoMatchingSectionForImportError struct {
	importDef importPkg.ImportDef
}

func (n NoMatchingSectionForImportError) Error() string {
	return fmt.Sprintf("No section found for Import: %v", n.importDef)
}

func (n NoMatchingSectionForImportError) Is(err error) bool {
	_, ok := err.(NoMatchingSectionForImportError)
	return ok
}

type InvalidImportSplitError struct {
	segments []string
}

func (i InvalidImportSplitError) Error() string {
	return fmt.Sprintf("seperating the inline comment from the import yielded an invalid number of segments: %v", i.segments)
}

func (i InvalidImportSplitError) Is(err error) bool {
	_, ok := err.(InvalidImportSplitError)
	return ok
}

type InvalidAliasSplitError struct {
	segments []string
}

func (i InvalidAliasSplitError) Error() string {
	return fmt.Sprintf("seperating the alias from the path yielded an invalid number of segments: %v", i.segments)
}

func (i InvalidAliasSplitError) Is(err error) bool {
	_, ok := err.(InvalidAliasSplitError)
	return ok
}

var MissingImportStatementError = FileParsingError{errors.New("no import statement present in File")}
var ImportStatementNotClosedError = FileParsingError{errors.New("import statement not closed")}

type FileParsingError struct {
	error
}

func (f FileParsingError) Unwrap() error {
	return f.error
}

func (f FileParsingError) Is(err error) bool {
	_, ok := err.(FileParsingError)
	return ok
}
