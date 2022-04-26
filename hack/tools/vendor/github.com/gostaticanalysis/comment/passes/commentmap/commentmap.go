package commentmap

import (
	"reflect"

	"golang.org/x/tools/go/analysis"

	"github.com/gostaticanalysis/comment"
)

var Analyzer = &analysis.Analyzer{
	Name:             "commentmap",
	Doc:              "create comment map",
	Run:              run,
	RunDespiteErrors: true,
	ResultType:       reflect.TypeOf(comment.Maps{}),
}

func run(pass *analysis.Pass) (interface{}, error) {
	return comment.New(pass.Fset, pass.Files), nil
}
