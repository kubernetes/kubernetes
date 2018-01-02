package influxql // import "github.com/influxdata/influxdb/influxql"

//go:generate tmpl -data=@tmpldata iterator.gen.go.tmpl
//go:generate tmpl -data=@tmpldata point.gen.go.tmpl
//go:generate tmpl -data=@tmpldata functions.gen.go.tmpl

//go:generate protoc --gogo_out=. internal/internal.proto
