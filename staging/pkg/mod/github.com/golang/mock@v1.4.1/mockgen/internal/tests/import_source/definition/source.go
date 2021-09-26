//go:generate mockgen -destination ../source_mock.go -source=source.go
//go:generate mockgen -package source -destination source_mock.go -source=source.go
package source

type X struct{}

type S interface {
	F(X)
}
