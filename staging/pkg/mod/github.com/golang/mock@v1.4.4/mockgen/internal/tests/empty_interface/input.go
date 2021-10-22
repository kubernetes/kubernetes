//go:generate mockgen -package empty_interface -destination mock.go -source input.go
package empty_interface

type Empty interface{}
