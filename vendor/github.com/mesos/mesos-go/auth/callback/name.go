package callback

type Name struct {
	name string
}

func NewName() *Name {
	return &Name{}
}

func (cb *Name) Get() string {
	return cb.name
}

func (cb *Name) Set(name string) {
	cb.name = name
}
