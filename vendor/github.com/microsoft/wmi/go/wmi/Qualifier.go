package wmi

type Qualifier interface {
	Name() string
	Value() string
}
