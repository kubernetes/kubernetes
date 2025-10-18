package issue330

type TypeIdentifier uint32

const (
	UnknownType TypeIdentifier = 0
	UserType    TypeIdentifier = 20
)

func (t TypeIdentifier) String() string {
	switch t {
	case 20:
		return "User"
	default:
		return "Unknown"
	}
}
