package types

type TestNode interface {
	Type() SpecComponentType
	CodeLocation() CodeLocation

	Text() string
	SetText(text string)
	Flag() FlagType
	SetFlag(flag FlagType)
}
