package ansiterm

type AnsiContext struct {
	currentChar byte
	paramBuffer []byte
	interBuffer []byte
}
