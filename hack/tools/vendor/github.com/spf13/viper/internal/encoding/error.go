package encoding

type encodingError string

func (e encodingError) Error() string {
	return string(e)
}
