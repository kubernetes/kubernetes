package typedecl

type Dropped struct {
	Name string
	Age  int32
}

func (d *Dropped) Drop() bool {
	return true
}

type DroppedWithoutGetters struct {
	Width  int64
	Height int64
}

func (d *DroppedWithoutGetters) GetHeight() int64 {
	return d.Height
}
