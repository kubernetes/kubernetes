package custombytesnonstruct

type CustomType int

func (c *CustomType) Unmarshal(data []byte) error {
	data[0] = 42
	return nil
}
