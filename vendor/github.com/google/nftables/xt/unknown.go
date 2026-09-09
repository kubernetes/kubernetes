package xt

// Unknown represents the bytes Info payload for unknown Info types where no
// dedicated match/target info type has (yet) been defined.
type Unknown []byte

func (x *Unknown) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	// In case of unknown payload we assume its creator knows what she/he does
	// and thus we don't do any alignment padding. Just take the payload "as
	// is".
	return *x, nil
}

func (x *Unknown) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	*x = data
	return nil
}
