package types

import "encoding/json"

type EnumSupport struct {
	toString map[uint]string
	toEnum   map[string]uint
	maxEnum  uint
}

func NewEnumSupport(toString map[uint]string) EnumSupport {
	toEnum, maxEnum := map[string]uint{}, uint(0)
	for k, v := range toString {
		toEnum[v] = k
		if maxEnum < k {
			maxEnum = k
		}
	}
	return EnumSupport{toString: toString, toEnum: toEnum, maxEnum: maxEnum}
}

func (es EnumSupport) String(e uint) string {
	if e > es.maxEnum {
		return es.toString[0]
	}
	return es.toString[e]
}

func (es EnumSupport) UnmarshJSON(b []byte) (uint, error) {
	var dec string
	if err := json.Unmarshal(b, &dec); err != nil {
		return 0, err
	}
	out := es.toEnum[dec] // if we miss we get 0 which is what we want anyway
	return out, nil
}

func (es EnumSupport) MarshJSON(e uint) ([]byte, error) {
	if e == 0 || e > es.maxEnum {
		return json.Marshal(nil)
	}
	return json.Marshal(es.toString[e])
}
