package gmeasure

import "encoding/json"

type enumSupport struct {
	toString map[uint]string
	toEnum   map[string]uint
	maxEnum  uint
}

func newEnumSupport(toString map[uint]string) enumSupport {
	toEnum, maxEnum := map[string]uint{}, uint(0)
	for k, v := range toString {
		toEnum[v] = k
		if maxEnum < k {
			maxEnum = k
		}
	}
	return enumSupport{toString: toString, toEnum: toEnum, maxEnum: maxEnum}
}

func (es enumSupport) String(e uint) string {
	if e > es.maxEnum {
		return es.toString[0]
	}
	return es.toString[e]
}

func (es enumSupport) UnmarshJSON(b []byte) (uint, error) {
	var dec string
	if err := json.Unmarshal(b, &dec); err != nil {
		return 0, err
	}
	out := es.toEnum[dec] // if we miss we get 0 which is what we want anyway
	return out, nil
}

func (es enumSupport) MarshJSON(e uint) ([]byte, error) {
	if e == 0 || e > es.maxEnum {
		return json.Marshal(nil)
	}
	return json.Marshal(es.toString[e])
}
