package types

// BoolValues defines the name and value mappings for ParseBool.
var BoolValues = map[string]interface{}{
	"true": true, "yes": true, "on": true, "1": true,
	"false": false, "no": false, "off": false, "0": false,
}

var boolParser = func() *EnumParser {
	ep := &EnumParser{}
	ep.AddVals(BoolValues)
	return ep
}()

// ParseBool parses bool values according to the definitions in BoolValues.
// Parsing is case-insensitive.
func ParseBool(s string) (bool, error) {
	v, err := boolParser.Parse(s)
	if err != nil {
		return false, err
	}
	return v.(bool), nil
}
