package swagger

import "testing"

type Boat struct {
	Length int `json:"-"` // on default, this makes the fields not required
	Weight int `json:"-"`
}

// PostBuildModel is from swagger.ModelBuildable
func (b Boat) PostBuildModel(m *Model) *Model {
	// override required
	m.Required = []string{"Length", "Weight"}

	// add model property (just to test is can be added; is this a real usecase?)
	extraType := "string"
	m.Properties.Put("extra", ModelProperty{
		Description: "extra description",
		DataTypeFields: DataTypeFields{
			Type: &extraType,
		},
	})
	return m
}

func TestCustomPostModelBuilde(t *testing.T) {
	testJsonFromStruct(t, Boat{}, `{
  "swagger.Boat": {
   "id": "swagger.Boat",
   "required": [
    "Length",
    "Weight"
   ],
   "properties": {
    "extra": {
     "type": "string",
     "description": "extra description"
    }
   }
  }
}`)
}
