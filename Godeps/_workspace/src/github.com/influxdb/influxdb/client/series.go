package client

type Series struct {
	Name    string          `json:"name"`
	Columns []string        `json:"columns"`
	Points  [][]interface{} `json:"points"`
}

func (self *Series) GetName() string {
	return self.Name
}

func (self *Series) GetColumns() []string {
	return self.Columns
}

func (self *Series) GetPoints() [][]interface{} {
	return self.Points
}
