package tests

//easyjson:json
type SnakeStruct struct {
	WeirdHTTPStuff   bool
	CustomNamedField string `json:"cUsToM"`
}

var snakeStructValue SnakeStruct
var snakeStructString = `{"weird_http_stuff":false,"cUsToM":""}`
