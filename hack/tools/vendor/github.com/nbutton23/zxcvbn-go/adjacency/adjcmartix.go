package adjacency

import (
	"encoding/json"
	"log"

	"github.com/nbutton23/zxcvbn-go/data"
)

// Graph holds information about different graphs
type Graph struct {
	Graph         map[string][]string
	averageDegree float64
	Name          string
}

// GraphMap is a map of all graphs
var GraphMap = make(map[string]Graph)

func init() {
	GraphMap["qwerty"] = BuildQwerty()
	GraphMap["dvorak"] = BuildDvorak()
	GraphMap["keypad"] = BuildKeypad()
	GraphMap["macKeypad"] = BuildMacKeypad()
	GraphMap["l33t"] = BuildLeet()
}

//BuildQwerty builds the Qwerty Graph
func BuildQwerty() Graph {
	data, err := data.Asset("data/Qwerty.json")
	if err != nil {
		panic("Can't find asset")
	}
	return getAdjancencyGraphFromFile(data, "qwerty")
}

//BuildDvorak builds the Dvorak Graph
func BuildDvorak() Graph {
	data, err := data.Asset("data/Dvorak.json")
	if err != nil {
		panic("Can't find asset")
	}
	return getAdjancencyGraphFromFile(data, "dvorak")
}

//BuildKeypad builds the Keypad Graph
func BuildKeypad() Graph {
	data, err := data.Asset("data/Keypad.json")
	if err != nil {
		panic("Can't find asset")
	}
	return getAdjancencyGraphFromFile(data, "keypad")
}

//BuildMacKeypad builds the Mac Keypad Graph
func BuildMacKeypad() Graph {
	data, err := data.Asset("data/MacKeypad.json")
	if err != nil {
		panic("Can't find asset")
	}
	return getAdjancencyGraphFromFile(data, "mac_keypad")
}

//BuildLeet builds the L33T Graph
func BuildLeet() Graph {
	data, err := data.Asset("data/L33t.json")
	if err != nil {
		panic("Can't find asset")
	}
	return getAdjancencyGraphFromFile(data, "keypad")
}

func getAdjancencyGraphFromFile(data []byte, name string) Graph {

	var graph Graph
	err := json.Unmarshal(data, &graph)
	if err != nil {
		log.Fatal(err)
	}
	graph.Name = name
	return graph
}

// CalculateAvgDegree calclates the average degree between nodes in the graph
//on qwerty, 'g' has degree 6, being adjacent to 'ftyhbv'. '\' has degree 1.
//this calculates the average over all keys.
//TODO double check that i ported this correctly scoring.coffee ln 5
func (adjGrp Graph) CalculateAvgDegree() float64 {
	if adjGrp.averageDegree != float64(0) {
		return adjGrp.averageDegree
	}
	var avg float64
	var count float64
	for _, value := range adjGrp.Graph {

		for _, char := range value {
			if len(char) != 0 || char != " " {
				avg += float64(len(char))
				count++
			}
		}

	}

	adjGrp.averageDegree = avg / count

	return adjGrp.averageDegree
}
