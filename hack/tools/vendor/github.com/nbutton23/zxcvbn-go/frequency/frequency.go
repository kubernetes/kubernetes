package frequency

import (
	"encoding/json"
	"log"

	"github.com/nbutton23/zxcvbn-go/data"
)

// List holds a frequency list
type List struct {
	Name string
	List []string
}

// Lists holds all the frequency list in a map
var Lists = make(map[string]List)

func init() {
	maleFilePath := getAsset("data/MaleNames.json")
	femaleFilePath := getAsset("data/FemaleNames.json")
	surnameFilePath := getAsset("data/Surnames.json")
	englishFilePath := getAsset("data/English.json")
	passwordsFilePath := getAsset("data/Passwords.json")

	Lists["MaleNames"] = getStringListFromAsset(maleFilePath, "MaleNames")
	Lists["FemaleNames"] = getStringListFromAsset(femaleFilePath, "FemaleNames")
	Lists["Surname"] = getStringListFromAsset(surnameFilePath, "Surname")
	Lists["English"] = getStringListFromAsset(englishFilePath, "English")
	Lists["Passwords"] = getStringListFromAsset(passwordsFilePath, "Passwords")

}
func getAsset(name string) []byte {
	data, err := data.Asset(name)
	if err != nil {
		panic("Error getting asset " + name)
	}

	return data
}
func getStringListFromAsset(data []byte, name string) List {

	var tempList List
	err := json.Unmarshal(data, &tempList)
	if err != nil {
		log.Fatal(err)
	}
	tempList.Name = name
	return tempList
}
