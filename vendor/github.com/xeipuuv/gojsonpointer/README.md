# gojsonpointer
An implementation of JSON Pointer - Go language

## Usage
	jsonText := `{
		"name": "Bobby B",
		"occupation": {
			"title" : "King",
			"years" : 15,
			"heir" : "Joffrey B"			
		}
	}`
	
    var jsonDocument map[string]interface{}
    json.Unmarshal([]byte(jsonText), &jsonDocument)
    
    //create a JSON pointer
    pointerString := "/occupation/title"
    pointer, _ := NewJsonPointer(pointerString)
    
    //SET a new value for the "title" in the document     
    pointer.Set(jsonDocument, "Supreme Leader of Westeros")
    
    //GET the new "title" from the document
    title, _, _ := pointer.Get(jsonDocument)
    fmt.Println(title) //outputs "Supreme Leader of Westeros"
    
    //DELETE the "heir" from the document
    deletePointer := NewJsonPointer("/occupation/heir")
    deletePointer.Delete(jsonDocument)
    
    b, _ := json.Marshal(jsonDocument)
    fmt.Println(string(b))
    //outputs `{"name":"Bobby B","occupation":{"title":"Supreme Leader of Westeros","years":15}}`


## References
http://tools.ietf.org/html/draft-ietf-appsawg-json-pointer-07

### Note
The 4.Evaluation part of the previous reference, starting with 'If the currently referenced value is a JSON array, the reference token MUST contain either...' is not implemented.
