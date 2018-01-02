// code examples for godoc

package toml

import (
	"fmt"
)

func ExampleNodeFilterFn_filterExample() {
	tree, _ := Load(`
      [struct_one]
      foo = "foo"
      bar = "bar"

      [struct_two]
      baz = "baz"
      gorf = "gorf"
    `)

	// create a query that references a user-defined-filter
	query, _ := CompileQuery("$[?(bazOnly)]")

	// define the filter, and assign it to the query
	query.SetFilter("bazOnly", func(node interface{}) bool {
		if tree, ok := node.(*TomlTree); ok {
			return tree.Has("baz")
		}
		return false // reject all other node types
	})

	// results contain only the 'struct_two' TomlTree
	query.Execute(tree)
}

func ExampleQuery_queryExample() {
	config, _ := Load(`
      [[book]]
      title = "The Stand"
      author = "Stephen King"
      [[book]]
      title = "For Whom the Bell Tolls"
      author = "Ernest Hemmingway"
      [[book]]
      title = "Neuromancer"
      author = "William Gibson"
    `)

	// find and print all the authors in the document
	authors, _ := config.Query("$.book.author")
	for _, name := range authors.Values() {
		fmt.Println(name)
	}
}

func Example_comprehensiveExample() {
	config, err := LoadFile("config.toml")

	if err != nil {
		fmt.Println("Error ", err.Error())
	} else {
		// retrieve data directly
		user := config.Get("postgres.user").(string)
		password := config.Get("postgres.password").(string)

		// or using an intermediate object
		configTree := config.Get("postgres").(*TomlTree)
		user = configTree.Get("user").(string)
		password = configTree.Get("password").(string)
		fmt.Println("User is ", user, ". Password is ", password)

		// show where elements are in the file
		fmt.Printf("User position: %v\n", configTree.GetPosition("user"))
		fmt.Printf("Password position: %v\n", configTree.GetPosition("password"))

		// use a query to gather elements without walking the tree
		results, _ := config.Query("$..[user,password]")
		for ii, item := range results.Values() {
			fmt.Printf("Query result %d: %v\n", ii, item)
		}
	}
}
