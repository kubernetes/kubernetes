package sanitized_anchor_name_test

import (
	"fmt"

	"github.com/shurcooL/sanitized_anchor_name"
)

func ExampleCreate() {
	anchorName := sanitized_anchor_name.Create("This is a header")

	fmt.Println(anchorName)

	// Output:
	// this-is-a-header
}

func ExampleCreate2() {
	fmt.Println(sanitized_anchor_name.Create("This is a header"))
	fmt.Println(sanitized_anchor_name.Create("This is also          a header"))
	fmt.Println(sanitized_anchor_name.Create("main.go"))
	fmt.Println(sanitized_anchor_name.Create("Article 123"))
	fmt.Println(sanitized_anchor_name.Create("<- Let's try this, shall we?"))
	fmt.Printf("%q\n", sanitized_anchor_name.Create("        "))
	fmt.Println(sanitized_anchor_name.Create("Hello, 世界"))

	// Output:
	// this-is-a-header
	// this-is-also-a-header
	// main-go
	// article-123
	// let-s-try-this-shall-we
	// ""
	// hello-世界
}
