//go:generate mockgen -destination mock_user/mock_user.go github.com/golang/mock/sample Index,Embed,Embedded

// An example package with an interface.
package user

// Random bunch of imports to test mockgen.
import "io"
import (
	btz "bytes"
	"hash"
	"log"
	"net"
	"net/http"

	// Two imports with the same base name.
	t1 "html/template"
	t2 "text/template"
)

// Dependencies outside the standard library.
import (
	"github.com/golang/mock/sample/imp1"
	renamed2 "github.com/golang/mock/sample/imp2"
	. "github.com/golang/mock/sample/imp3"
	"github.com/golang/mock/sample/imp4" // calls itself "imp_four"
)

// A bizarre interface to test corner cases in mockgen.
// This would normally be in its own file or package,
// separate from the user of it (e.g. io.Reader).
type Index interface {
	Get(key string) interface{}
	GetTwo(key1, key2 string) (v1, v2 interface{})
	Put(key string, value interface{})

	// Check that imports are handled correctly.
	Summary(buf *btz.Buffer, w io.Writer)
	Other() hash.Hash
	Templates(a t1.CSS, b t2.FuncMap)

	// A method with an anonymous argument.
	Anon(string)

	// Methods using foreign types outside the standard library.
	ForeignOne(imp1.Imp1)
	ForeignTwo(renamed2.Imp2)
	ForeignThree(Imp3)
	ForeignFour(imp_four.Imp4)

	// A method that returns a nillable type.
	NillableRet() error
	// A method that returns a non-interface type.
	ConcreteRet() chan<- bool

	// Methods with an ellipsis argument.
	Ellip(fmt string, args ...interface{})
	EllipOnly(...string)

	// A method with a pointer argument that we will set.
	Ptr(arg *int)

	// A method with a slice argument and an array return.
	Slice(a []int, b []byte) [3]int

	// A method with channel arguments.
	Chan(a chan int, b chan<- hash.Hash)

	// A method with a function argument.
	Func(f func(http.Request) (int, bool))

	// A method with a map argument.
	Map(a map[int]hash.Hash)

	// Methods with an unnamed empty struct argument.
	Struct(a struct{})          // not so likely
	StructChan(a chan struct{}) // a bit more common
}

// An interface with an embedded interface.
type Embed interface {
	RegularMethod()
	Embedded
	imp1.ForeignEmbedded
}

type Embedded interface {
	EmbeddedMethod()
}

// some random use of another package that isn't needed by the interface.
var _ net.Addr

// A function that we will test that uses the above interface.
// It takes a list of keys and values, and puts them in the index.
func Remember(index Index, keys []string, values []interface{}) {
	for i, k := range keys {
		index.Put(k, values[i])
	}
	err := index.NillableRet()
	if err != nil {
		log.Fatalf("Woah! %v", err)
	}
	if len(keys) > 0 && keys[0] == "a" {
		index.Ellip("%d", 0, 1, 1, 2, 3)
		index.Ellip("%d", 1, 3, 6, 10, 15)
		index.EllipOnly("arg")
	}
}

func GrabPointer(index Index) int {
	var a int
	index.Ptr(&a)
	return a
}
