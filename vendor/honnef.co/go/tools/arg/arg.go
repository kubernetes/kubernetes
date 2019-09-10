package arg

var args = map[string]int{
	"(*encoding/json.Decoder).Decode.v":    0,
	"(*encoding/json.Encoder).Encode.v":    0,
	"(*encoding/xml.Decoder).Decode.v":     0,
	"(*encoding/xml.Encoder).Encode.v":     0,
	"(*sync.Pool).Put.x":                   0,
	"(*text/template.Template).Parse.text": 0,
	"(io.Seeker).Seek.offset":              0,
	"(time.Time).Sub.u":                    0,
	"append.elems":                         1,
	"append.slice":                         0,
	"bytes.Equal.a":                        0,
	"bytes.Equal.b":                        1,
	"encoding/binary.Write.data":           2,
	"errors.New.text":                      0,
	"fmt.Fprintf.format":                   1,
	"fmt.Printf.format":                    0,
	"fmt.Sprintf.a[0]":                     1,
	"fmt.Sprintf.format":                   0,
	"json.Marshal.v":                       0,
	"json.Unmarshal.v":                     1,
	"len.v":                                0,
	"make.size[0]":                         1,
	"make.size[1]":                         2,
	"make.t":                               0,
	"net/url.Parse.rawurl":                 0,
	"os.OpenFile.flag":                     1,
	"os/exec.Command.name":                 0,
	"os/signal.Notify.c":                   0,
	"regexp.Compile.expr":                  0,
	"runtime.SetFinalizer.finalizer":       1,
	"runtime.SetFinalizer.obj":             0,
	"sort.Sort.data":                       0,
	"time.Parse.layout":                    0,
	"time.Sleep.d":                         0,
	"xml.Marshal.v":                        0,
	"xml.Unmarshal.v":                      1,
}

func Arg(name string) int {
	n, ok := args[name]
	if !ok {
		panic("unknown argument " + name)
	}
	return n
}
