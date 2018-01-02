/*

Package winresources is used to embed Windows resources into docker.exe.
These resources are used to provide

    * Version information
    * An icon
    * A Windows manifest declaring Windows version support

The resource object files are generated in hack/make/.go-autogen from
source files in hack/make/.resources-windows. This occurs automatically
when you run hack/make.sh.

These object files are picked up automatically by go build when this package
is included.

*/
package winresources
