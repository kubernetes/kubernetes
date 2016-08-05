// assetfs allows packages to serve static content embedded
// with the go-bindata tool with the standard net/http package.
//
// See https://github.com/jteeuwen/go-bindata for more information
// about embedding binary data with go-bindata.
//
// Usage example, after running
//    $ go-bindata data/...
// use:
//     http.Handle("/",
//        http.FileServer(
//        &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, Prefix: "data"}))
package assetfs
