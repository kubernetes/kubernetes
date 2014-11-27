go-bindata-http
===============

Serve embedded files from [jteeuwen/go-bindata](https://github.com/jteeuwen/go-bindata) with `net/http`.

[GoDoc](http://godoc.org/github.com/elazarl/go-bindata-assetfs)

After running

    $ go-bindata data/...

Use

     http.Handle("/",
        http.FileServer(
        &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, Prefix: "data"}))

to serve files embedded from the `data` directory.
