# hugo

`hugo` is a container that convert markdown into html using [hugo static site generator](http://gohugo.io/).

## Usage

```
# build the container
docker build -t hugo .
# run the hugo container
docker run -e HUGO_BASE_URL=example.com -v /path/to/md:/src -v /path/to/html:/dest hugo
``


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/git-sync/demo/hugo/README.md?pixel)]()
