# Containerd website

The containerd website is built using Jekyll and published to Github pages.

In order to build and test locally:
```
docker run -it -v "$PWD":/usr/src/app -p "4000:4000" starefossen/github-pages
```
Then browser to localhost:4000 to see the rendered site. The site autorefreshes when you modify files locally.

