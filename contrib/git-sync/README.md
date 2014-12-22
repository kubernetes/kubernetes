# git-sync

git-sync is a command that periodically sync a git repository to a local directory.

It can be used to source a container volume with the content of a git repo.

## Usage

```
# build the container
docker build -t git-sync .
# run the git-sync container
docker run -d -e INTERVAL=1s -e REPO=https://github.com/GoogleCloudPlatform/kubernetes -e BRANCH=gh-pages -v /git-data:/usr/share/nginx/html git-sync
# run a nginx container to serve sync'ed content
docker run -d -p 8080:80 -v /git-data:/var/www nginx 
```
