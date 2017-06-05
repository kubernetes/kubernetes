# Apache HTTPd sample for Registry v1, v2 and mirror

3 containers involved 

* Docker Registry v1 (registry 0.9.1)
* Docker Registry v2 (registry 2.0.0)
* Docker Registry v1 in mirror mode

HTTP for mirror and HTTPS for v1 & v2

* http://registry.example.com proxify Docker Registry 1.0 in Mirror mode
* https://registry.example.com proxify Docker Registry 1.0 or 2.0 in Hosting mode

## 3 Docker containers should be started 

* Docker Registry 1.0 in Mirror mode : port 5001
* Docker Registry 1.0 in Hosting mode : port 5000
* Docker Registry 2.0 in Hosting mode : port 5002

### Registry v1

    docker run -d -e SETTINGS_FLAVOR=dev -v /var/lib/docker-registry/storage/hosting-v1:/tmp -p 5000:5000 registry:0.9.1"

### Mirror

    docker run -d -e SETTINGS_FLAVOR=dev -e STANDALONE=false -e MIRROR_SOURCE=https://registry-1.docker.io -e MIRROR_SOURCE_INDEX=https://index.docker.io \
                  -e MIRROR_TAGS_CACHE_TTL=172800 -v /var/lib/docker-registry/storage/mirror:/tmp -p 5001:5000 registry:0.9.1"

### Registry v2

    docker run -d -e SETTINGS_FLAVOR=dev -v /var/lib/axway/docker-registry/storage/hosting2-v2:/tmp -p 5002:5000 registry:2"

# For Hosting mode access

* users should have account (valid-user) to be able to fetch images
* only users using account docker-deployer will be allowed to push images
