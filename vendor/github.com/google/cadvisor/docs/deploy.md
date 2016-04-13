# Building and Deploying the cAdvisor Docker Container

## Building

Building the cAdvisor Docker container is simple, just run:

```
$ ./deploy/build.sh
```

Which will statically build the cAdvisor binary and then build the Docker image. The resulting Docker image will be called `google/cadvisor:canary`. This image is very bare, containing the cAdvisor binary and nothing else.

## Deploying

All cAdvisor releases are tagged and correspond to a Docker image. The latest supported release uses the `latest` tag. We have a `beta` and `canary` tag for pre-release versions with newer features. You can see more details about this in the cAdvisor Docker [registry](https://registry.hub.docker.com/u/google/cadvisor/) page.
