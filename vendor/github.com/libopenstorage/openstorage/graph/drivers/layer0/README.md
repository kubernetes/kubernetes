## Layer0 Graph Driver

Layer0 implemenation piggy backs on existing overlay graphdriver implementation to provide persistent storage for the uppermost/writeable layer in the container rootfs. 
The persistent storage is derived from one of the OSD volume drivers.

To use this as the graphdriver in Docker with aws as the backend volume provider:

```
DOCKER_STORAGE_OPTIONS= -s layer0 --storage-opt layer0.volume_driver=aws
```
