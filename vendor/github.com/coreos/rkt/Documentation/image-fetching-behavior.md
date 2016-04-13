# Image Fetching Behavior

When fetching, rkt will try to avoid unnecessary network transfers: if an updated image is already in the local store there's no need to download it again.

This behavior can be controlled with the `--store-only` and `--no-store` flags.

## General Behavior

The following table describes the meaning of the `--store-only` and `--no-store` flags.

Flags                     | Description
------------------------- | ---------------------------------------------------------------------------------------------------
_no flags_                | **Default behavior.** Do `store` and if it doesn't return an image do `remote`.
`--store-only`            | Check the local store only.
`--no-store`              | Execute a remote download, while handling caching logic for `http(s)://` and `docker://`.
`--store-only --no-store` | Invalid option.

## Details

Here we detail the actions taken by rkt when fetching from store and remote for each type of image argument.

Fetch from   | Image argument     | Detailed behavior
------------ | ------------------ | --------------------------------------------------------------------------------------------
store        | file://            | Use the specified file
store        | http(s)://         | Check for the URL in the local store. If found, use the corresponding image.
store        | docker://          | Check for the URL in the local store. If found, use the corresponding image.
store        | image name         | Check local store. If found, use that image. If there's a file in the current directory named like the image name, use that file instead.
remote       | file://            | Use the specified file
remote       | http(s)://         | Search in the store if the URL is available. If it's available and the saved Cache-Control maxage > 0 determine if the image should be downloaded. If it's not expired use the image. Otherwise download (sending if available the saved ETag). If the download returns a `304 Not Modified` use the image already saved in the local store.
remote       | docker://          | Fetch using docker2aci.
remote       | image name         | Execute [discovery logic](https://github.com/appc/spec/blob/master/spec/discovery.md#app-container-image-discovery). If discovery is successful use the discovered URL doing the above `remote` http(s):// image case. If there's a file in the current directory named like the image name, use that file instead.
