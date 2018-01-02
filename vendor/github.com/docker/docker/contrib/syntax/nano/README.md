Dockerfile.nanorc
=================

Dockerfile syntax highlighting for nano

Single User Installation
------------------------
1. Create a nano syntax directory in your home directory:
 * `mkdir -p ~/.nano/syntax`

2. Copy `Dockerfile.nanorc` to` ~/.nano/syntax/`
 * `cp Dockerfile.nanorc ~/.nano/syntax/`

3. Add the following to your `~/.nanorc` to tell nano where to find the `Dockerfile.nanorc` file
  ```
## Dockerfile files
include "~/.nano/syntax/Dockerfile.nanorc"
  ```

System Wide Installation
------------------------
1. Create a nano syntax directory: 
  * `mkdir /usr/local/share/nano`

2. Copy `Dockerfile.nanorc` to `/usr/local/share/nano`
  * `cp Dockerfile.nanorc /usr/local/share/nano/`

3. Add the following to your `/etc/nanorc`:
  ```
## Dockerfile files
include "/usr/local/share/nano/Dockerfile.nanorc"
  ```
