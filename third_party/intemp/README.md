# intemp

A bash script to execute a command within a temporary work directory.


## Dependencies

Requires: mktemp


## Install

```
git clone https://github.com/karlkfi/intemp
cd intemp
make install
```

or

```
curl -o- https://raw.githubusercontent.com/karlkfi/intemp/master/install.sh | bash
```

## Usage

```
intemp.sh [-t prefix] "<command>"
```

Example (install intemp using intemp):

```
intemp.sh -t intemp "git clone https://github.com/karlkfi/intemp . && make install"
```


## License

Copyright 2015 Karl Isenberg

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.