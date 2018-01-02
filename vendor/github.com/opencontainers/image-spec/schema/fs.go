// Copyright 2016 The Linux Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package schema

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"sync"
	"time"
)

type _escLocalFS struct{}

var _escLocal _escLocalFS

type _escStaticFS struct{}

var _escStatic _escStaticFS

type _escDirectory struct {
	fs   http.FileSystem
	name string
}

type _escFile struct {
	compressed string
	size       int64
	modtime    int64
	local      string
	isDir      bool

	once sync.Once
	data []byte
	name string
}

func (_escLocalFS) Open(name string) (http.File, error) {
	f, present := _escData[path.Clean(name)]
	if !present {
		return nil, os.ErrNotExist
	}
	return os.Open(f.local)
}

func (_escStaticFS) prepare(name string) (*_escFile, error) {
	f, present := _escData[path.Clean(name)]
	if !present {
		return nil, os.ErrNotExist
	}
	var err error
	f.once.Do(func() {
		f.name = path.Base(name)
		if f.size == 0 {
			return
		}
		var gr *gzip.Reader
		b64 := base64.NewDecoder(base64.StdEncoding, bytes.NewBufferString(f.compressed))
		gr, err = gzip.NewReader(b64)
		if err != nil {
			return
		}
		f.data, err = ioutil.ReadAll(gr)
	})
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (fs _escStaticFS) Open(name string) (http.File, error) {
	f, err := fs.prepare(name)
	if err != nil {
		return nil, err
	}
	return f.File()
}

func (dir _escDirectory) Open(name string) (http.File, error) {
	return dir.fs.Open(dir.name + name)
}

func (f *_escFile) File() (http.File, error) {
	type httpFile struct {
		*bytes.Reader
		*_escFile
	}
	return &httpFile{
		Reader:   bytes.NewReader(f.data),
		_escFile: f,
	}, nil
}

func (f *_escFile) Close() error {
	return nil
}

func (f *_escFile) Readdir(count int) ([]os.FileInfo, error) {
	return nil, nil
}

func (f *_escFile) Stat() (os.FileInfo, error) {
	return f, nil
}

func (f *_escFile) Name() string {
	return f.name
}

func (f *_escFile) Size() int64 {
	return f.size
}

func (f *_escFile) Mode() os.FileMode {
	return 0
}

func (f *_escFile) ModTime() time.Time {
	return time.Unix(f.modtime, 0)
}

func (f *_escFile) IsDir() bool {
	return f.isDir
}

func (f *_escFile) Sys() interface{} {
	return f
}

// _escFS returns a http.Filesystem for the embedded assets. If useLocal is true,
// the filesystem's contents are instead used.
func _escFS(useLocal bool) http.FileSystem {
	if useLocal {
		return _escLocal
	}
	return _escStatic
}

// _escDir returns a http.Filesystem for the embedded assets on a given prefix dir.
// If useLocal is true, the filesystem's contents are instead used.
func _escDir(useLocal bool, name string) http.FileSystem {
	if useLocal {
		return _escDirectory{fs: _escLocal, name: name}
	}
	return _escDirectory{fs: _escStatic, name: name}
}

// _escFSByte returns the named file from the embedded assets. If useLocal is
// true, the filesystem's contents are instead used.
func _escFSByte(useLocal bool, name string) ([]byte, error) {
	if useLocal {
		f, err := _escLocal.Open(name)
		if err != nil {
			return nil, err
		}
		b, err := ioutil.ReadAll(f)
		f.Close()
		return b, err
	}
	f, err := _escStatic.prepare(name)
	if err != nil {
		return nil, err
	}
	return f.data, nil
}

// _escFSMustByte is the same as _escFSByte, but panics if name is not present.
func _escFSMustByte(useLocal bool, name string) []byte {
	b, err := _escFSByte(useLocal, name)
	if err != nil {
		panic(err)
	}
	return b
}

// _escFSString is the string version of _escFSByte.
func _escFSString(useLocal bool, name string) (string, error) {
	b, err := _escFSByte(useLocal, name)
	return string(b), err
}

// _escFSMustString is the string version of _escFSMustByte.
func _escFSMustString(useLocal bool, name string) string {
	return string(_escFSMustByte(useLocal, name))
}

var _escData = map[string]*_escFile{

	"/config-schema.json": {
		local:   "config-schema.json",
		size:    2771,
		modtime: 1489087148,
		compressed: `
H4sIAAAAAAAA/+RWQW/bPAy9+1cYbo9t/R2+U67dbgMyINh2KIZAsemEnSVqFD3MGPLfB8vJZtmym3XI
aScDFB/f4xMl60eSplkJrmC0gmSyVZqtLZhHMqLQAKePZCrcpxsLBVZYKJ9118FuXXEArTrIQcSu8vzZ
kbnvow/E+7xkVcn9f//nfeymx2F5hrhVnpMFU5zZnIf12TlqtYe88Pw9UloLHZZ2z1BIH7NMFlgQXLZK
u3bSNCsYlED5KzCAOmE0fTkfr4i1km6lVAL3ghoyv3bsUzLVyIF4oVSYzcUBBQppGC7FkLs08+RFJHvg
iI9HXPHxDw44iMwwDlh9ztvvlhyU74nFjfG3DJU3ECr30I3ATV5ChQa7UXG5VnbjK697jfH65tucLMWs
2uxuuIQCeixjoZE0Pc6QCreW0MiYmwysu56eAoKQblHigswXpIZyR5IXVZimrsNKwzqfoxY86vKf7f0j
1Y2GyThf2P9rp/7aXX0i/oJm/wZfdc7fqR3U17ZkE9n4a1qyEbIb3BtVX2xJMvyer18mkip6WV96/ZZY
VVssJwZf/6475S91H9CCafRkx7NatcAuizuejFgzhq8Nsv8PP0U8GKtLhhXPnh/QCXEbMz00K2LU3PbM
b1D07fCyW0vviMlWxN4UcYpZ/Enjdtf+RQ3SGiZ/vj8oANpKu/UTMV9kR1SDMjPzGZ6y5MQwnZvwWfX7
2RSey6SbnWPyMwAA//9KY9sL0woAAA==
`,
	},

	"/content-descriptor.json": {
		local:   "content-descriptor.json",
		size:    1085,
		modtime: 1494380450,
		compressed: `
H4sIAAAAAAAA/5yTwW7UMBCG73mKUVqpl27NoeIQVb3AnQPcEAevPY6nbGwznlW1oL47mniXJoAo3Vsy
+r+Zz8n4RwfQe6yOqQjl1A/QfyiY3uUklhIy6BMmgffHUGb4WNBRIGdn4lpbXFYXcbKKR5EyGPNQc9q0
6k3m0Xi2QTZvbk2rXTSO/AmpgzG5YHKnyXXGWtr4X9MbJ4eCSubtAzpptcK5IAth7QfQgwH0E3qyn1q4
lf48r0SEOadNIQfQAmNAxuTQw2LGjF8yBuU8hrp5FrvRE18Yj4ESae9qnqfP7FNr0Vf6/pKPRoASbA+C
9ZVOfxGhJG9v1xKeRqzygobjQ5E8si2RHLiI7mvdT9DYk1ZzuVZdfS1WBDnB1Z3djZlJ4nQ/3OmP9ejv
r875jkfXlf+ed/Uf9hZ21BQ1CIHzBI+RXASJVI/OMNkDbBF8fky7bD36c+xmk5WbTSnLfDtWiv+77DTZ
ERcrb5b9zhBc4s2zO7r2jN/2xKhin3+/McttXS9NB/Cle+p+BgAA///HjexwPQQAAA==
`,
	},

	"/defs-descriptor.json": {
		local:   "defs-descriptor.json",
		size:    922,
		modtime: 1494380450,
		compressed: `
H4sIAAAAAAAA/6STX2/TMBTF3/spLl7FgDZN4QFp0Ria2DsP42lTV93ZN/Ed8R/ZrqYy9bsjJ1naFYFA
PCSyj67Pub8b52kCIBRFGdgndlZUIK6oZst5F8FjSCw3LQZIDr56sl+cTciWAlwNx1yAa0+Sa5bYecx7
09FFVJBzAIQhxfht62mUAASrnKpT8rEqS+fJyueMuHChKaPUZLBkgw2Vakwt927zZ6/Ue4uYAttmr3tM
iUKHd3d7Wdxg8WNZnK32y1cn09fF3XoxWz0t5+8/fNyVf1c2FV3Erk8SihuK6ZDuaLhJE8iw9ck1Ab1m
CVKT/B43Bvqz4GrIRe7+gWSaA9tuOwDA6Tm2jQuctLmozvOoFKmL03+cwMA1e/O5up0t1sVqVN6+q/L6
srhZFmef1sVqdkS4CW38Ax9Cyz1ELoQ6OAOPmqWGpDkOVGBwC/cEyj3a1qEi9Wv/GAJu9zInMoe5vycF
ELULBvNXEJvAYtB3LzDQWpfw5fX8n7t46Dc2PQ1UZz9FdVw8RGdPyoPfojTor7ve+/cw50l+dpOfAQAA
//8aH/C2mgMAAA==
`,
	},

	"/defs.json": {
		local:   "defs.json",
		size:    1670,
		modtime: 1489087148,
		compressed: `
H4sIAAAAAAAA/7STza6bMBCF9zzFyO2S9oJtbGDb7hMpy6oLSiaJq2AjY6RWEe9e8RNChFuJKneRgGc8
3zmeMbcAgByxKa2qnTKa5EC+4klp1a8aaBs8grtY054vpnXgLgi7GvUXo12hNFo41FiqkyqLoTwceTOA
5NBLABClXTqvAIj7XWOvprTDM9qhckhUSquqrUgOn2KaPsLFrykcUzkEu3Amx2IrmlEpfPA+vsIzuhVP
Yy55ygT3aczJlZDgW4UyShmTNGIiTbiUIooij6Jn15N0+x/T8enQJFlxN8/GBxZJwtbozXPxoTnNeCYk
zdb8zePw8eOUcyE5jySTUZYk1Nf8WOxNz7VLQaNxdyI5fJsCMKeG9EeLfZZ8eFt8cG9Ty+eNXeivvp9G
t9frYvf09t3Ti1c6FPy1DhtnlT5vd3jXGOtf66kq6sOAHf99V8n8+Imle9ykunAOrd5bU6N1CptFEQD5
fIvD7in0ryMEy+fK1G6UfmdTE+tvpoL+1wV/AgAA//96IpqyhgYAAA==
`,
	},

	"/image-index-schema.json": {
		local:   "image-index-schema.json",
		size:    3009,
		modtime: 1495059260,
		compressed: `
H4sIAAAAAAAA/6yWv27bMBDGdz/FQQmQJYmKIuhgBFnaJVOHBl2KDAx5ki61SPVIJ3ELvXtBMrJlUXZt
1Zt95H33+07892cGkCm0kqlxZHQ2h+xrg/qz0U6QRob7WpQI91rhG3xrUFJBUoSplz733MoKa+HzKuea
eZ4/W6OvYvTacJkrFoW7+nCTx9hZzCPVpdh5npsGtexK2pAWZ+fky+fky8dEt2rQp5qnZ5Quxho2DbIj
tNkcvCWALOZ/R7bRVgynbh8qslAQLhTYaA8tuAohVIZQGaIYvEQ1EBaEBtIOS+SAEJQneMq3MddSncuk
Rk2a6mWdzeHjJibeulgItXEkq4WmAq2zffudsmAWqx67w7o/72g7XbEv7+01G+jxr/Y+wvhrSYy+1o91
1MOjIvHg0y77YUu/BxFFJVqXrUOPPfGRhZHIbw+kC8SvhTDbewBThMXBWCCjlqggsRREzhkLn62wsFdq
3ZNrvzvOcoUFafIVbL4h6Sm0qelDOP1EIA1PK4d2EusIIGn36WY33Hv/D8GTvGqcKVk0FUmQFcqfdllD
VOhwI+Olt+H/NsI5ZA0Xt2JRGiZX1XfzW78WFaq7i+l9H66boa8lL4arJnUlYEER3U+Hgk0NrxXJCpw/
V6IXqMUKnhCUedULIxSq6dSBaidzsxCuMFyn3Mdt5rXOgHPnNoY9WzmMCZYVOZRuyTjIA8jMlqetPQx7
93GqnY5Pdp/vhe61wzomXWaDCe2YzVPiGXsaqOuX5JY8Bdxa9jSQBQr/HU7dwo3uHszty7JfNrk2DzYJ
0P7T9otgEjo9XA/p4d7/7c4jRGhtXHjgjZx+x3V5c5DlfdXJZ19fZDbmpfvVbj2Dxh1Neq2N2fgfAx40
YKZnZzb2Muw96WYAj7N29jcAAP//8RsPuMELAAA=
`,
	},

	"/image-layout-schema.json": {
		local:   "image-layout-schema.json",
		size:    439,
		modtime: 1489792109,
		compressed: `
H4sIAAAAAAAA/2yPQUvEMBCF7/0VQ/Sg4DYVPOW6pwVhD4IX8VDTaTvLNonJVFik/12SaRXRU5g38+W9
91kBqA6TjRSYvFMG1DGg23vHLTmMcJjaAeGxvfiZ4cmOOLXqLlPXSQYDamQORutT8m4nau3joLvY9rxr
HrRoV8JRtyHJaO0DOruZpYLJtaZsrM/FWEi+BMysfzuhXbUQfcDIhEkZyG2yQyYl8TPGJLVk97fth1yA
74FHhOP+8LvyDbmy8JZ2EgZ6OuNtsS8fbrESR3LDj45unpSBl3UGUPd1UzdqnV/Lu1QAS2kS8X2miN03
8l+PKnNL9RUAAP//k31n5bcBAAA=
`,
	},

	"/image-manifest-schema.json": {
		local:   "image-manifest-schema.json",
		size:    921,
		modtime: 1489087148,
		compressed: `
H4sIAAAAAAAA/5ySMW8iMRCF+/0VI0MJ+O501bZXUZxSJEoTpXB2x7uDWNsZmygo4r9HtnHAkCKifTvv
zTdv/dEAiB59x+QCWSNaEHcOzT9rgiKDDOtJDQj/lSGNPsC9w440dSpNL6J97rsRJxWtYwiulXLjrVlm
dWV5kD0rHZa//sqszbKP+mLxrZTWoenKVp9seVpSJJDTkSB7w95hdNuXDXZHzbF1yIHQixbiYQAiRzwi
+3xclq9vfhjJgybc9uDzheghjAhpOZTlkPPgLQeC8qAMkAk4ICeKFH7bZbKG/Uort16tmcjQtJtEC39O
mnovWpIO+YvorNE0nDcwZ9QxNqKhCcvSiOVV/H+ism/VHtmf2wuVYlb7imkdcIqjv099HJVi/ul2gENF
oYyxIb28CuXGus/TFpet9Kj9JdRM9qjJULJU9qawJlLB+Lojxoj19N07rP9JXXED8Nwcms8AAAD//7u3
Dj+ZAwAA
`,
	},

	"/": {
		isDir: true,
		local: "",
	},
}
