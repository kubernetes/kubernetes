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

package schema_test

import (
	_ "crypto/sha256"
	"strings"
	"testing"

	"github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/schema"
	"github.com/opencontainers/image-spec/specs-go/v1"
)

var compatMap = map[string]string{
	"application/vnd.docker.distribution.manifest.list.v2+json": v1.MediaTypeImageIndex,
	"application/vnd.docker.distribution.manifest.v2+json":      v1.MediaTypeImageManifest,
	"application/vnd.docker.image.rootfs.diff.tar.gzip":         v1.MediaTypeImageLayerGzip,
	"application/vnd.docker.container.image.v1+json":            v1.MediaTypeImageConfig,
}

// convertFormats converts Docker v2.2 image format JSON documents to OCI
// format by simply replacing instances of the strings found in the compatMap
// found in the input string.
func convertFormats(input string) string {
	out := input
	for k, v := range compatMap {
		out = strings.Replace(out, k, v, -1)
	}
	return out
}

func TestBackwardsCompatibilityImageIndex(t *testing.T) {
	for i, tt := range []struct {
		imageIndex string
		digest     digest.Digest
		fail       bool
	}{
		{
			digest: "sha256:4ffd0883f25635999f04ea543240a27c9a4341979ff7d46a9774f71512eebb1f",
			imageIndex: `{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
   "manifests": [
      {
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "size": 2094,
         "digest": "sha256:7820f9a86d4ad15a2c4f0c0e5479298df2aa7c2f6871288e2ef8546f3e7b6783",
         "platform": {
            "architecture": "ppc64le",
            "os": "linux"
         }
      },
      {
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "size": 1922,
         "digest": "sha256:ae1b0e06e8ade3a11267564a26e750585ba2259c0ecab59ab165ad1af41d1bdd",
         "platform": {
            "architecture": "amd64",
            "os": "linux"
         }
      },
      {
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "size": 2084,
         "digest": "sha256:e4c0df75810b953d6717b8f8f28298d73870e8aa2a0d5e77b8391f16fdfbbbe2",
         "platform": {
            "architecture": "s390x",
            "os": "linux"
         }
      },
      {
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "size": 2084,
         "digest": "sha256:07ebe243465ef4a667b78154ae6c3ea46fdb1582936aac3ac899ea311a701b40",
         "platform": {
            "architecture": "arm",
            "os": "linux",
            "variant": "v7"
         }
      },
      {
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "size": 2090,
         "digest": "sha256:fb2fc0707b86dafa9959fe3d29e66af8787aee4d9a23581714be65db4265ad8a",
         "platform": {
            "architecture": "arm64",
            "os": "linux",
            "variant": "v8"
         }
      }
   ]
}`,
			fail: false,
		},
	} {
		got := digest.FromString(tt.imageIndex)
		if tt.digest != got {
			t.Errorf("test %d: expected digest %s but got %s", i, tt.digest, got)
		}

		imageIndex := convertFormats(tt.imageIndex)
		r := strings.NewReader(imageIndex)
		err := schema.ValidatorMediaTypeImageIndex.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}

func TestBackwardsCompatibilityManifest(t *testing.T) {
	for i, tt := range []struct {
		manifest string
		digest   digest.Digest
		fail     bool
	}{
		// manifest pulled from docker hub using hash value
		//
		// curl -L -H "Authorization: Bearer ..." -H \
		// "Accept: application/vnd.docker.distribution.manifest.v2+json" \
		// https://registry-1.docker.io/v2/library/docker/manifests/sha256:888206c77cd2811ec47e752ba291e5b7734e3ef137dfd222daadaca39a9f17bc
		{
			digest: "sha256:888206c77cd2811ec47e752ba291e5b7734e3ef137dfd222daadaca39a9f17bc",
			manifest: `{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/octet-stream",
      "size": 3210,
      "digest": "sha256:5359a4f250650c20227055957e353e8f8a74152f35fe36f00b6b1f9fc19c8861"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2310272,
         "digest": "sha256:fae91920dcd4542f97c9350b3157139a5d901362c2abec284de5ebd1b45b4957"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 913022,
         "digest": "sha256:f384f6ab36adad485192f09379c0b58dc612a3cde82c551e082a7c29a87c95da"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 9861668,
         "digest": "sha256:ed0d2dd5e1a0e5e650a330a864c8a122e9aa91fa6ba9ac6f0bd1882e59df55e7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 465,
         "digest": "sha256:ec4d00b58417c45f7ddcfde7bcad8c9d62a7d6d5d17cdc1f7d79bcb2e22c1491"
      }
   ]
}`,
			fail: false,
		},
	} {
		got := digest.FromString(tt.manifest)
		if tt.digest != got {
			t.Errorf("test %d: expected digest %s but got %s", i, tt.digest, got)
		}

		manifest := convertFormats(tt.manifest)
		r := strings.NewReader(manifest)
		err := schema.ValidatorMediaTypeManifest.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}

func TestBackwardsCompatibilityConfig(t *testing.T) {
	for i, tt := range []struct {
		config string
		digest digest.Digest
		fail   bool
	}{
		// config pulled from docker hub blob store
		//
		// $ TOKEN=$(curl https://auth.docker.io/token\?service\=registry.docker.io\&scope\=repository:library/docker:pull  | jq -r .token)
		// $ CONFIG_DIGEST=$(curl -H "Authorization: Bearer ${TOKEN}" -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' https://index.docker.io/v2/library/docker/manifests/1.12.1 | jq -r .config.digest)
		// $ curl -LH "Authorization: Bearer ${TOKEN}" https://index.docker.io/v2/library/docker/blobs/${CONFIG_DIGEST}
		{
			digest: "sha256:a059ea7356d5b5a9e0f6352bfa463e7bd4721c2ade3ef168603826e0de6fe54b",
			config: `{"architecture":"amd64","config":{"Hostname":"09713501c176","Domainname":"","User":"","AttachStdin":false,"AttachStdout":false,"AttachStderr":false,"Tty":false,"OpenStdin":false,"StdinOnce":false,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","DOCKER_BUCKET=get.docker.com","DOCKER_VERSION=1.12.1","DOCKER_SHA256=05ceec7fd937e1416e5dce12b0b6e1c655907d349d52574319a1e875077ccb79"],"Cmd":["sh"],"Image":"sha256:32e2e3ccf2a4fbaa75b078bf539cd5ea2e374a4242665a5ec3f3c01e7a3eefb8","Volumes":null,"WorkingDir":"","Entrypoint":["docker-entrypoint.sh"],"OnBuild":[],"Labels":{}},"container":"15a30be053fb3069a7879b4ea537e84689d8e8e8ba94dc4dd499271506803ba1","container_config":{"Hostname":"09713501c176","Domainname":"","User":"","AttachStdin":false,"AttachStdout":false,"AttachStderr":false,"Tty":false,"OpenStdin":false,"StdinOnce":false,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","DOCKER_BUCKET=get.docker.com","DOCKER_VERSION=1.12.1","DOCKER_SHA256=05ceec7fd937e1416e5dce12b0b6e1c655907d349d52574319a1e875077ccb79"],"Cmd":["/bin/sh","-c","#(nop) ","CMD [\"sh\"]"],"Image":"sha256:32e2e3ccf2a4fbaa75b078bf539cd5ea2e374a4242665a5ec3f3c01e7a3eefb8","Volumes":null,"WorkingDir":"","Entrypoint":["docker-entrypoint.sh"],"OnBuild":[],"Labels":{}},"created":"2016-10-10T23:04:00.821781828Z","docker_version":"1.12.1","history":[{"created":"2016-09-23T16:29:57.276868291Z","created_by":"/bin/sh -c #(nop) ADD file:d6ee3ba7a4d59b161917082cc7242c660c61bb3f3cc1549c7e2dfff2b0de7104 in / "},{"created":"2016-09-23T16:36:54.024611637Z","created_by":"/bin/sh -c apk add --no-cache \t\tca-certificates \t\tcurl \t\topenssl"},{"created":"2016-09-23T16:36:54.365914519Z","created_by":"/bin/sh -c #(nop)  ENV DOCKER_BUCKET=get.docker.com","empty_layer":true},{"created":"2016-09-23T16:36:54.662005049Z","created_by":"/bin/sh -c #(nop)  ENV DOCKER_VERSION=1.12.1","empty_layer":true},{"created":"2016-09-23T16:36:54.946033025Z","created_by":"/bin/sh -c #(nop)  ENV DOCKER_SHA256=05ceec7fd937e1416e5dce12b0b6e1c655907d349d52574319a1e875077ccb79","empty_layer":true},{"created":"2016-09-23T16:36:58.535084011Z","created_by":"/bin/sh -c set -x \t\u0026\u0026 curl -fSL \"https://${DOCKER_BUCKET}/builds/Linux/x86_64/docker-${DOCKER_VERSION}.tgz\" -o docker.tgz \t\u0026\u0026 echo \"${DOCKER_SHA256} *docker.tgz\" | sha256sum -c - \t\u0026\u0026 tar -xzvf docker.tgz \t\u0026\u0026 mv docker/* /usr/local/bin/ \t\u0026\u0026 rmdir docker \t\u0026\u0026 rm docker.tgz \t\u0026\u0026 docker -v"},{"created":"2016-10-10T23:04:00.334158993Z","created_by":"/bin/sh -c #(nop) COPY file:399605dc1850a60a586b5494ab538bad495fd6f94eabca0c5f8a26468ce6030f in /usr/local/bin/ "},{"created":"2016-10-10T23:04:00.577900192Z","created_by":"/bin/sh -c #(nop)  ENTRYPOINT [\"docker-entrypoint.sh\"]","empty_layer":true},{"created":"2016-10-10T23:04:00.821781828Z","created_by":"/bin/sh -c #(nop)  CMD [\"sh\"]","empty_layer":true}],"os":"linux","rootfs":{"type":"layers","diff_ids":["sha256:9007f5987db353ec398a223bc5a135c5a9601798ba20a1abba537ea2f8ac765f","sha256:1b06990ff0df8dad281fad7e6e4c5e91f32f8f8c095d6c74cf1e90a6f4407e28","sha256:9d12251ce74aac7619a83641ab72431a8d82e58bcd8a262c2bb0cdb280f1f3b5","sha256:17a7f292c2427adfc75c3a789bab8efec925dc38c5437bf83d2f528013ab80e2"]}}`,
			fail:   false,
		},
		{
			// fedora:23 from docker hub
			// both Entrypoint and Cmd can be nullable
			digest: "sha256:a20665eb1fe2912accb3d5dadaed360430df0d1aa46874875886947d61d3d4ee",
			config: `{"architecture":"amd64","author":"Patrick Uiterwijk \u003cpatrick@puiterwijk.org\u003e","config":{"Hostname":"8dfe43d80430","Domainname":"","User":"","AttachStdin":false,"AttachStdout":false,"AttachStderr":false,"Tty":false,"OpenStdin":false,"StdinOnce":false,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],"Cmd":null,"Image":"sha256:6986ae504bbf843512d680cc959484452034965db15f75ee8bdd1b107f61500b","Volumes":null,"WorkingDir":"","Entrypoint":null,"OnBuild":null,"Labels":{}},"container":"6249cd2c4b1d6b1bf05903364cbcb95781508994d6407c1564d494e748ea1b41","container_config":{"Hostname":"8dfe43d80430","Domainname":"","User":"","AttachStdin":false,"AttachStdout":false,"AttachStderr":false,"Tty":false,"OpenStdin":false,"StdinOnce":false,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],"Cmd":["/bin/sh","-c","#(nop) ADD file:293a6e463aa402bb8f80eb5cfc937f375cedc6843abaeb9eccfe3923bb3fc80b in /"],"Image":"sha256:6986ae504bbf843512d680cc959484452034965db15f75ee8bdd1b107f61500b","Volumes":null,"WorkingDir":"","Entrypoint":null,"OnBuild":null,"Labels":{}},"created":"2016-06-10T18:44:31.784795904Z","docker_version":"1.10.3","history":[{"created":"2016-06-10T18:44:03.360264073Z","author":"Patrick Uiterwijk \u003cpatrick@puiterwijk.org\u003e","created_by":"/bin/sh -c #(nop) MAINTAINER Patrick Uiterwijk \u003cpatrick@puiterwijk.org\u003e","empty_layer":true},{"created":"2016-06-10T18:44:31.784795904Z","author":"Patrick Uiterwijk \u003cpatrick@puiterwijk.org\u003e","created_by":"/bin/sh -c #(nop) ADD file:293a6e463aa402bb8f80eb5cfc937f375cedc6843abaeb9eccfe3923bb3fc80b in /"}],"os":"linux","rootfs":{"type":"layers","diff_ids":["sha256:d43f38155a799dc53d8fbb9f3bc11f51805f4027cd5c3d10b9823201cd5b9400"]}}`,
			fail:   false,
		},
	} {
		got := digest.FromString(tt.config)
		if tt.digest != got {
			t.Errorf("test %d: expected digest %s but got %s", i, tt.digest, got)
		}

		config := convertFormats(tt.config)
		r := strings.NewReader(config)
		err := schema.ValidatorMediaTypeImageConfig.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}
